"""
Latte training script
"""
import datetime
import logging
import os
import sys
from omegaconf import OmegaConf
import yaml

import mindspore as ms
from mindspore import Model, nn
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import TimeMonitor

from args_train import parse_args
from opensora.data.dataset import create_dataloader
from opensora.pipelines import DiffusionWithLoss 
from opensora.utils.model_utils import remove_pname_prefix
from opensora.diffusion import create_diffusion

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)

from mindone.env import init_train_env
from mindone.models.stdit import STDiT_XL_2

# load training modules
from mindone.trainers.callback import EvalSaveCallback, OverflowMonitor, ProfilerCallback
from mindone.trainers.checkpoint import resume_train_network
from mindone.trainers.ema import EMA
from mindone.trainers.lr_schedule import create_scheduler
from mindone.trainers.optim import create_optimizer
from mindone.trainers.train_step import TrainOneStepWrapper
from mindone.utils.amp import auto_mixed_precision
from mindone.utils.logger import set_logger
from mindone.utils.params import count_params

os.environ["HCCL_CONNECT_TIMEOUT"] = "6000"

logger = logging.getLogger(__name__)


def main(args):
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    args.output_path = os.path.join(args.output_path, time_str)

    # 1. init
    _, rank_id, device_num = init_train_env(
        args.mode,
        seed=args.seed,
        distributed=args.use_parallel,
        device_target=args.device_target,
        max_device_memory=args.max_device_memory,
        ascend_config=None if args.precision_mode is None else {"precision_mode": args.precision_mode},
    )
    set_logger(name="", output_dir=args.output_path, rank=rank_id, log_level=eval(args.log_level))

    # 2. model initiate and weight loading
    # 2.1 latte
    vae_t_compress = 1
    vae_s_compress = 8
    vae_out_channels = 4

    text_emb_dim = 4096
    max_tokens = 120

    input_size = (args.num_frames//vae_t_compress, args.image_size//vae_s_compress,  args.image_size//vae_s_compress)
    
    # FIXME: set this parameter by config file
    model_extra_args = dict(
        input_size=input_size,
        in_channels=vae_out_channels,
        caption_channels=text_emb_dim,
        model_max_length=max_tokens,
        space_scale=0.5,  # 0.5 for 256x256. diff for 512. # TODO: align to torch
        time_scale=1.0,
        )
    latte_model = STDiT_XL_2(**model_extra_args)

    if args.dtype == "fp16":
        model_dtype = ms.float16
        latte_model = auto_mixed_precision(latte_model, amp_level="O2", dtype=model_dtype)
    elif args.dtype == "bf16":
        model_dtype = ms.bfloat16
        latte_model = auto_mixed_precision(latte_model, amp_level="O2", dtype=model_dtype)
    else:
        model_dtype = ms.float32

    if len(args.pretrained_model_path) > 0:
        param_dict = ms.load_checkpoint(args.pretrained_model_path)
        logger.info(f"Loading ckpt {args.pretrained_model_path} into Latte...")
        # in case a save ckpt with "network." prefix, removing it before loading
        param_dict = remove_pname_prefix(param_dict, prefix="network.")
        latte_model.load_params_from_ckpt(param_dict)
    else:
        logger.info("Use random initialization for Latte")
    # set train
    latte_model.set_train(True)
    # TODO: tell ddd it's risk. non-trainable params like PE, norm should not be set requires_grad.
    # for param in latte_model.get_parameters():
    #    param.requires_grad = True
    
    vae = None
    # TODO: set vae and t5 non-trainable

    # select dataset
    ds_config = dict(
        sample_size=args.image_size,
        sample_n_frames=args.num_frames,
        space_compress=vae_s_compress,
        time_compress=vae_t_compress,
        vae_embed_dim=vae_out_channels,
        text_embed_dim=text_emb_dim,
        num_tokens=max_tokens,
        )
    dataset = create_dataloader(ds_config, batch_size=args.batch_size, shuffle=True, device_num=1, rank_id=0)
    dataset_size = dataset.get_dataset_size()

    diffusion = create_diffusion(timestep_respacing="")
    latent_diffusion_with_loss = DiffusionWithLoss(
        latte_model,
        diffusion,
        vae=None,
        scale_factor=args.sd_scale_factor,
        condition='text',
        text_encoder=None,
        cond_stage_trainable=False,
        train_with_embed=True,
    )

    # 4. build training utils: lr, optim, callbacks, trainer
    # build learning rate scheduler
    if not args.decay_steps:
        args.decay_steps = args.epochs * dataset_size - args.warmup_steps  # fix lr scheduling
        if args.decay_steps <= 0:
            logger.warning(
                f"decay_steps is {args.decay_steps}, please check epochs, dataset_size and warmup_steps. "
                f"Will force decay_steps to be set to 1."
            )
            args.decay_steps = 1

    lr = create_scheduler(
        steps_per_epoch=dataset_size,
        name=args.scheduler,
        lr=args.start_learning_rate,
        end_lr=args.end_learning_rate,
        warmup_steps=args.warmup_steps,
        decay_steps=args.decay_steps,
        num_epochs=args.epochs,
    )

    # build optimizer
    optimizer = create_optimizer(
        latent_diffusion_with_loss.trainable_params(),
        name=args.optim,
        betas=args.betas,
        eps=args.optim_eps,
        group_strategy=args.group_strategy,
        weight_decay=args.weight_decay,
        lr=lr,
    )

    if args.loss_scaler_type == "dynamic":
        loss_scaler = DynamicLossScaleUpdateCell(
            loss_scale_value=args.init_loss_scale, scale_factor=args.loss_scale_factor, scale_window=args.scale_window
        )
    elif args.loss_scaler_type == "static":
        loss_scaler = nn.FixedLossScaleUpdateCell(args.init_loss_scale)
    else:
        raise ValueError

    # resume ckpt
    ckpt_dir = os.path.join(args.output_path, "ckpt")
    start_epoch = 0
    if args.resume:
        resume_ckpt = os.path.join(ckpt_dir, "train_resume.ckpt") if isinstance(args.resume, bool) else args.resume

        start_epoch, loss_scale, cur_iter, last_overflow_iter = resume_train_network(
            latte_model, optimizer, resume_ckpt
        )
        loss_scaler.loss_scale_value = loss_scale
        loss_scaler.cur_iter = cur_iter
        loss_scaler.last_overflow_iter = last_overflow_iter

    # trainer (standalone and distributed)
    ema = (
        EMA(
            latent_diffusion_with_loss.network,
            ema_decay=0.9999,
        )
        if args.use_ema
        else None
    )

    net_with_grads = TrainOneStepWrapper(
        latent_diffusion_with_loss,
        optimizer=optimizer,
        scale_sense=loss_scaler,
        drop_overflow_update=args.drop_overflow_update,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        clip_grad=args.clip_grad,
        clip_norm=args.max_grad_norm,
        ema=ema,
    )

    # import pdb
    # pdb.set_trace()

    model = Model(net_with_grads)
    # callbacks
    callback = [TimeMonitor(args.log_interval)]
    ofm_cb = OverflowMonitor()
    callback.append(ofm_cb)

    if rank_id == 0:
        save_cb = EvalSaveCallback(
            network=latent_diffusion_with_loss.network,  # save latte only
            rank_id=rank_id,
            ckpt_save_dir=ckpt_dir,
            ema=ema,
            ckpt_save_policy="latest_k",
            ckpt_max_keep=args.ckpt_max_keep,
            step_mode=args.step_mode,
            ckpt_save_interval=args.ckpt_save_interval,
            log_interval=args.log_interval,
            start_epoch=start_epoch,
            model_name="Latte",
            record_lr=False,  # TODO: check LR retrival for new MS on 910b
        )
        callback.append(save_cb)
        if args.profile:
            callback.append(ProfilerCallback())
    
    # FIXME: debug
    for param in latte_model.get_parameters():
        if param.requires_grad:
            print(param.name, tuple(param.shape))

    # 5. log and save config
    if rank_id == 0:
        # 4. print key info
        if vae is not None:
            num_params_vae, num_params_vae_trainable = count_params(vae)
        else:
            num_params_vae, num_params_vae_trainable = 0, 0
        num_params_latte, num_params_latte_trainable = count_params(latte_model)
        num_params = num_params_vae + num_params_latte
        num_params_trainable = num_params_vae_trainable + num_params_latte_trainable
        key_info = "Key Settings:\n" + "=" * 50 + "\n"
        key_info += "\n".join(
            [
                f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.mode}",
                f"Distributed mode: {args.use_parallel}",
                f"Num params: {num_params:,} (latte: {num_params_latte:,}, vae: {num_params_vae:,})",
                f"Num trainable params: {num_params_trainable:,}",
                f"Use model dtype: {model_dtype}",
                f"Learning rate: {args.start_learning_rate}",
                f"Batch size: {args.batch_size}",
                f"Image size: {args.image_size}",
                f"Frames: {args.num_frames}",
                f"Weight decay: {args.weight_decay}",
                f"Grad accumulation steps: {args.gradient_accumulation_steps}",
                f"Num epochs: {args.epochs}",
                f"Loss scaler: {args.loss_scaler_type}",
                f"Init loss scale: {args.init_loss_scale}",
                f"Grad clipping: {args.clip_grad}",
                f"Max grad norm: {args.max_grad_norm}",
                f"EMA: {args.use_ema}",
                f"Enable flash attention: {args.enable_flash_attention}",
                f"Use recompute: {args.use_recompute}",
                f"Dataset sink: {args.dataset_sink_mode}",
            ]
        )
        key_info += "\n" + "=" * 50
        logger.info(key_info)

        logger.info("Start training...")

        with open(os.path.join(args.output_path, "args.yaml"), "w") as f:
            yaml.safe_dump(vars(args), stream=f, default_flow_style=False, sort_keys=False)

    # 6. train
    model.train(
        args.epochs,
        dataset,
        callbacks=callback,
        dataset_sink_mode=args.dataset_sink_mode,
        sink_size=args.sink_size,
        initial_epoch=start_epoch,
    )


if __name__ == "__main__":
    logger.debug("process id:", os.getpid())
    args = parse_args()
    main(args)
