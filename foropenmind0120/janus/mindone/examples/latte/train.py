"""
Latte training script
"""
import datetime
import logging
import os
import sys

import yaml
from args_train import parse_args
from data.dataset import get_dataset
from modules.text_encoders import initiate_clip_text_encoder
from omegaconf import OmegaConf
from pipelines import get_model_with_loss
from utils.model_utils import remove_pname_prefix

import mindspore as ms
from mindspore import Model, nn
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import TimeMonitor

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)

from diffusion import create_diffusion
from modules.autoencoder import SD_CONFIG, AutoencoderKL

from mindone.models.latte import Latte_models

# load training modules
from mindone.trainers.callback import EvalSaveCallback, OverflowMonitor, ProfilerCallback
from mindone.trainers.checkpoint import resume_train_network
from mindone.trainers.ema import EMA
from mindone.trainers.lr_schedule import create_scheduler
from mindone.trainers.optim import create_optimizer
from mindone.trainers.train_step import TrainOneStepWrapper
from mindone.utils.amp import auto_mixed_precision
from mindone.utils.env import init_train_env
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
    if args.ms_mode == ms.GRAPH_MODE:
        try:
            if args.jit_level in ["O0", "O1", "O2"]:
                ms.set_context(jit_config={"jit_level": args.jit_level})
                logger.info(f"set jit_level: {args.jit_level}.")
            else:
                logger.warning(
                    f"Unsupport jit_level: {args.jit_level}. The framework automatically selects the execution method"
                )
        except Exception:
            logger.warning(
                "The current jit_level is not suitable because current MindSpore version does not match,"
                "please ensure the MindSpore version >= ms2.3_0615."
            )
    set_logger(name="", output_dir=args.output_path, rank=rank_id, log_level=eval(args.log_level))

    # 2. model initiate and weight loading
    # 2.1 latte
    logger.info(f"{args.model_name}-{args.image_size}x{args.image_size} init")
    latent_size = args.image_size // 8
    latte_model = Latte_models[args.model_name](
        input_size=latent_size,
        num_classes=args.num_classes,
        block_kwargs={"enable_flash_attention": args.enable_flash_attention},
        condition=args.condition,
        num_frames=args.num_frames,
        use_recompute=args.use_recompute,
        patch_embedder=args.patch_embedder,
    )

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
    for param in latte_model.get_parameters():
        param.requires_grad = True

    # select dataset
    data_config = OmegaConf.load(args.data_config_file).data_config
    # set some data params from argument parser
    data_config.sample_size = args.image_size
    data_config.sample_n_frames = args.num_frames
    data_config.batch_size = args.train_batch_size
    train_with_embed = True if data_config.get("train_data_type", None) in ["numpy", "mindrecord"] else False

    if not train_with_embed:
        # 2.2 vae
        logger.info("vae init")
        vae = AutoencoderKL(
            SD_CONFIG,
            4,
            ckpt_path=args.vae_checkpoint,
            use_fp16=False,  # disable amp for vae . TODO: set by config file
        )
        vae = vae.set_train(False)
        for param in vae.get_parameters():  # freeze vae
            param.requires_grad = False
    else:
        vae = None

    if args.condition == "text" and not train_with_embed:
        text_encoder = initiate_clip_text_encoder(
            use_fp16=True,  # TODO: set by config file
            ckpt_path=args.clip_checkpoint,
            trainable=False,
        )
        tokenizer = text_encoder.tokenizer
    else:
        text_encoder, tokenizer = None, None

    dataset = get_dataset(
        args.dataset_name,
        data_config,
        tokenizer=tokenizer,
        device_num=device_num,
        rank_id=rank_id,
    )
    dataset_size = dataset.get_dataset_size()

    diffusion = create_diffusion(timestep_respacing="")
    latent_diffusion_with_loss = get_model_with_loss(args.condition)(
        latte_model,
        diffusion,
        vae,
        args.sd_scale_factor,
        args.condition,
        text_encoder=text_encoder,
        cond_stage_trainable=False,
        train_with_embed=train_with_embed,
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
                f"Num params: {num_params:,} (dit: {num_params_latte:,}, vae: {num_params_vae:,})",
                f"Num trainable params: {num_params_trainable:,}",
                f"Use model dtype: {model_dtype}",
                f"Learning rate: {args.start_learning_rate}",
                f"Batch size: {args.train_batch_size}",
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
