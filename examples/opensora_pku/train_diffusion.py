"""
STDiT training script
"""
import datetime
import logging
import os
import sys
from typing import Tuple

import yaml
from args_train import parse_args
from omegaconf import OmegaConf

import mindspore as ms
from mindspore import Model, nn
from mindspore.communication.management import get_group_size, get_rank, init
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import TimeMonitor

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)
from opensora.data.t2v_dataset import create_dataloader
from opensora.diffusion import create_diffusion
from opensora.models.diffusion.latte_t2v import Attention, Latte_models, LayerNorm
from opensora.pipelines import DiffusionWithLoss

from mindone.trainers.callback import EvalSaveCallback, OverflowMonitor, ProfilerCallback
from mindone.trainers.checkpoint import resume_train_network
from mindone.trainers.ema import EMA
from mindone.trainers.lr_schedule import create_scheduler
from mindone.trainers.optim import create_optimizer
from mindone.trainers.train_step import TrainOneStepWrapper
from mindone.utils.amp import auto_mixed_precision
from mindone.utils.config import instantiate_from_config
from mindone.utils.logger import set_logger
from mindone.utils.params import count_params
from mindone.utils.seed import set_random_seed

os.environ["HCCL_CONNECT_TIMEOUT"] = "6000"
os.environ["MS_ASCEND_CHECK_OVERFLOW_MODE"] = "INFNAN_MODE"

logger = logging.getLogger(__name__)


def init_env(
    mode: int = ms.GRAPH_MODE,
    seed: int = 42,
    distributed: bool = False,
    max_device_memory: str = None,
    device_target: str = "Ascend",
    parallel_mode: str = "data",
    enable_dvm: bool = False,
) -> Tuple[int, int, int]:
    """
    Initialize MindSpore environment.

    Args:
        mode: MindSpore execution mode. Default is 0 (ms.GRAPH_MODE).
        seed: The seed value for reproducibility. Default is 42.
        distributed: Whether to enable distributed training. Default is False.
    Returns:
        A tuple containing the device ID, rank ID and number of devices.
    """
    set_random_seed(seed)

    if max_device_memory is not None:
        ms.set_context(max_device_memory=max_device_memory)

    if distributed:
        ms.set_context(
            mode=mode,
            device_target=device_target,
            ascend_config={"precision_mode": "allow_fp32_to_fp16"},  # ms2.2.23 parallel needs
            # ascend_config={"precision_mode": "must_keep_origin_dtype"},  # TODO: tune
        )
        if parallel_mode == "optim":
            print("D--: use optim parallel")
            ms.set_auto_parallel_context(
                parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL,
                enable_parallel_optimizer=True,
            )
            init()
            device_num = get_group_size()
            rank_id = get_rank()
        else:
            init()
            device_num = get_group_size()
            rank_id = get_rank()
            logger.debug(f"rank_id: {rank_id}, device_num: {device_num}")
            ms.reset_auto_parallel_context()

            ms.set_auto_parallel_context(
                parallel_mode=ms.ParallelMode.DATA_PARALLEL,
                gradients_mean=True,
                device_num=device_num,
            )

        var_info = ["device_num", "rank_id", "device_num / 8", "rank_id / 8"]
        var_value = [device_num, rank_id, int(device_num / 8), int(rank_id / 8)]
        logger.info(dict(zip(var_info, var_value)))

    else:
        device_num = 1
        rank_id = 0
        ms.set_context(
            mode=mode,
            device_target=device_target,
            ascend_config={"precision_mode": "allow_fp32_to_fp16"},  # TODO: tune for better precision
        )

    if enable_dvm:
        print("D--: enable dvm")
        ms.set_context(enable_graph_kernel=True)

    return rank_id, device_num


def main(args):
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    args.output_path = os.path.join(args.output_path, time_str)

    # 1. init
    rank_id, device_num = init_env(
        args.mode,
        seed=args.seed,
        distributed=args.use_parallel,
        device_target=args.device_target,
        max_device_memory=args.max_device_memory,
        parallel_mode=args.parallel_mode,
        enable_dvm=args.enable_dvm,
    )
    set_logger(name="", output_dir=args.output_path, rank=rank_id, log_level=eval(args.log_level))

    # 2.2 vae
    logger.info("vae init")
    ae_config = OmegaConf.load(args.vae_config)
    vae = instantiate_from_config(ae_config.generator)
    vae.init_from_ckpt(args.vae_checkpoint)
    vae.set_train(False)

    vae = auto_mixed_precision(vae, amp_level="O2", dtype=ms.float16)
    logger.info("Use amp level O2 for causal 3D VAE.")

    for param in vae.get_parameters():  # freeze vae
        param.requires_grad = False

    # latent_size = (image_size // ae_stride_config[args.ae][1], image_size // ae_stride_config[args.ae][2])
    latent_size = args.image_size // 8
    vae.latent_size = (latent_size, latent_size)

    logger.info(f"Init Latte T2V model: {args.model_version}")
    video_length = args.num_frames // ae_config.generator.params.ddconfig.time_compress + 1
    latte_model = Latte_models[args.model_version](
        in_channels=ae_config.generator.params.ddconfig.z_channels,
        output_cahnnels=ae_config.generator.params.ddconfig.z_channels * 2,
        attention_bias=True,
        sample_size=latent_size,
        num_vector_embeds=None,
        activation_fn="gelu-approximate",
        num_embeds_ada_norm=1000,
        use_linear_projection=False,
        only_cross_attention=False,
        double_self_attention=False,
        upcast_attention=False,
        norm_elementwise_affine=False,
        norm_eps=1e-6,
        attention_type="default",
        video_length=video_length,
        enable_flash_attention=args.enable_flash_attention,
        use_recompute=args.use_recompute,
    )

    # mixed precision
    if args.dtype == "fp32":
        model_dtype = ms.float32
    else:
        model_dtype = {"fp16": ms.float16, "bf16": ms.bfloat16}[args.dtype]
        latte_model = auto_mixed_precision(
            latte_model,
            amp_level=args.amp_level,
            dtype=model_dtype,
            custom_fp32_cells=[LayerNorm, Attention, nn.SiLU, nn.GELU],
        )
    # load checkpoint
    if len(args.pretrained_model_path) > 0:
        logger.info(f"Loading ckpt {args.pretrained_model_path}...")
        latte_model.load_from_checkpoint(args.pretrained_model_path)
    else:
        logger.info("Use random initialization for Latte")
    latte_model.set_train(True)

    # 2.3 ldm with loss
    train_with_vae_latent = args.vae_latent_folder is not None and os.path.exists(args.vae_latent_folder)
    logger.info(f"Train with vae latent cache: {train_with_vae_latent}")
    diffusion = create_diffusion(timestep_respacing="")
    latent_diffusion_with_loss = DiffusionWithLoss(
        latte_model,
        diffusion,
        vae=vae,
        scale_factor=args.sd_scale_factor,
        condition="text",
        text_encoder=None,
        cond_stage_trainable=False,
        text_emb_cached=True,
        video_emb_cached=train_with_vae_latent,
    )

    # 3. create dataset
    ds_config = dict(
        csv_path=args.csv_path,
        video_folder=args.video_folder,
        text_emb_folder=args.text_embed_folder,
        return_text_emb=True,
        vae_latent_folder=args.vae_latent_folder,
        return_vae_latent=train_with_vae_latent,
        vae_scale_factor=args.sd_scale_factor,
        sample_size=args.image_size,
        sample_stride=args.frame_stride,
        sample_n_frames=args.num_frames,
        tokenizer=None,
        video_column=args.video_column,
        caption_column=args.caption_column,
        disable_flip=args.disable_flip,
    )
    dataset = create_dataloader(
        ds_config,
        batch_size=args.batch_size,
        shuffle=True,
        device_num=device_num,
        rank_id=rank_id,
        num_parallel_workers=args.num_parallel_workers,
        max_rowsize=args.max_rowsize,
    )
    dataset_size = dataset.get_dataset_size()

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
            network=latent_diffusion_with_loss.network,
            rank_id=rank_id,
            ckpt_save_dir=ckpt_dir,
            ema=ema,
            ckpt_save_policy="latest_k",
            ckpt_max_keep=args.ckpt_max_keep,
            step_mode=args.step_mode,
            ckpt_save_interval=args.ckpt_save_interval,
            log_interval=args.log_interval,
            start_epoch=start_epoch,
            model_name="STDiT",
            record_lr=False,
        )
        callback.append(save_cb)
        if args.profile:
            callback.append(ProfilerCallback())

    # 5. log and save config
    if rank_id == 0:
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
