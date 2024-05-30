"""
STDiT training script
"""
import datetime
import logging
import math
import os
import sys
from typing import Tuple

import yaml

import mindspore as ms
from mindspore import Model, nn
from mindspore.communication.management import get_group_size, get_rank, init
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import TimeMonitor

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.insert(0, mindone_lib_path)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))
from args_train import parse_args
from opensora.models.stdit import STDiT2_XL_2, STDiT_XL_2
from opensora.models.vae.vae import SD_CONFIG, AutoencoderKL
from opensora.pipelines import DiffusionWithLoss
from opensora.schedulers.iddpm import create_diffusion
from opensora.utils.amp import auto_mixed_precision

from mindone.trainers.callback import EvalSaveCallback, OverflowMonitor, ProfilerCallbackEpoch
from mindone.trainers.checkpoint import resume_train_network
from mindone.trainers.ema import EMA
from mindone.trainers.lr_schedule import create_scheduler
from mindone.trainers.optim import create_optimizer
from mindone.trainers.train_step import TrainOneStepWrapper
from mindone.utils.logger import set_logger
from mindone.utils.params import count_params
from mindone.utils.seed import set_random_seed

# from opensora.utils.model_utils import WHITELIST_OPS


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
    global_bf16: bool = False,
    debug: bool = False,
) -> Tuple[int, int]:
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

    if debug and mode == ms.GRAPH_MODE:  # force PyNative mode when debugging
        logger.warning("Debug mode is on, switching execution mode to PyNative.")
        mode = ms.PYNATIVE_MODE

    if max_device_memory is not None:
        ms.set_context(max_device_memory=max_device_memory)

    if distributed:
        ms.set_context(
            mode=mode,
            device_target=device_target,
        )
        if parallel_mode == "optim":
            print("use optim parallel")
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
            pynative_synchronize=debug,
        )

    if enable_dvm:
        print("enable dvm")
        # FIXME: the graph_kernel_flags settting is a temp solution to fix dvm loss convergence in ms2.3-rc2. Refine it for future ms version.
        ms.set_context(enable_graph_kernel=True, graph_kernel_flags="--disable_cluster_ops=Pow,Select")

    if global_bf16:
        ms.set_context(ascend_config={"precision_mode": "allow_mix_precision_bf16"})

    return rank_id, device_num


def set_all_reduce_fusion(
    params,
    split_num: int = 7,
    distributed: bool = False,
    parallel_mode: str = "data",
) -> None:
    """Set allreduce fusion strategy by split_num."""

    if distributed and parallel_mode == "data":
        all_params_num = len(params)
        step = all_params_num // split_num
        split_list = [i * step for i in range(1, split_num)]
        split_list.append(all_params_num - 1)
        logger.info(f"Distribute config set: dall_params_num: {all_params_num}, set all_reduce_fusion: {split_list}")
        ms.set_auto_parallel_context(all_reduce_fusion_config=split_list)


def main(args):
    if args.add_datetime:
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
        global_bf16=args.global_bf16,
        debug=args.debug,
    )
    set_logger(name="", output_dir=args.output_path, rank=rank_id, log_level=eval(args.log_level))

    # 2. model initiate and weight loading
    # 2.1 stdit
    VAE_T_COMPRESS = 1
    VAE_S_COMPRESS = 8
    VAE_Z_CH = SD_CONFIG["z_channels"]
    img_h, img_w = args.image_size if isinstance(args.image_size, list) else (args.image_size, args.image_size)
    if args.model_version == "v1":
        assert img_h == img_w, "OpenSora v1 support square images only."

    input_size = (
        args.num_frames // VAE_T_COMPRESS,
        img_h // VAE_S_COMPRESS,
        img_w // VAE_S_COMPRESS,
    )
    model_extra_args = dict(
        input_size=input_size,
        in_channels=VAE_Z_CH,
        model_max_length=args.model_max_length,
        patchify_conv3d_replace="conv2d",  # for Ascend
        enable_flashattn=args.enable_flash_attention,
        use_recompute=args.use_recompute,
    )
    if args.model_version == "v1":
        model_extra_args.update(
            {
                "space_scale": args.space_scale,  # 0.5 for 256x256. 1. for 512
                "time_scale": args.time_scale,
                "num_recompute_blocks": args.num_recompute_blocks,
            }
        )
        logger.info(f"STDiT input size: {input_size}")
        latte_model = STDiT_XL_2(**model_extra_args)
    elif args.model_version == "v1.1":
        model_extra_args.update(
            {
                "input_sq_size": 512,
                "qk_norm": True,
            }
        )
        logger.info(f"STDiT2 input size: {input_size if args.bucket_config is None else 'Variable'}")
        latte_model = STDiT2_XL_2(**model_extra_args)
    else:
        raise ValueError(f"Unknown model version: {args.model_version}")

    # mixed precision
    dtype_map = {"fp16": ms.float16, "bf16": ms.bfloat16}
    if args.dtype in ["fp16", "bf16"]:
        if not args.global_bf16:
            latte_model = auto_mixed_precision(
                latte_model,
                amp_level=args.amp_level,
                dtype=dtype_map[args.dtype],
                # custom_fp32_cells=WHITELIST_OPS
            )
    # load checkpoint
    if len(args.pretrained_model_path) > 0:
        logger.info(f"Loading ckpt {args.pretrained_model_path}...")
        latte_model.load_from_checkpoint(args.pretrained_model_path)
    else:
        logger.info("Use random initialization for Latte")
    latte_model.set_train(True)

    # 2.2 vae
    # TODO: use mindone/models/autoencoders in future
    logger.info("vae init")
    train_with_vae_latent = args.vae_latent_folder is not None and os.path.exists(args.vae_latent_folder)
    if not train_with_vae_latent:
        vae = AutoencoderKL(
            SD_CONFIG,
            VAE_Z_CH,
            ckpt_path=args.vae_checkpoint,
        )
        vae = vae.set_train(False)
        for param in vae.get_parameters():
            param.requires_grad = False
            if args.vae_param_dtype in ["fp16", "bf16"]:
                # filter out norm
                if "norm" not in param.name:
                    param.set_dtype(dtype_map[args.vae_param_dtype])
        if args.vae_dtype in ["fp16", "bf16"]:
            vae = auto_mixed_precision(
                vae,
                amp_level=args.vae_amp_level,
                dtype=dtype_map[args.vae_dtype],
                custom_fp32_cells=[nn.GroupNorm] if args.vae_keep_gn_fp32 else [],
            )
    else:
        vae = None

    # 2.3 ldm with loss
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
        micro_batch_size=args.vae_micro_batch_size,
    )

    # 3. create dataset
    dataloader = None
    if args.model_version == "v1":
        from opensora.datasets.t2v_dataset import create_dataloader

        ds_config = dict(
            csv_path=args.csv_path,
            video_folder=args.video_folder,
            text_emb_folder=args.text_embed_folder,
            return_text_emb=True,
            vae_latent_folder=args.vae_latent_folder,
            return_vae_latent=train_with_vae_latent,
            vae_scale_factor=args.sd_scale_factor,
            sample_size=img_w,  # img_w == img_h
            sample_stride=args.frame_stride,
            sample_n_frames=args.num_frames,
            tokenizer=None,
            video_column=args.video_column,
            caption_column=args.caption_column,
            disable_flip=args.disable_flip,
        )
        dataloader = create_dataloader(
            ds_config,
            batch_size=args.batch_size,
            shuffle=True,
            device_num=device_num,
            rank_id=rank_id,
            num_parallel_workers=args.num_parallel_workers,
            max_rowsize=args.max_rowsize,
        )
    elif args.model_version == "v1.1":
        from opensora.datasets.bucket import Bucket, bucket_split_function
        from opensora.datasets.mask_generator import MaskGenerator
        from opensora.datasets.video_dataset_refactored import VideoDatasetRefactored

        from mindone.data import create_dataloader

        mask_gen = MaskGenerator(args.mask_ratios)
        buckets = Bucket(args.bucket_config) if args.bucket_config is not None else None

        dataset = VideoDatasetRefactored(
            csv_path=args.csv_path,
            video_folder=args.video_folder,
            text_emb_folder=args.text_embed_folder,
            vae_latent_folder=args.vae_latent_folder,
            vae_scale_factor=args.sd_scale_factor,
            sample_n_frames=args.num_frames,
            sample_stride=args.frame_stride,
            frames_mask_generator=mask_gen,
            buckets=buckets,
            output_columns=["video", "caption", "mask", "fps", "num_frames", "frames_mask"],
        )

        dataloader = create_dataloader(
            dataset,
            batch_size=args.batch_size if buckets is None else 0,  # Turn off batching if using buckets
            transforms=dataset.train_transforms(
                target_size=(img_h, img_w), tokenizer=None  # Tokenizer isn't supported yet
            ),
            shuffle=True,
            device_num=device_num,
            rank_id=rank_id,
            num_workers=args.num_parallel_workers,
            python_multiprocessing=args.data_multiprocessing,
            max_rowsize=args.max_rowsize,
            debug=args.debug,
            # Sort output columns to match DiffusionWithLoss input
            project_columns=["video", "caption", "mask", "frames_mask", "num_frames", "height", "width", "fps", "ar"],
        )

        if buckets is not None:
            hash_func, bucket_boundaries, bucket_batch_sizes = bucket_split_function(buckets)
            dataloader = dataloader.bucket_batch_by_length(
                ["video"], bucket_boundaries, bucket_batch_sizes, element_length_function=hash_func, drop_remainder=True
            )

    dataset_size = dataloader.get_dataset_size()

    # compute total steps and data epochs (in unit of data sink size)
    if args.train_steps == -1:
        assert args.epochs != -1
        total_train_steps = args.epochs * dataset_size
    else:
        total_train_steps = args.train_steps

    if args.dataset_sink_mode and args.sink_size != -1:
        steps_per_sink = args.sink_size
    else:
        steps_per_sink = dataset_size
    sink_epochs = math.ceil(total_train_steps / steps_per_sink)

    if args.ckpt_save_steps == -1:
        ckpt_save_interval = args.ckpt_save_interval
        step_mode = False
    else:
        step_mode = not args.dataset_sink_mode
        if not args.dataset_sink_mode:
            ckpt_save_interval = args.ckpt_save_steps
        else:
            # still need to count interval in sink epochs
            ckpt_save_interval = max(1, args.ckpt_save_steps // steps_per_sink)
            if args.ckpt_save_steps % steps_per_sink != 0:
                logger.warning(
                    f"`ckpt_save_steps` must be times of sink size or dataset_size under dataset sink mode."
                    f"Checkpoint will be saved every {ckpt_save_interval * steps_per_sink} steps."
                )
    step_mode = step_mode if args.step_mode is None else args.step_mode

    logger.info(f"train_steps: {total_train_steps}, train_epochs: {args.epochs}, sink_size: {args.sink_size}")
    logger.info(f"total train steps: {total_train_steps}, sink epochs: {sink_epochs}")
    logger.info(
        "ckpt_save_interval: {} {}".format(
            ckpt_save_interval, "steps" if (not args.dataset_sink_mode and step_mode) else "sink epochs"
        )
    )

    # 4. build training utils: lr, optim, callbacks, trainer
    # build learning rate scheduler
    if not args.decay_steps:
        args.decay_steps = total_train_steps - args.warmup_steps  # fix lr scheduling
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
        total_steps=total_train_steps,
    )

    set_all_reduce_fusion(
        latent_diffusion_with_loss.trainable_params(),
        split_num=7,
        distributed=args.use_parallel,
        parallel_mode=args.parallel_mode,
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

    if args.global_bf16:
        model = Model(net_with_grads, amp_level="O0")
    else:
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
            step_mode=step_mode,
            use_step_unit=(args.ckpt_save_steps != -1),
            ckpt_save_interval=ckpt_save_interval,
            log_interval=args.log_interval,
            start_epoch=start_epoch,
            model_name="STDiT",
            record_lr=False,
        )
        callback.append(save_cb)
        if args.profile:
            callback.append(ProfilerCallbackEpoch(2, 3, "./profile_data"))

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
                f"Use model dtype: {args.dtype}",
                f"Learning rate: {args.start_learning_rate}",
                f"Batch size: {args.batch_size if args.bucket_config is None else 'Variable'}",
                f"Image size: {(img_h, img_w) if args.bucket_config is None else 'Variable'}",
                f"Frames: {args.num_frames if args.bucket_config is None else 'Variable'}",
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
        sink_epochs,
        dataloader,
        callbacks=callback,
        dataset_sink_mode=args.dataset_sink_mode,
        sink_size=args.sink_size,
        initial_epoch=start_epoch,
    )


if __name__ == "__main__":
    logger.debug("process id:", os.getpid())
    args = parse_args()
    main(args)
