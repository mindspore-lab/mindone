"""
STDiT training script
"""
import datetime
import logging
import math
import os
import sys
import time
from typing import Optional, Tuple

import yaml

import mindspore as ms
from mindspore import nn
from mindspore._c_expression import reset_op_id
from mindspore.communication.management import get_group_size, get_rank, init
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import TimeMonitor

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.insert(0, mindone_lib_path)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))
from args_train import parse_args
from opensora.acceleration.parallel_states import create_parallel_group
from opensora.datasets.aspect import ASPECT_RATIOS, get_image_size
from opensora.models.layers.operation_selector import set_dynamic_mode
from opensora.models.stdit import STDiT2_XL_2, STDiT3_XL_2, STDiT3_XL_2_DSP, STDiT_XL_2
from opensora.models.vae.vae import SD_CONFIG, OpenSoraVAE_V1_2, VideoAutoencoderKL
from opensora.pipelines import (
    DiffusionWithLoss,
    DiffusionWithLossFiTLike,
    RFlowDiffusionWithLoss,
    RFlowEvalDiffusionWithLoss,
)
from opensora.schedulers.iddpm import create_diffusion
from opensora.utils.amp import auto_mixed_precision
from opensora.utils.callbacks import EMAEvalSwapCallback, PerfRecorderCallback
from opensora.utils.ema import EMA, save_ema_ckpts
from opensora.utils.metrics import BucketLoss
from opensora.utils.model_utils import WHITELIST_OPS, Model
from opensora.utils.resume import flush_from_cache, get_resume_ckpt, get_resume_states, resume_train_net, save_train_net

from mindone.trainers.callback import EvalSaveCallback, OverflowMonitor, ProfilerCallbackEpoch, StopAtStepCallback
from mindone.trainers.checkpoint import CheckpointManager
from mindone.trainers.lr_schedule import create_scheduler
from mindone.trainers.optim import create_optimizer
from mindone.trainers.recorder import PerfRecorder
from mindone.trainers.zero import prepare_train_network
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
    jit_level: str = "O0",
    global_bf16: bool = False,
    dynamic_shape: bool = False,
    enable_sequence_parallelism: bool = False,
    sequence_parallel_shards: int = 1,
    save_graphs: int = False,
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

    if enable_sequence_parallelism:
        if parallel_mode != "data" or not distributed:
            raise ValueError(
                "sequence parallel can only be used in data parallel mode, "
                f"but get parallel_mode=`{parallel_mode}` with distributed=`{distributed}`."
            )

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

            if enable_sequence_parallelism:
                create_parallel_group(sequence_parallel_shards)
                ms.set_auto_parallel_context(enable_alltoall=True)

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

    try:
        if jit_level in ["O0", "O1", "O2"]:
            ms.set_context(jit_config={"jit_level": jit_level})
        else:
            logger.warning(
                f"Unsupported jit_level: {jit_level}. The framework will automatically select the execution mode."
            )
    except Exception:
        logger.warning(
            "The current jit_level is not suitable because current MindSpore version or mode does not match,"
            "please ensure the MindSpore version >= ms2.3_0615, and use GRAPH_MODE."
        )

    if global_bf16:
        ms.set_context(ascend_config={"precision_mode": "allow_mix_precision_bf16"})

    if dynamic_shape:
        logger.info("Dynamic shape mode enabled, repeat_interleave/split/chunk will be called from mint module")
        set_dynamic_mode(True)

    if save_graphs > 0:
        ms.set_context(save_graphs=args.save_graphs)

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


def initialize_dataset(
    args,
    csv_path,
    video_folder,
    text_embed_folder,
    vae_latent_folder,
    train_with_vae_latent,
    batch_size,
    img_h,
    img_w,
    latte_model,
    vae,
    bucket_config: Optional[dict] = None,
    validation: bool = False,
    device_num: int = 1,
    rank_id: int = 0,
):
    if args.model_version == "v1":
        from opensora.datasets.t2v_dataset import create_dataloader

        ds_config = dict(
            csv_path=csv_path,
            video_folder=video_folder,
            text_emb_folder=text_embed_folder,
            return_text_emb=True,
            vae_latent_folder=vae_latent_folder,
            return_vae_latent=train_with_vae_latent,
            vae_scale_factor=args.sd_scale_factor,
            sample_size=img_w,  # img_w == img_h
            sample_stride=args.frame_stride,
            sample_n_frames=args.num_frames,
            tokenizer=None,
            video_column=args.video_column,
            caption_column=args.caption_column,
            disable_flip=args.disable_flip,
            filter_data=args.filter_data,
        )
        dataloader = create_dataloader(
            ds_config,
            batch_size=batch_size,
            shuffle=True,
            device_num=device_num,
            rank_id=rank_id,
            num_parallel_workers=args.num_parallel_workers,
            max_rowsize=args.max_rowsize,
        )
        num_src_samples = batch_size * dataloader.get_dataset_size() * (device_num if device_num is not None else 1)
    else:
        from opensora.datasets.bucket import Bucket, bucket_split_function
        from opensora.datasets.mask_generator import MaskGenerator
        from opensora.datasets.video_dataset_refactored import VideoDatasetRefactored, create_dataloader

        # from mindone.data import create_dataloader
        if validation:
            mask_gen = MaskGenerator({"identity": 1.0})
            all_buckets, individual_buckets = None, [None]
            if bucket_config is not None:
                all_buckets = Bucket(bucket_config)
                # Build a new bucket for each resolution and number of frames for the validation stage
                individual_buckets = [
                    Bucket({res: {num_frames: [1.0, bucket_config[res][num_frames][1]]}})
                    for res in bucket_config.keys()
                    for num_frames in bucket_config[res].keys()
                ]
        else:
            mask_gen = MaskGenerator(args.mask_ratios)
            all_buckets = Bucket(bucket_config) if bucket_config is not None else None
            individual_buckets = [all_buckets]

        # output_columns=["video", "caption", "mask", "fps", "num_frames", "frames_mask"],
        output_columns = ["video", "caption", "mask", "frames_mask", "num_frames", "height", "width", "fps", "ar"]
        if args.pre_patchify:
            output_columns.extend(["spatial_pos", "spatial_mask", "temporal_pos", "temporal_mask"])

        datasets = [
            VideoDatasetRefactored(
                csv_path=csv_path,
                video_folder=video_folder,
                text_emb_folder=text_embed_folder,
                vae_latent_folder=vae_latent_folder,
                vae_scale_factor=args.sd_scale_factor,
                sample_n_frames=args.num_frames,
                sample_stride=args.frame_stride,
                frames_mask_generator=mask_gen,
                t_compress_func=(lambda x: vae.get_latent_size((x, None, None))[0]) if vae is not None else None,
                buckets=buckets,
                filter_data=args.filter_data,
                pre_patchify=args.pre_patchify,
                patch_size=latte_model.patch_size,
                embed_dim=latte_model.hidden_size,
                num_heads=latte_model.num_heads,
                max_target_size=args.max_image_size,
                input_sq_size=latte_model.input_sq_size,
                in_channels=latte_model.in_channels,
                apply_train_transforms=True,
                target_size=(img_h, img_w),
                video_backend=args.video_backend,
                output_columns=output_columns,
            )
            for buckets in individual_buckets
        ]

        num_src_samples = sum([len(ds) for ds in datasets])

        dataloaders = [
            create_dataloader(
                dataset,
                batch_size=batch_size if all_buckets is None else 0,  # Turn off batching if using buckets
                shuffle=not validation,
                device_num=device_num,
                rank_id=rank_id,
                num_parallel_workers=args.num_parallel_workers,
                drop_remainder=not validation,
                prefetch_size=args.prefetch_size,
                max_rowsize=args.max_rowsize,
                debug=args.debug,
            )
            for dataset in datasets
        ]
        dataloader = ms.dataset.ConcatDataset(dataloaders) if len(dataloaders) > 1 else dataloaders[0]

        if all_buckets is not None:
            hash_func, bucket_boundaries, bucket_batch_sizes = bucket_split_function(all_buckets)
            dataloader = dataloader.bucket_batch_by_length(
                ["video"],
                bucket_boundaries,
                bucket_batch_sizes,
                element_length_function=hash_func,
                drop_remainder=False,
            )
    return dataloader, num_src_samples


def main(args):
    if args.add_datetime:
        time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        args.output_path = os.path.join(args.output_path, time_str)

    if (args.image_size or (args.resolution and args.aspect_ratio)) and args.bucket_config:
        logger.info("Image size is provided, bucket configuration will be ignored.")
        args.bucket_config = None

    img_h, img_w = None, None
    if args.pre_patchify:
        img_h, img_w = args.max_image_size, args.max_image_size
    elif args.image_size:
        img_h, img_w = args.image_size if isinstance(args.image_size, list) else (args.image_size, args.image_size)
    elif args.bucket_config is None:
        if args.resolution is None or args.aspect_ratio is None:
            raise ValueError(
                "`resolution` and `aspect_ratio` must be provided if `image_size` or `bucket_config` are not provided"
            )
        img_h, img_w = get_image_size(args.resolution, args.aspect_ratio)

    if args.model_version == "v1":
        assert img_h == img_w, "OpenSora v1 support square images only."

    # 1. init
    rank_id, device_num = init_env(
        args.mode,
        seed=args.seed,
        distributed=args.use_parallel,
        device_target=args.device_target,
        max_device_memory=args.max_device_memory,
        parallel_mode=args.parallel_mode,
        jit_level=args.jit_level,
        global_bf16=args.global_bf16,
        dynamic_shape=(args.bucket_config is not None),
        enable_sequence_parallelism=args.enable_sequence_parallelism,
        sequence_parallel_shards=args.sequence_parallel_shards,
        debug=args.debug,
        save_graphs=args.save_graphs,
    )

    set_logger(name="", output_dir=args.output_path, rank=rank_id, log_level=eval(args.log_level))

    # 2. model initiate and weight loading
    dtype_map = {"fp16": ms.float16, "bf16": ms.bfloat16}

    # 2.1 vae
    logger.info("vae init")
    train_with_vae_latent = args.vae_latent_folder is not None and os.path.exists(args.vae_latent_folder)
    if not train_with_vae_latent:
        if args.vae_type in [None, "VideoAutoencoderKL"]:
            vae = VideoAutoencoderKL(
                config=SD_CONFIG, ckpt_path=args.vae_checkpoint, micro_batch_size=args.vae_micro_batch_size
            )
        elif args.vae_type == "OpenSoraVAE_V1_2":
            if args.vae_micro_frame_size != 17:
                logger.warning("vae_micro_frame_size should be 17 to align with the vae pretrain setting.")
            vae = OpenSoraVAE_V1_2(
                micro_batch_size=args.vae_micro_batch_size,
                micro_frame_size=args.vae_micro_frame_size,
                micro_batch_parallel=args.enable_sequence_parallelism,
                micro_frame_parallel=args.enable_sequence_parallelism,
                ckpt_path=args.vae_checkpoint,
                freeze_vae_2d=True,
            )
        else:
            raise ValueError(f"Unknown VAE type: {args.vae_type}")
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

        # infer latent size
        VAE_Z_CH = vae.out_channels
        latent_size = vae.get_latent_size((args.num_frames, img_h, img_w))
    else:
        # vae cache
        vae = None
        assert args.vae_type != "OpenSoraVAE_V1_2", "vae cache is not supported with 3D VAE currently."
        VAE_Z_CH = SD_CONFIG["z_channels"]
        VAE_T_COMPRESS = 1
        VAE_S_COMPRESS = 8
        latent_size = (args.num_frames // VAE_T_COMPRESS, img_h // VAE_S_COMPRESS, img_w // VAE_S_COMPRESS)

    # 2.2 stdit
    if args.model_version == "v1":
        assert img_h == img_w, "OpenSora v1 support square images only."

    patchify_conv3d_replace = "linear" if args.pre_patchify else args.patchify
    model_extra_args = dict(
        input_size=latent_size,
        in_channels=VAE_Z_CH,
        model_max_length=args.model_max_length,
        patchify_conv3d_replace=patchify_conv3d_replace,  # for Ascend
        manual_pad=args.manual_pad,
        enable_flashattn=args.enable_flash_attention,
        enable_sequence_parallelism=args.enable_sequence_parallelism,
        use_recompute=args.use_recompute,
        num_recompute_blocks=args.num_recompute_blocks,
    )

    if args.pre_patchify and args.model_version != "v1.1":
        raise ValueError("`pre_patchify=True` can only be used in model version 1.1.")

    if args.model_version == "v1":
        model_name = "STDiT"
        model_extra_args.update(
            {
                "space_scale": args.space_scale,  # 0.5 for 256x256. 1. for 512
                "time_scale": args.time_scale,
            }
        )
        latte_model = STDiT_XL_2(**model_extra_args)
    elif args.model_version == "v1.1":
        model_name = "STDiT2"
        model_extra_args.update({"input_sq_size": 512, "qk_norm": True})
        latte_model = STDiT2_XL_2(**model_extra_args)
    elif args.model_version == "v1.2":
        model_name = "STDiT3"
        model_extra_args["qk_norm"] = True
        model_extra_args["freeze_y_embedder"] = args.freeze_y_embedder
        if args.dsp:
            latte_model = STDiT3_XL_2_DSP(**model_extra_args)
        else:
            latte_model = STDiT3_XL_2(**model_extra_args)
    else:
        raise ValueError(f"Unknown model version: {args.model_version}")
    logger.info(f"{model_name} input size: {latent_size if args.bucket_config is None else 'Variable'}")

    # mixed precision
    if args.dtype in ["fp16", "bf16"]:
        if not args.global_bf16:
            latte_model = auto_mixed_precision(
                latte_model,
                amp_level=args.amp_level,
                dtype=dtype_map[args.dtype],
                custom_fp32_cells=WHITELIST_OPS,
            )
    # load checkpoint
    if len(args.pretrained_model_path) > 0:
        logger.info(f"Loading ckpt {args.pretrained_model_path}...")
        latte_model.load_from_checkpoint(args.pretrained_model_path)
    else:
        logger.info("Use random initialization for Latte")
    latte_model.set_train(True)

    if (latent_size[1] and latent_size[1] % latte_model.patch_size[1]) or (
        latent_size[2] and latent_size[2] % latte_model.patch_size[2]
    ):
        height_ = latte_model.patch_size[1] * 8  # FIXME
        width_ = latte_model.patch_size[2] * 8  # FIXME
        msg = f"Image height ({img_h}) and width ({img_w}) should be divisible by {height_} and {width_} respectively."
        if patchify_conv3d_replace == "linear":
            raise ValueError(msg)
        else:
            logger.warning(msg)

    # sequence parallel check
    if args.enable_sequence_parallelism:
        if args.num_frames % args.vae_micro_batch_size != 0 or args.num_frames % args.vae_micro_frame_size != 0:
            raise ValueError(
                f"number of frames `{args.num_frames}` must be divisible by "
                f"VAE micro batch size `{args.vae_micro_batch_size}` and VAE micro frame size `{args.vae_micro_frame_size}`."
            )

        if (
            latte_model.num_heads % args.sequence_parallel_shards != 0
            or latte_model.num_heads < args.sequence_parallel_shards
        ):
            raise ValueError(
                f"number of heads `{latte_model.num_heads}` must be divisble and less than the sequence_parallel_shards `{args.sequence_parallel_shards}`."
            )

        if args.num_frames % args.sequence_parallel_shards != 0:
            logger.warning(
                f"To avoid extra computation cost, number of frames `{args.num_frames}` "
                f"should be divisible by the number of SP shards `{args.sequence_parallel_shards}`."
            )

    # 2.3 ldm with loss
    logger.info(f"Train with vae latent cache: {train_with_vae_latent}")
    diffusion = create_diffusion(timestep_respacing="")
    latent_diffusion_eval, metrics = None, {}
    pipeline_kwargs = dict(
        scale_factor=args.sd_scale_factor,
        cond_stage_trainable=False,
        text_emb_cached=True,
        video_emb_cached=train_with_vae_latent,
    )
    if args.noise_scheduler.lower() == "ddpm":
        if args.validate:
            logger.warning(
                "Validation is supported with Rectified Flow noise scheduler only. No validation will be performed."
            )
        if args.pre_patchify:
            additional_pipeline_kwargs = dict(
                patch_size=latte_model.patch_size,
                max_image_size=args.max_image_size,
                vae_downsample_rate=8.0,
                in_channels=latte_model.in_channels,
            )
            pipeline_kwargs.update(additional_pipeline_kwargs)
            pipeline_ = DiffusionWithLossFiTLike
        else:
            pipeline_ = DiffusionWithLoss
    elif args.noise_scheduler.lower() == "rflow":
        if args.validate:
            if args.val_bucket_config is None:
                metrics = {"Validation loss": BucketLoss(str((img_h, img_w)), {(img_h, img_w)}, args.num_frames)}
            else:
                metrics = {
                    f"Validation loss {res}x{frames}": BucketLoss(res, set(ASPECT_RATIOS[res][1].values()), frames)
                    for res, val in args.val_bucket_config.items()
                    for frames in val.keys()
                }
            latent_diffusion_eval = RFlowEvalDiffusionWithLoss(
                latte_model,
                diffusion,
                num_eval_timesteps=args.num_eval_timesteps,
                vae=vae,
                text_encoder=None,
                **pipeline_kwargs,
            )
        pipeline_kwargs.update(
            dict(sample_method=args.sample_method, use_timestep_transform=args.use_timestep_transform)
        )
        pipeline_ = RFlowDiffusionWithLoss
    else:
        raise ValueError(f"Unknown noise scheduler: {args.noise_scheduler}")

    latent_diffusion_with_loss = pipeline_(latte_model, diffusion, vae=vae, text_encoder=None, **pipeline_kwargs)

    # 3. create dataset
    if args.enable_sequence_parallelism:
        data_device_num = device_num // args.sequence_parallel_shards
        data_rank_id = rank_id // args.sequence_parallel_shards
        logger.info(f"Creating dataloader: ID={rank_id}, group={data_rank_id}, num_groups={data_device_num}")
    else:
        data_device_num = device_num
        data_rank_id = rank_id

    dataloader, num_src_samples = initialize_dataset(
        args,
        args.csv_path,
        args.video_folder,
        args.text_embed_folder,
        args.vae_latent_folder,
        train_with_vae_latent,
        args.batch_size,
        img_h,
        img_w,
        latte_model,
        vae,
        bucket_config=args.bucket_config,
        device_num=data_device_num,
        rank_id=data_rank_id,
    )

    # FIXME: get_dataset_size() is extremely slow when used with bucket_batch_by_length
    if args.bucket_config is None:
        dataset_size = dataloader.get_dataset_size()
    else:
        # steps per epoch is not constant in bucket config training
        # FIXME: It is a highly relaxed estimation to ensure enough steps per epoch to sustain training. \
        # A more precise estimation or run-time infer is to be implemented.
        dataset_size = math.ceil(num_src_samples / device_num)
        dataloader.dataset_size = dataset_size
        logger.warning(
            f"Manually set dataset_size to {dataset_size} to skip get_dataset_size() for bucket config training."
        )

    val_dataloader = None
    if args.validate:
        val_dataloader = initialize_dataset(
            args,
            args.val_csv_path,
            args.val_video_folder,
            args.val_text_embed_folder,
            args.val_vae_latent_folder,
            train_with_vae_latent,
            args.val_batch_size,
            img_h,
            img_w,
            latte_model,
            vae,
            bucket_config=args.val_bucket_config,
            validation=True,
            device_num=data_device_num,
            rank_id=data_rank_id,
        )

    # compute total steps and data epochs (in unit of data sink size)
    if args.dataset_sink_mode and args.sink_size != -1:
        # in data sink mode, data sink size determines the number of training steps per epoch.
        steps_per_epoch = args.sink_size
    else:
        # without data sink, number of training steps is determined by number of data batches of the whole training set.
        steps_per_epoch = dataset_size

    if args.train_steps == -1:
        assert args.epochs != -1
        total_train_steps = args.epochs * dataset_size
        sink_epochs = math.ceil(total_train_steps / steps_per_epoch)
    else:
        total_train_steps = args.train_steps
        # asume one step need one whole epoch data to ensure enough batch loading for training
        sink_epochs = total_train_steps

    if args.ckpt_save_steps == -1:
        ckpt_save_interval = args.ckpt_save_interval
        step_mode = False
    else:
        step_mode = not args.dataset_sink_mode
        if not args.dataset_sink_mode:
            ckpt_save_interval = args.ckpt_save_steps
        else:
            # still need to count interval in sink epochs
            ckpt_save_interval = max(1, args.ckpt_save_steps // steps_per_epoch)
            if args.ckpt_save_steps % steps_per_epoch != 0:
                logger.warning(
                    f"`ckpt_save_steps` must be times of sink size or dataset_size under dataset sink mode."
                    f"Checkpoint will be saved every {ckpt_save_interval * steps_per_epoch} steps."
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
        steps_per_epoch=dataset_size,  # not used
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
    cur_iter = 0
    if args.resume:
        resume_ckpt = get_resume_ckpt(args.resume, args.output_path)
        if resume_ckpt is not None:
            start_epoch, cur_iter, loss_scale = get_resume_states(resume_ckpt)
            loss_scaler.loss_scale_value = loss_scale
            logger.info(f"Resumed loss_scaler, prev epoch: {start_epoch}, global step {cur_iter}")

    # trainer (standalone and distributed)
    ema = EMA(latent_diffusion_with_loss.network, ema_decay=args.ema_decay, offloading=True) if args.use_ema else None

    net_with_grads = prepare_train_network(
        latent_diffusion_with_loss,
        optimizer=optimizer,
        scale_sense=loss_scaler,
        drop_overflow_update=args.drop_overflow_update,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        clip_grad=args.clip_grad,
        clip_norm=args.max_grad_norm,
        ema=ema,
        zero_stage=args.zero_stage,
    )

    # resume train net states
    if args.resume and resume_ckpt is not None:
        resume_train_net(net_with_grads, resume_ckpt)

    if (args.mode == 0) and (args.bucket_config is not None):
        _bs = ms.Symbol(unique=True)
        video = ms.Tensor(shape=[_bs, None, 3, None, None], dtype=ms.float32)
        caption = ms.Tensor(shape=[_bs, args.model_max_length, 4096], dtype=ms.float32)
        mask = ms.Tensor(shape=[_bs, args.model_max_length], dtype=ms.uint8)
        frames_mask = ms.Tensor(shape=[_bs, None], dtype=ms.bool_)
        # fmt: off
        num_frames = ms.Tensor(shape=[_bs, ], dtype=ms.float32)
        height = ms.Tensor(shape=[_bs, ], dtype=ms.float32)
        width = ms.Tensor(shape=[_bs, ], dtype=ms.float32)
        fps = ms.Tensor(shape=[_bs, ], dtype=ms.float32)
        ar = ms.Tensor(shape=[_bs, ], dtype=ms.float32)
        # fmt: on
        net_with_grads.set_inputs(video, caption, mask, frames_mask, num_frames, height, width, fps, ar)
        logger.info("Dynamic inputs are initialized for bucket config training in Graph mode!")

    if not args.custom_train:
        if args.global_bf16:
            model = Model(net_with_grads, eval_network=latent_diffusion_eval, metrics=metrics, amp_level="O0")
        else:
            model = Model(net_with_grads, eval_network=latent_diffusion_eval, metrics=metrics)

        # callbacks
        callbacks = [OverflowMonitor(), EMAEvalSwapCallback(ema)]
        if args.bucket_config is None:
            callbacks.append(TimeMonitor(args.log_interval))
        else:
            logger.info(
                "As steps per epoch are inaccurate with bucket config, TimeMonitor is disabled. See result.log for the actual step time"
            )
        if rank_id == 0:
            save_cb = EvalSaveCallback(
                network=latent_diffusion_with_loss.network,
                rank_id=rank_id,
                ckpt_save_dir=ckpt_dir,
                ema=ema,
                save_ema_only=False,
                ckpt_save_policy="latest_k",
                ckpt_max_keep=args.ckpt_max_keep,
                step_mode=step_mode,
                use_step_unit=(args.ckpt_save_steps != -1),
                ckpt_save_interval=ckpt_save_interval,
                log_interval=args.log_interval,
                start_epoch=start_epoch,
                model_name=model_name,
                resume_prefix_blacklist=("vae.", "swap."),
                record_lr=False,
                train_steps=args.train_steps,
            )
            rec_cb = PerfRecorderCallback(
                save_dir=args.output_path,
                file_name="result_val.log",
                metric_names=list(metrics.keys()),
                resume=args.resume,
            )
            callbacks.extend([save_cb, rec_cb])
            if args.profile:
                callbacks.append(ProfilerCallbackEpoch(2, 3, "./profile_data"))

        if args.train_steps > 0:
            callbacks.append(StopAtStepCallback(args.train_steps, global_step=cur_iter))

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
                f"Jit level: {args.jit_level}",
                f"Distributed mode: {args.use_parallel}",
                f"Num params: {num_params:,} (latte: {num_params_latte:,}, vae: {num_params_vae:,})",
                f"Num trainable params: {num_params_trainable:,}",
                f"{model_name} dtype: {args.dtype}",
                f"VAE dtype: {args.vae_dtype}",
                f"Learning rate: {args.start_learning_rate}",
                f"Batch size: {args.batch_size if args.bucket_config is None else 'Variable'}",
                f"Image size: {(img_h, img_w) if args.bucket_config is None else 'Variable'}",
                f"Frames: {args.num_frames if args.bucket_config is None else 'Variable'}",
                f"Latent size: {latent_size if args.bucket_config is None else 'Variable'}",
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
                f"Resume training: {args.resume}",
            ]
        )
        key_info += "\n" + "=" * 50
        logger.info(key_info)

        logger.info("Start training...")

        with open(os.path.join(args.output_path, "args.yaml"), "w") as f:
            yaml.safe_dump(vars(args), stream=f, default_flow_style=False, sort_keys=False)

    # 6. train
    if not args.custom_train:
        model.fit(
            sink_epochs,
            dataloader,
            valid_dataset=val_dataloader,
            valid_frequency=args.val_interval,
            callbacks=callbacks,
            dataset_sink_mode=args.dataset_sink_mode,
            valid_dataset_sink_mode=False,  # TODO: add support?
            sink_size=args.sink_size,
            initial_epoch=start_epoch,
        )

    else:
        assert not args.dataset_sink_mode, "data sink not supported for custom train process currently"

        # re-count training steps and epochs
        if args.train_steps > 0:
            # ensure num_epochs >= train_steps/steps_per_epoch, but steps_per_epoch is uncertain with dynamic BS, the safest bound is to assume it to be 1.
            # Note that it's not the actual data epochs that will be run. Training process will terminate in train_steps
            num_epochs = args.train_steps
        else:
            assert args.epochs > 0, "args.epochs must be given and > 0 if train_steps is not specified"
            # the actual data epochs to be run in this case
            num_epochs = args.epochs
        global_step = cur_iter  # index start from 1 (after first-step network update)

        if args.ckpt_save_steps > 0:
            save_by_step = True
        else:
            save_by_step = False

        if rank_id == 0:
            ckpt_manager = CheckpointManager(ckpt_dir, "latest_k", k=args.ckpt_max_keep)
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            perf_columns = ["step", "loss", "train_time(s)", "shape"]
            output_dir = ckpt_dir.replace("/ckpt", "")
            if start_epoch == 0:
                record = PerfRecorder(output_dir, metric_names=perf_columns)
            else:
                record = PerfRecorder(output_dir, resume=True)

        ds_iter = dataloader.create_tuple_iterator(num_epochs=num_epochs - start_epoch)
        # ds_iter = dataloader.create_tuple_iterator(num_epochs=-1) # infinite
        end_train = False
        for epoch in range(start_epoch + 1, num_epochs + 1):
            if (args.train_steps > 0) and (global_step >= args.train_steps):
                logger.warning("resumed steps >= train_steps, will end training")
                break

            start_time_s = time.time()
            for step, data in enumerate(ds_iter, 1):
                loss, overflow, scaling_sens = net_with_grads(*data)
                global_step += 1
                step_time = time.time() - start_time_s

                # log
                # print(data[0].shape)
                loss_val = float(loss.asnumpy())
                logger.info(
                    f"Epoch {epoch}, Step {step}, loss {loss_val:.5f}, Global step {global_step},"
                    + f" Shape: {tuple(data[0].shape)}, Step time {step_time*1000:.2f}ms"
                )
                if overflow:
                    logger.warning("overflow detected")

                if rank_id == 0:
                    step_pref_value = [global_step, loss_val, step_time, tuple(data[0].shape)]
                    record.add(*step_pref_value)
                # save and eval in step
                if save_by_step and rank_id == 0:
                    if (global_step % args.ckpt_save_steps == 0) or (global_step == args.train_steps):
                        ckpt_name = f"{model_name}-s{global_step}.ckpt"
                        # save model ckpt and ema ckpt
                        save_ema_ckpts(latent_diffusion_with_loss.network, ema, ckpt_manager, ckpt_name)
                        # save train state for resume
                        save_train_net(net_with_grads, ckpt_dir, epoch - 1, global_step)
                if (args.train_steps > 0) and (global_step >= args.train_steps):
                    end_train = True
                    break

                start_time_s = time.time()

            # save and eval in epoch
            if not save_by_step and rank_id == 0:
                if (epoch % args.ckpt_save_interval == 0) or (epoch == num_epochs):
                    ckpt_name = f"{model_name}-e{epoch}.ckpt"
                    # save model ckpt and ema ckpt
                    save_ema_ckpts(latent_diffusion_with_loss.network, ema, ckpt_manager, ckpt_name)
                    # save train state for resume
                    save_train_net(net_with_grads, ckpt_dir, epoch, global_step)

            dataloader.reset()
            flush_from_cache(net_with_grads)

            if end_train:
                break

        logger.info("Finished training. Ending process...")
        reset_op_id()
        time.sleep(60)
        logger.info("End")


if __name__ == "__main__":
    logger.debug("process id:", os.getpid())
    args = parse_args()
    main(args)
