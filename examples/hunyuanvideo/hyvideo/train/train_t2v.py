import logging
import math
import os
import sys

import yaml

import mindspore as ms
from mindspore import Model
from mindspore.communication.management import GlobalComm
from mindspore.train import get_metric_fn
from mindspore.train.callback import TimeMonitor

mindone_lib_path = os.path.abspath("../../")
sys.path.insert(0, mindone_lib_path)
sys.path.append("./")
from hyvideo.constants import PRECISION_TO_TYPE, PRECISIONS, VAE_PATH
from hyvideo.dataset import getdataset
from hyvideo.dataset.loader import create_dataloader
from hyvideo.diffusion.net_with_loss import DiffusionWithLoss, DiffusionWithLossEval
from hyvideo.diffusion.schedulers import FlowMatchDiscreteScheduler
from hyvideo.modules.models import HUNYUAN_VIDEO_CONFIG, HYVideoDiffusionTransformer
from hyvideo.train.commons import create_loss_scaler, parse_args
from hyvideo.utils.callbacks import EMAEvalSwapCallback, PerfRecorderCallback
from hyvideo.utils.dataset_utils import Collate, LengthGroupedSampler
from hyvideo.utils.ema import EMA
from hyvideo.utils.helpers import set_model_param_dtype
from hyvideo.utils.message_utils import print_banner
from hyvideo.utils.ms_utils import init_env
from hyvideo.utils.utils import get_precision
from hyvideo.vae import load_vae
from hyvideo.vae.unet_causal_3d_blocks import GroupNorm, MSInterpolate, MSPad

from examples.hunyuanvideo.hyvideo.utils.parallel_states import get_sequence_parallel_state, hccl_info

# from mindone.diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from mindone.diffusers.schedulers import DDPMScheduler
from mindone.trainers.callback import EvalSaveCallback, OverflowMonitor, ProfilerCallbackEpoch, StopAtStepCallback
from mindone.trainers.checkpoint import resume_train_network
from mindone.trainers.lr_schedule import create_scheduler
from mindone.trainers.optim import create_optimizer
from mindone.trainers.train_step import TrainOneStepWrapper
from mindone.trainers.zero import prepare_train_network
from mindone.transformers import CLIPTextModelWithProjection, MT5EncoderModel, T5EncoderModel
from mindone.utils.amp import auto_mixed_precision
from mindone.utils.config import str2bool
from mindone.utils.logger import set_logger
from mindone.utils.params import count_params

logger = logging.getLogger(__name__)


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


#################################################################################
#                                  Training Loop                                #
#################################################################################


def main(args):
    # 1. init
    if args.num_frames == 1 or args.use_image_num != 0:
        args.sp_size = 1
    save_src_strategy = args.use_parallel and args.parallel_mode != "data"
    rank_id, device_num = init_env(
        args.mode,
        seed=args.seed,
        distributed=args.use_parallel,
        device_target=args.device,
        max_device_memory=args.max_device_memory,
        parallel_mode=args.parallel_mode,
        mempool_block_size=args.mempool_block_size,
        global_bf16=args.global_bf16,
        strategy_ckpt_save_file=os.path.join(args.output_dir, "src_strategy.ckpt") if save_src_strategy else "",
        optimizer_weight_shard_size=args.optimizer_weight_shard_size,
        sp_size=args.sp_size if args.num_frames != 1 and args.use_image_num == 0 else 1,
        jit_level=args.jit_level,
        enable_parallel_fusion=args.enable_parallel_fusion,
        jit_syntax_level=args.jit_syntax_level,
        comm_fusion=args.comm_fusion,
    )
    set_logger(name="", output_dir=args.output_dir, rank=rank_id, log_level=eval(args.log_level))
    if args.profile_memory:
        if args.mode == 1:
            # maybe slow
            ms.context.set_context(pynative_synchronize=True)
        profiler = ms.Profiler(output_path="./mem_info", profile_memory=True)
        # ms.context.set_context(memory_optimize_level="O0")  # enabling it may consume more memory
        logger.info(f"Memory profiling: {profiler}")
    # 2. Init and load models
    # Load VAE
    train_with_vae_latent = args.vae_latent_folder is not None and len(args.vae_latent_folder) > 0
    if train_with_vae_latent:
        assert os.path.exists(
            args.vae_latent_folder
        ), f"The provided vae latent folder {args.vae_latent_folder} is not existent!"
        logger.info("Train with vae latent cache.")
        vae = None
    else:
        print_banner("vae init")
        if args.vae_fp32:
            logger.info("Force VAE running in FP32")
            args.vae_precision = "fp32"
        vae, _, s_ratio, t_ratio = load_vae(
            args.vae,
            logger=logger,
        )
        # vae_kwargs = {"s_ratio": s_ratio, "t_ratio": t_ratio}

        if args.vae_tiling:
            vae.enable_tiling()

        if args.vae_precision in ["fp16", "bf16"]:
            amp_level = "O2"
            vae_dtype = PRECISION_TO_TYPE[args.vae_precision]
            if vae_dtype == ms.float16:
                custom_fp32_cells = [GroupNorm] if args.vae_keep_gn_fp32 else []
            else:
                custom_fp32_cells = [MSPad, MSInterpolate]

            vae = auto_mixed_precision(vae, amp_level, vae_dtype, custom_fp32_cells=custom_fp32_cells)
            logger.info(
                f"Set mixed precision to {amp_level} with dtype={args.vae_precision}, custom fp32_cells {custom_fp32_cells}"
            )
        elif args.vae_precision == "fp32":
            vae_dtype = PRECISION_TO_TYPE[args.vae_precision]
        else:
            raise ValueError(f"Unsupported precision {args.vae_precision}")

    ae_stride_t, ae_stride_h, ae_stride_w = 4, 8, 8

    assert (
        ae_stride_h == ae_stride_w
    ), f"Support only ae_stride_h == ae_stride_w now, but found ae_stride_h ({ae_stride_h}), ae_stride_w ({ae_stride_w})"
    args.ae_stride_t, args.ae_stride_h, args.ae_stride_w = ae_stride_t, ae_stride_h, ae_stride_w
    args.ae_stride = args.ae_stride_h

    patch_size_t, patch_size_h, patch_size_w = 1, 2, 2
    args.patch_size = patch_size_h
    args.patch_size_t, args.patch_size_h, args.patch_size_w = patch_size_t, patch_size_h, patch_size_w
    assert (
        patch_size_h == patch_size_w
    ), f"Support only patch_size_h == patch_size_w now, but found patch_size_h ({patch_size_h}), patch_size_w ({patch_size_w})"
    assert (
        args.max_height % ae_stride_h == 0
    ), f"Height must be divisible by ae_stride_h, but found Height ({args.max_height}), ae_stride_h ({ae_stride_h})."
    assert (
        args.num_frames - 1
    ) % ae_stride_t == 0, f"(Frames - 1) must be divisible by ae_stride_t, but found num_frames ({args.num_frames}), ae_stride_t ({ae_stride_t})."
    assert (
        args.max_width % ae_stride_h == 0
    ), f"Width size must be divisible by ae_stride_h, but found Width ({args.max_width}), ae_stride_h ({ae_stride_h})."

    args.stride_t = ae_stride_t * patch_size_t
    args.stride = ae_stride_h * patch_size_h
    vae.latent_size = (args.max_height // ae_stride_h, args.max_width // ae_stride_w)
    args.latent_size_t = (args.num_frames - 1) // ae_stride_t + 1

    # Load diffusion transformer
    print_banner("Transformer model init")
    FA_dtype = get_precision(args.precision) if get_precision(args.precision) != ms.float32 else ms.bfloat16
    model_dtype = PRECISION_TO_TYPE[args.precision]

    factor_kwargs = {
        "dtype": model_dtype,
        "use_recompute": args.gradient_checkpointing,
        "num_no_recompute": args.num_no_recompute,
    }
    model = HYVideoDiffusionTransformer(
        args,
        in_channels=args.latent_channels,
        use_conv2d_patchify=True,
        attn_mode="vanilla",
        **HUNYUAN_VIDEO_CONFIG[args.model],
        **factor_kwargs,
    )

    # mixed precision
    if model_dtype != ms.float32:
        set_model_param_dtype(model, dtype=model_dtype)

    if model_dtype != ms.float32:
        auto_mixed_precision(model, amp_level="O2", dtype=model_dtype)

    # load checkpoint
    if args.pretrained is not None and len(args.pretrained) > 0:
        assert os.path.exists(args.pretrained), f"Provided checkpoint file {args.pretrained} does not exist!"
        logger.info(f"Loading ckpt {args.pretrained}...")
        model.load_from_checkpoint(args.pretrained)
    else:
        logger.info("Use random initialization for transformer")
    model.set_train(True)

    # Load text encoder
    if not args.text_embed_cache:
        print_banner("text encoder init")
        text_encoder_dtype = get_precision(args.text_encoder_precision)
        if "mt5" in args.text_encoder_name_1:
            text_encoder_1, loading_info = MT5EncoderModel.from_pretrained(
                args.text_encoder_name_1,
                cache_dir=args.cache_dir,
                output_loading_info=True,
                mindspore_dtype=text_encoder_dtype,
                use_safetensors=True,
            )
            loading_info.pop("unexpected_keys")  # decoder weights are ignored
            logger.info(f"Loaded MT5 Encoder: {loading_info}")
            text_encoder_1 = text_encoder_1.set_train(False)
        else:
            text_encoder_1 = T5EncoderModel.from_pretrained(
                args.text_encoder_name_1, cache_dir=args.cache_dir, mindspore_dtype=text_encoder_dtype
            ).set_train(False)
        text_encoder_2 = None
        if args.text_encoder_name_2 is not None:
            text_encoder_2, loading_info = CLIPTextModelWithProjection.from_pretrained(
                args.text_encoder_name_2,
                cache_dir=args.cache_dir,
                mindspore_dtype=text_encoder_dtype,
                output_loading_info=True,
                use_safetensors=True,
            )
            loading_info.pop("unexpected_keys")  # only load text model, ignore vision model
            # loading_info.pop("mising_keys") # Note: missed keys when loading open-clip models
            logger.info(f"Loaded CLIP Encoder: {loading_info}")
            text_encoder_2 = text_encoder_2.set_train(False)
    else:
        text_encoder_1 = None
        text_encoder_2 = None
        text_encoder_dtype = None

    kwargs = dict(prediction_type=args.prediction_type, rescale_betas_zero_snr=args.rescale_betas_zero_snr)
    if args.cogvideox_scheduler:
        from mindone.diffusers import CogVideoXDDIMScheduler

        noise_scheduler = CogVideoXDDIMScheduler(**kwargs)
    elif args.v1_5_scheduler:
        kwargs["beta_start"] = 0.00085
        kwargs["beta_end"] = 0.0120
        kwargs["beta_schedule"] = "scaled_linear"
        noise_scheduler = DDPMScheduler(**kwargs)
    elif args.rf_scheduler:
        noise_scheduler = FlowMatchDiscreteScheduler()
    else:
        noise_scheduler = DDPMScheduler(**kwargs)

    assert args.use_image_num >= 0, f"Expect to have use_image_num>=0, but got {args.use_image_num}"
    if args.use_image_num > 0:
        logger.info("Enable video-image-joint training")
    else:
        if args.num_frames == 1:
            logger.info("Training on image datasets only.")
        else:
            logger.info("Training on video datasets only.")

    latent_diffusion_with_loss = DiffusionWithLoss(
        model,
        noise_scheduler,
        vae=vae,
        text_encoder=text_encoder_1,
        text_emb_cached=args.text_embed_cache,
        video_emb_cached=False,
        use_image_num=args.use_image_num,
        dtype=model_dtype,
        noise_offset=args.noise_offset,
        snr_gamma=args.snr_gamma,
        rf_scheduler=args.rf_scheduler,
        rank_id=rank_id,
        device_num=device_num,
    )
    latent_diffusion_eval, metrics, eval_indexes = None, None, None

    # 3. create dataset
    # TODO: replace it with new dataset
    assert args.dataset == "t2v", "Support t2v dataset only."
    print_banner("Training dataset Loading...")

    # Setup data:
    # TODO: to use in v1.3
    if args.trained_data_global_step is not None:
        initial_global_step_for_sampler = args.trained_data_global_step
    else:
        initial_global_step_for_sampler = 0
    total_batch_size = args.train_batch_size * device_num * args.gradient_accumulation_steps
    total_batch_size = total_batch_size // args.sp_size * args.train_sp_batch_size
    args.total_batch_size = total_batch_size
    if args.max_hxw is not None and args.min_hxw is None:
        args.min_hxw = args.max_hxw // 4

    train_dataset = getdataset(args, dataset_file=args.data)
    sampler = LengthGroupedSampler(
        args.train_batch_size,
        world_size=device_num if not get_sequence_parallel_state() else (device_num // hccl_info.world_size),
        gradient_accumulation_size=args.gradient_accumulation_steps,
        initial_global_step=initial_global_step_for_sampler,
        lengths=train_dataset.lengths,
        group_data=args.group_data,
    )
    collate_fn = Collate(args.train_batch_size, args)
    dataloader = create_dataloader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=sampler is None,
        device_num=device_num if not get_sequence_parallel_state() else (device_num // hccl_info.world_size),
        rank_id=rank_id if not get_sequence_parallel_state() else hccl_info.group_id,
        num_parallel_workers=args.dataloader_num_workers,
        max_rowsize=args.max_rowsize,
        prefetch_size=args.dataloader_prefetch_size,
        collate_fn=collate_fn,
        sampler=sampler,
        column_names=[
            "pixel_values",
            "attention_mask",
            "text_embed",
            "encoder_attention_mask",
            "text_embed_2",
            "encoder_attention_mask_2",
        ],
    )
    dataloader_size = dataloader.get_dataset_size()
    assert (
        dataloader_size > 0
    ), "Incorrect training dataset size. Please check your dataset size and your global batch size"

    val_dataloader = None
    if args.validate:
        assert args.val_data is not None, f"validation dataset must be specified, but got {args.val_data}"
        assert os.path.exists(args.val_data), f"validation dataset file must exist, but got {args.val_data}"
        print_banner("Validation dataset Loading...")
        val_dataset = getdataset(args, dataset_file=args.val_data)
        sampler = LengthGroupedSampler(
            args.val_batch_size,
            world_size=device_num if not get_sequence_parallel_state() else (device_num // hccl_info.world_size),
            lengths=val_dataset.lengths,
            gradient_accumulation_size=args.gradient_accumulation_steps,
            initial_global_step=initial_global_step_for_sampler,
            group_data=args.group_data,
        )

        collate_fn = Collate(args.val_batch_size, args)
        val_dataloader = create_dataloader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=sampler is None,
            device_num=device_num if not get_sequence_parallel_state() else (device_num // hccl_info.world_size),
            rank_id=rank_id if not get_sequence_parallel_state() else hccl_info.group_id,
            num_parallel_workers=args.dataloader_num_workers,
            max_rowsize=args.max_rowsize,
            prefetch_size=args.dataloader_prefetch_size,
            collate_fn=collate_fn,
            sampler=sampler,
            column_names=[
                "pixel_values",
                "attention_mask",
                "text_embed",
                "encoder_attention_mask",
                "text_embed_2",
                "encoder_attention_mask_2",
            ],
        )
        val_dataloader_size = val_dataloader.get_dataset_size()
        assert (
            val_dataloader_size > 0
        ), "Incorrect validation dataset size. Please check your dataset size and your global batch size"

        # create eval network
        latent_diffusion_eval = DiffusionWithLossEval(
            model,
            noise_scheduler,
            vae=vae,
            text_encoder=text_encoder_1,
            text_emb_cached=args.text_embed_cache,
            video_emb_cached=False,
            use_image_num=args.use_image_num,
            dtype=model_dtype,
            noise_offset=args.noise_offset,
            snr_gamma=args.snr_gamma,
        )
        metrics = {"val loss": get_metric_fn("loss")}
        eval_indexes = [0, 1, 2]  # the indexes of the output of eval network: loss. pred and label
    # 4. build training utils: lr, optim, callbacks, trainer
    if args.scale_lr:
        learning_rate = args.start_learning_rate * args.train_batch_size * args.gradient_accumulation_steps * device_num
        end_learning_rate = (
            args.end_learning_rate * args.train_batch_size * args.gradient_accumulation_steps * device_num
        )
    else:
        learning_rate = args.start_learning_rate
        end_learning_rate = args.end_learning_rate

    if args.dataset_sink_mode and args.sink_size != -1:
        assert args.sink_size > 0, f"Expect that sink size is a positive integer, but got {args.sink_size}"
        steps_per_sink = args.sink_size
    else:
        steps_per_sink = dataloader_size

    if args.max_train_steps is not None:
        assert args.max_train_steps > 0, f"max_train_steps should a positive integer, but got {args.max_train_steps}"
        total_train_steps = args.max_train_steps
        args.num_train_epochs = math.ceil(total_train_steps / dataloader_size)
    else:
        # use args.num_train_epochs
        assert (
            args.num_train_epochs is not None and args.num_train_epochs > 0
        ), f"When args.max_train_steps is not provided, args.num_train_epochs must be a positive integer! but got {args.num_train_epochs}"
        total_train_steps = args.num_train_epochs * dataloader_size

    sink_epochs = math.ceil(total_train_steps / steps_per_sink)
    total_train_steps = sink_epochs * steps_per_sink

    if steps_per_sink == dataloader_size:
        logger.info(
            f"Number of training steps: {total_train_steps}, Number of epochs: {args.num_train_epochs}, "
            f"Number of batches in a epoch (dataloader size): {dataloader_size}"
        )
    else:
        logger.info(
            f"Number of training steps: {total_train_steps}, Number of sink epochs: {sink_epochs}, Number of batches in a sink (sink_size): {steps_per_sink}"
        )
    if args.checkpointing_steps is None:
        ckpt_save_interval = args.ckpt_save_interval
        step_mode = False
    else:
        step_mode = not args.dataset_sink_mode
        if not args.dataset_sink_mode:
            ckpt_save_interval = args.checkpointing_steps
        else:
            # still need to count interval in sink epochs
            ckpt_save_interval = max(1, args.checkpointing_steps // steps_per_sink)
            if args.checkpointing_steps % steps_per_sink != 0:
                logger.warning(
                    "`checkpointing_steps` must be times of sink size or dataset_size under dataset sink mode."
                    f"Checkpoint will be saved every {ckpt_save_interval * steps_per_sink} steps."
                )
    if step_mode != args.step_mode:
        logger.info("Using args.checkpointing_steps to determine whether to use step mode to save ckpt.")
        if args.checkpointing_steps is None:
            logger.warning(f"args.checkpointing_steps is not provided. Force step_mode to {step_mode}!")
        else:
            logger.warning(
                f"args.checkpointing_steps is provided. data sink mode is {args.dataset_sink_mode}. Force step mode to {step_mode}!"
            )
    logger.info(
        "ckpt_save_interval: {} {}".format(
            ckpt_save_interval,
            "steps"
            if (not args.dataset_sink_mode and step_mode)
            else ("epochs" if steps_per_sink == dataloader_size else "sink epochs"),
        )
    )
    # build learning rate scheduler
    if not args.lr_decay_steps:
        args.lr_decay_steps = total_train_steps - args.lr_warmup_steps  # fix lr scheduling
        if args.lr_decay_steps <= 0:
            logger.warning(
                f"decay_steps is {args.lr_decay_steps}, please check epochs, dataloader_size and warmup_steps. "
                f"Will force decay_steps to be set to 1."
            )
            args.lr_decay_steps = 1
    assert (
        args.lr_warmup_steps >= 0
    ), f"Expect args.lr_warmup_steps to be no less than zero,  but got {args.lr_warmup_steps}"

    lr = create_scheduler(
        steps_per_epoch=dataloader_size,
        name=args.lr_scheduler,
        lr=learning_rate,
        end_lr=end_learning_rate,
        warmup_steps=args.lr_warmup_steps,
        decay_steps=args.lr_decay_steps,
        total_steps=total_train_steps,
    )
    set_all_reduce_fusion(
        latent_diffusion_with_loss.trainable_params(),
        split_num=7,
        distributed=args.use_parallel,
        parallel_mode=args.parallel_mode,
    )

    # build optimizer
    assert args.optim.lower() == "adamw" or args.optim.lower() == "adamw_re", f"Not support optimizer {args.optim}!"
    optimizer = create_optimizer(
        latent_diffusion_with_loss.trainable_params(),
        name=args.optim,
        betas=args.betas,
        eps=args.optim_eps,
        group_strategy=args.group_strategy,
        weight_decay=args.weight_decay,
        lr=lr,
    )

    loss_scaler = create_loss_scaler(args)
    # resume ckpt
    ckpt_dir = os.path.join(args.output_dir, "ckpt")
    start_epoch = 0
    cur_iter = 0
    if args.resume_from_checkpoint:
        resume_ckpt = (
            os.path.join(ckpt_dir, "train_resume.ckpt")
            if isinstance(args.resume_from_checkpoint, bool)
            else args.resume_from_checkpoint
        )

        start_epoch, loss_scale, cur_iter, last_overflow_iter = resume_train_network(model, optimizer, resume_ckpt)
        loss_scaler.loss_scale_value = loss_scale
        loss_scaler.cur_iter = cur_iter
        loss_scaler.last_overflow_iter = last_overflow_iter
        logger.info(f"Resume training from {resume_ckpt}")

    # trainer (standalone and distributed)
    ema = (
        EMA(
            latent_diffusion_with_loss.network,
            ema_decay=args.ema_decay,
            offloading=args.ema_offload,
            update_after_step=args.ema_start_step,
        )
        if args.use_ema
        else None
    )
    assert (
        args.gradient_accumulation_steps > 0
    ), f"Expect gradient_accumulation_steps is a positive integer, but got {args.gradient_accumulation_steps}"
    if args.parallel_mode == "zero":
        assert args.zero_stage in [0, 1, 2, 3], f"Unsupported zero stage {args.zero_stage}"
        logger.info(f"Training with zero{args.zero_stage} parallelism")
        comm_fusion_dict = None
        if args.comm_fusion:
            comm_fusion_dict = {
                "allreduce": {"openstate": True, "bucket_size": 5e8},
                "reduce_scatter": {"openstate": True, "bucket_size": 5e8},
                "allgather": {"openstate": False, "bucket_size": 5e8},
            }
        net_with_grads = prepare_train_network(
            latent_diffusion_with_loss,
            optimizer,
            zero_stage=args.zero_stage,
            op_group=GlobalComm.WORLD_COMM_GROUP,
            comm_fusion=comm_fusion_dict,
            scale_sense=loss_scaler,
            drop_overflow_update=args.drop_overflow_update,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            clip_grad=args.clip_grad,
            clip_norm=args.max_grad_norm,
            ema=ema,
        )
    else:
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

    # set dynamic inputs
    _bs = ms.Symbol(unique=True)
    video = ms.Tensor(shape=[_bs, 3, None, None, None], dtype=ms.float32)  # (b, c, f, h, w)
    attention_mask = ms.Tensor(shape=[_bs, None, None, None], dtype=ms.float32)  # (b, f, h, w)
    text_tokens = (
        ms.Tensor(shape=[_bs, args.model_max_length_1, None], dtype=ms.float32)
        if args.text_embed_cache
        else ms.Tensor(shape=[_bs, args.model_max_length_1], dtype=ms.float32)
    )
    encoder_attention_mask = ms.Tensor(shape=[_bs, args.model_max_length_1], dtype=ms.uint8)

    text_tokens_2 = (
        ms.Tensor(shape=[_bs, None], dtype=ms.float32)  # pooled hidden states
        if args.text_embed_cache
        else ms.Tensor(shape=[_bs, args.model_max_length_2], dtype=ms.float32)
    )
    encoder_attention_mask_2 = ms.Tensor(shape=[_bs, args.model_max_length_2], dtype=ms.uint8)
    net_with_grads.set_inputs(
        video, attention_mask, text_tokens, encoder_attention_mask, text_tokens_2, encoder_attention_mask_2
    )
    logger.info("Dynamic inputs are initialized for training!")

    if not args.global_bf16:
        model = Model(
            net_with_grads,
            eval_network=latent_diffusion_eval,
            metrics=metrics,
            eval_indexes=eval_indexes,
        )
    else:
        model = Model(
            net_with_grads,
            eval_network=latent_diffusion_eval,
            metrics=metrics,
            eval_indexes=eval_indexes,
            amp_level="O0",
        )
    # callbacks
    callback = [TimeMonitor(args.log_interval), EMAEvalSwapCallback(ema)]
    ofm_cb = OverflowMonitor()
    callback.append(ofm_cb)
    if args.max_train_steps is not None and args.max_train_steps > 0:
        callback.append(StopAtStepCallback(args.max_train_steps, global_step=cur_iter))

    if args.parallel_mode == "optim":
        cb_rank_id = None
        ckpt_save_dir = os.path.join(ckpt_dir, f"rank_{rank_id}")
        output_dir = os.path.join(args.output_dir, "log", f"rank_{rank_id}")
        if args.ckpt_max_keep != 1:
            logger.warning("For semi-auto parallel training, the `ckpt_max_keep` is force to be 1.")
        ckpt_max_keep = 1
        integrated_save = False
        save_training_resume = False  # TODO: support training resume
    else:
        cb_rank_id = rank_id
        ckpt_save_dir = ckpt_dir
        output_dir = None
        ckpt_max_keep = args.ckpt_max_keep
        integrated_save = True
        save_training_resume = True

    if rank_id == 0 or args.parallel_mode == "optim":
        save_cb = EvalSaveCallback(
            network=latent_diffusion_with_loss.network,
            rank_id=cb_rank_id,
            ckpt_save_dir=ckpt_save_dir,
            output_dir=output_dir,
            ema=ema,
            save_ema_only=args.save_ema_only,
            ckpt_save_policy="latest_k",
            ckpt_max_keep=ckpt_max_keep,
            step_mode=step_mode,
            use_step_unit=(args.checkpointing_steps is not None),
            ckpt_save_interval=ckpt_save_interval,
            log_interval=args.log_interval,
            start_epoch=start_epoch,
            model_name=args.model.replace("/", "-"),
            record_lr=False,
            integrated_save=integrated_save,
            save_training_resume=save_training_resume,
        )
        callback.append(save_cb)
        if args.validate:
            assert metrics is not None, "Val during training must set the metric functions"
            rec_cb = PerfRecorderCallback(
                save_dir=args.output_dir,
                file_name="result_val.log",
                resume=args.resume_from_checkpoint,
                metric_names=list(metrics.keys()),
            )
            callback.append(rec_cb)
        if args.profile:
            callback.append(ProfilerCallbackEpoch(2, 2, "./profile_data"))

    # Train!

    # 5. log and save config
    if rank_id == 0:
        if vae is not None:
            num_params_vae, num_params_vae_trainable = count_params(vae)
        else:
            num_params_vae, num_params_vae_trainable = 0, 0
        num_params_transformer, num_params_transformer_trainable = count_params(latent_diffusion_with_loss.network)
        num_params = num_params_vae + num_params_transformer
        num_params_trainable = num_params_vae_trainable + num_params_transformer_trainable
        key_info = "Key Settings:\n" + "=" * 50 + "\n"
        key_info += "\n".join(
            [
                f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.mode}",
                f"Jit level: {args.jit_level}",
                f"Distributed mode: {args.use_parallel}"
                + (
                    f"\nParallel mode: {args.parallel_mode}"
                    + (f"{args.zero_stage}" if args.parallel_mode == "zero" else "")
                    if args.use_parallel
                    else ""
                )
                + (f"\nsp_size: {args.sp_size}" if args.sp_size != 1 else ""),
                f"Num params: {num_params} (transformer: {num_params_transformer}, vae: {num_params_vae})",
                f"Num trainable params: {num_params_trainable}",
                f"Transformer model dtype: {model_dtype}",
                f"Transformer AMP level: {args.amp_level}" if not args.global_bf16 else "Global BF16: True",
                f"VAE dtype: {vae_dtype}"
                + (f"\nText encoder dtype: {text_encoder_dtype}" if text_encoder_dtype is not None else ""),
                f"Learning rate: {learning_rate}",
                f"Instantaneous batch size per device: {args.train_batch_size}",
                f"Total train batch size (w. parallel, distributed & accumulation): {total_batch_size}",
                f"Image height: {args.max_height}",
                f"Image width: {args.max_width}",
                f"Number of frames: {args.num_frames}",
                f"Use image num: {args.use_image_num}",
                f"Optimizer: {args.optim}",
                f"Optimizer epsilon: {args.optim_eps}",
                f"Weight decay: {args.weight_decay}",
                f"Grad accumulation steps: {args.gradient_accumulation_steps}",
                f"Num of training steps: {total_train_steps}",
                f"Loss scaler: {args.loss_scaler_type}",
                f"Init loss scale: {args.init_loss_scale}",
                f"Grad clipping: {args.clip_grad}",
                f"Max grad norm: {args.max_grad_norm}",
                f"EMA: {args.use_ema}",
                f"EMA decay: {args.ema_decay}",
                f"EMA cpu offload: {args.ema_offload}",
                f"FA dtype: {FA_dtype}",
                f"Use recompute(gradient checkpoint): {args.gradient_checkpointing}",
                f"Dataset sink: {args.dataset_sink_mode}",
            ]
        )
        key_info += "\n" + "=" * 50
        logger.info(key_info)

        logger.info("Start training...")

        with open(os.path.join(args.output_dir, "args.yaml"), "w") as f:
            yaml.safe_dump(vars(args), stream=f, default_flow_style=False, sort_keys=False)

    # 6. train
    model.fit(
        sink_epochs,
        dataloader,
        valid_dataset=val_dataloader,
        valid_frequency=args.val_interval,
        callbacks=callback,
        dataset_sink_mode=args.dataset_sink_mode,
        valid_dataset_sink_mode=False,  # TODO: add support?
        sink_size=args.sink_size,
        initial_epoch=start_epoch,
    )


def parse_t2v_train_args(parser):
    # TODO: NEW in v1.3 , but may not use
    # dataset & dataloader
    parser.add_argument("--max_hxw", type=int, default=None)
    parser.add_argument("--min_hxw", type=int, default=None)
    parser.add_argument("--group_data", action="store_true")
    parser.add_argument("--hw_stride", type=int, default=32)
    parser.add_argument("--force_resolution", action="store_true")
    parser.add_argument("--trained_data_global_step", type=int, default=None)
    parser.add_argument(
        "--use_decord",
        type=str2bool,
        default=True,
        help="whether to use decord to load videos. If not, use opencv to load videos.",
    )

    # text encoder & vae & diffusion model
    parser.add_argument(
        "--latent_channels",
        default=16,
        type=int,
    )
    parser.add_argument("--text_states_dim", type=int, default=4096, help="Dimension of text embeddings")

    parser.add_argument("--text_states_dim_2", type=int, default=768, help="Second dimension of text embeddings")
    parser.add_argument("--vae_fp32", action="store_true")

    parser.add_argument("--text_encoder_name_1", type=str, default="DeepFloyd/t5-v1_1-xxl")
    parser.add_argument("--text_encoder_name_2", type=str, default=None)

    parser.add_argument("--cogvideox_scheduler", action="store_true")
    parser.add_argument("--v1_5_scheduler", action="store_true")
    parser.add_argument("--rf_scheduler", action="store_true")
    parser.add_argument(
        "--weighting_scheme", type=str, default="logit_normal", choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"]
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )

    # diffusion setting
    parser.add_argument("--offload_ema", action="store_true", help="Offload EMA model to CPU during training step.")
    parser.add_argument("--foreach_ema", action="store_true", help="Use faster foreach implementation of EMAModel.")
    parser.add_argument("--rescale_betas_zero_snr", action="store_true")

    # validation & logs
    parser.add_argument("--enable_profiling", action="store_true")
    parser.add_argument("--num_sampling_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=4.5)

    parser.add_argument("--output_dir", default="outputs/", help="The directory where training results are saved.")
    parser.add_argument("--dataset", type=str, default="t2v")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="The training dataset text file specifying the path of video folder, text embedding cache folder, and the annotation json file",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default=None,
        help="The validation dataset text file, same format as the training dataset text file.",
    )
    parser.add_argument("--cache_dir", type=str, default="./ckpts")
    parser.add_argument(
        "--filter_nonexistent",
        type=str2bool,
        default=True,
        help="Whether to filter out non-existent samples in image datasets and video datasets." "Defaults to True.",
    )
    parser.add_argument(
        "--text_embed_cache",
        type=str2bool,
        default=True,
        help="Whether to use T5 embedding cache. Must be provided in image/video_data.",
    )
    parser.add_argument("--vae_latent_folder", default=None, type=str, help="root dir for the vae latent data")
    parser.add_argument(
        "--vae",
        type=str,
        default="884-16c-hy",
        choices=list(VAE_PATH),
        help="Name of the VAE model.",
    )
    parser.add_argument("--model", type=str, choices=list(HUNYUAN_VIDEO_CONFIG.keys()), default="HYVideo-T/2")

    parser.add_argument("--downsampler", type=str, default=None)

    parser.add_argument("--sample_rate", type=int, default=1)
    parser.add_argument("--train_fps", type=int, default=24)
    parser.add_argument("--drop_short_ratio", type=float, default=1.0)
    parser.add_argument("--speed_factor", type=float, default=1.0)
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--max_height", type=int, default=320)
    parser.add_argument("--max_width", type=int, default=240)
    parser.add_argument("--group_frame", action="store_true")
    parser.add_argument("--group_resolution", action="store_true")
    parser.add_argument("--pretrained", type=str, default=None)

    parser.add_argument("--vae_tiling", action="store_true")

    # parser.add_argument("--attention_mode", type=str, choices=["xformers", "math", "flash"], default="xformers")

    parser.add_argument("--model_max_length_1", type=int, default=315)  # llava llama text encoder
    parser.add_argument(
        "--model_max_length_2", type=int, default=77
    )  # for text encoder 2 tokenizer, but CLIP text encoder returns pooled hidden states
    parser.add_argument("--multi_scale", action="store_true")

    parser.add_argument("--use_image_num", type=int, default=0)
    parser.add_argument("--use_img_from_vid", action="store_true")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument("--cfg", type=float, default=0.1)
    parser.add_argument(
        "--num_no_recompute",
        type=int,
        default=0,
        nargs="+",
        help="If gradient_checkpointing is True, `num_no_recompute` single_blocks and `num_no_recompute` double_blocks will be removed from recomputation list."
        "if it is a positive integer. If it is a tuple (m, n), the m single_blocks and n double_blocks will be removed from recomputation list.",
    )
    parser.add_argument("--dataloader_prefetch_size", type=int, default=None, help="minddata prefetch size setting")
    parser.add_argument("--sp_size", type=int, default=1, help="For sequence parallel")
    parser.add_argument("--train_sp_batch_size", type=int, default=1, help="Batch size for sequence parallel training")
    parser.add_argument(
        "--vae_keep_gn_fp32",
        default=False,
        type=str2bool,
        help="whether keep GroupNorm in fp32. Defaults to False in inference, better to set to True when training vae",
    )
    parser.add_argument(
        "--vae_precision",
        type=str,
        default="fp16",
        choices=PRECISIONS,
        help="Precision mode for the VAE model.",
    )
    parser.add_argument(
        "--text_encoder_precision",
        default="bf16",
        type=str,
        choices=["bf16", "fp16"],
        help="what data type to use for T5 text encoder. Default is `bf16`, which corresponds to ms.bfloat16",
    )
    parser.add_argument(
        "--enable_parallel_fusion", default=True, type=str2bool, help="Whether to parallel fusion for AdamW"
    )
    parser.add_argument("--jit_level", default="O0", help="Set jit level: # O0: KBK, O1:DVM, O2: GE")
    parser.add_argument("--noise_offset", type=float, default=0.02, help="The scale of noise offset.")
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. More details here: \
            https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. \
            If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument("--ema_start_step", type=int, default=0)
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )

    return parser


if __name__ == "__main__":
    logger.debug("process id:", os.getpid())
    args = parse_args(additional_parse_args=parse_t2v_train_args)
    if args.resume_from_checkpoint == "True":
        args.resume_from_checkpoint = True
    main(args)
