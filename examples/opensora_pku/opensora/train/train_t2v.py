import logging
import os
import sys

import yaml

import mindspore as ms
from mindspore import Model, nn
from mindspore.train.callback import TimeMonitor

mindone_lib_path = os.path.abspath(os.path.abspath("../../"))
sys.path.insert(0, mindone_lib_path)
sys.path.append("./")
from mindcv.optim.adamw import AdamW
from opensora.dataset.t2v_dataset import create_dataloader
from opensora.models.ae import ae_channel_config, ae_stride_config, getae_wrapper
from opensora.models.ae.videobase.modules.updownsample import TrilinearInterpolate
from opensora.models.diffusion.diffusion import create_diffusion_T as create_diffusion
from opensora.models.diffusion.latte.modeling_latte import Latte_models, LayerNorm
from opensora.models.diffusion.latte.modules import Attention
from opensora.models.diffusion.latte.net_with_loss import DiffusionWithLoss
from opensora.models.text_encoder.t5 import T5Embedder
from opensora.train.commons import create_loss_scaler, init_env, parse_args
from opensora.utils.utils import get_precision

from mindone.trainers.callback import EvalSaveCallback, OverflowMonitor, ProfilerCallback
from mindone.trainers.checkpoint import resume_train_network
from mindone.trainers.ema import EMA
from mindone.trainers.lr_schedule import create_scheduler
from mindone.trainers.train_step import TrainOneStepWrapper
from mindone.utils.amp import auto_mixed_precision
from mindone.utils.logger import set_logger
from mindone.utils.params import count_params

os.environ["HCCL_CONNECT_TIMEOUT"] = "6000"
os.environ["MS_ASCEND_CHECK_OVERFLOW_MODE"] = "INFNAN_MODE"

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


def main(args):
    # 1. init
    save_src_strategy = args.use_parallel and args.parallel_mode != "data"
    rank_id, device_num = init_env(
        args.mode,
        seed=args.seed,
        distributed=args.use_parallel,
        device_target=args.device,
        max_device_memory=args.max_device_memory,
        parallel_mode=args.parallel_mode,
        enable_dvm=args.enable_dvm,
        mempool_block_size=args.mempool_block_size,
        global_bf16=args.global_bf16,
        strategy_ckpt_save_file=os.path.join(args.output_dir, "src_strategy.ckpt") if save_src_strategy else "",
        optimizer_weight_shard_size=args.optimizer_weight_shard_size,
    )
    set_logger(output_dir=args.output_dir, rank=rank_id, log_level=eval(args.log_level))
    if args.use_deepspeed:
        raise NotImplementedError

    train_with_vae_latent = args.vae_latent_folder is not None and len(args.vae_latent_folder) > 0
    if train_with_vae_latent:
        assert os.path.exists(
            args.vae_latent_folder
        ), f"The provided vae latent folder {args.vae_latent_folder} is not existent!"
        logger.info("Train with vae latent cache.")
        vae = None
    else:
        logger.info("vae init")
        vae = getae_wrapper(args.ae)(args.ae_path, subfolder="vae")
        vae_dtype = ms.bfloat16
        custom_fp32_cells = [nn.GroupNorm] if vae_dtype == ms.float16 else [nn.AvgPool2d, TrilinearInterpolate]
        vae = auto_mixed_precision(vae, amp_level="O2", dtype=vae_dtype, custom_fp32_cells=custom_fp32_cells)
        logger.info(f"Use amp level O2 for causal 3D VAE. Use dtype {vae_dtype}")

        vae.set_train(False)
        for param in vae.get_parameters():  # freeze vae
            param.requires_grad = False
        if args.enable_tiling:
            vae.vae.enable_tiling()
            vae.vae.tile_overlap_factor = args.tile_overlap_factor

        ae_stride_t, ae_stride_h, ae_stride_w = ae_stride_config[args.ae]
        args.ae_stride_t, args.ae_stride_h, args.ae_stride_w = ae_stride_t, ae_stride_h, ae_stride_w
        args.ae_stride = args.ae_stride_h
        patch_size = args.model[-3:]
        patch_size_t, patch_size_h, patch_size_w = int(patch_size[0]), int(patch_size[1]), int(patch_size[2])
        args.patch_size = patch_size_h
        args.patch_size_t, args.patch_size_h, args.patch_size_w = patch_size_t, patch_size_h, patch_size_w
        assert (
            ae_stride_h == ae_stride_w
        ), f"Support only ae_stride_h == ae_stride_w now, but found ae_stride_h ({ae_stride_h}), ae_stride_w ({ae_stride_w})"
        assert (
            patch_size_h == patch_size_w
        ), f"Support only patch_size_h == patch_size_w now, but found patch_size_h ({patch_size_h}), patch_size_w ({patch_size_w})"
        assert (
            args.max_image_size % ae_stride_h == 0
        ), f"Image size must be divisible by ae_stride_h, but found max_image_size ({args.max_image_size}), "
        " ae_stride_h ({ae_stride_h})."

        latent_size = (args.max_image_size // ae_stride_h, args.max_image_size // ae_stride_w)
        vae.latent_size = latent_size
        args.stride_t = ae_stride_t * patch_size_t
        args.stride = ae_stride_h * patch_size_h

    logger.info(f"Init Latte T2V model: {args.model}")
    ae_time_stride = 4
    video_length = args.num_frames // ae_time_stride + 1
    latte_model = Latte_models[args.model](
        in_channels=ae_channel_config[args.ae],
        out_channels=ae_channel_config[args.ae] * 2,
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
        compress_kv_factor=args.compress_kv_factor,
        use_rope=args.use_rope,
        model_max_length=args.model_max_length,
    )

    # mixed precision
    if args.precision == "fp32":
        model_dtype = get_precision(args.precision)
    else:
        model_dtype = get_precision(args.precision)
        if not args.global_bf16:
            latte_model = auto_mixed_precision(
                latte_model,
                amp_level=args.amp_level,
                dtype=model_dtype,
                custom_fp32_cells=[LayerNorm, Attention, nn.SiLU, nn.GELU]
                if model_dtype == ms.float16
                else [nn.MaxPool2d],
            )
            logger.info(f"Set mixed precision to {args.amp_level} with dtype={args.precision}")
        else:
            logger.info(f"Using global bf16 for latte t2v model. Force model dtype from {model_dtype} to ms.bfloat16")
            model_dtype = ms.bfloat16
    # load checkpoint
    if len(args.pretrained) > 0:
        logger.info(f"Loading ckpt {args.pretrained}...")
        latte_model.load_from_checkpoint(args.pretrained)
    else:
        logger.info("Use random initialization for Latte")
    latte_model.set_train(True)

    use_text_embed = args.text_embed_folder is not None and len(args.text_embed_folder) > 0
    if not use_text_embed:
        logger.info("T5 init")
        text_encoder = T5Embedder(
            dir_or_name=args.text_encoder_name,
            cache_dir="./",
            model_max_length=args.model_max_length,
        )
        # mixed precision
        text_encoder_dtype = ms.bfloat16  # using bf16 for text encoder and vae
        text_encoder = auto_mixed_precision(text_encoder, amp_level="O2", dtype=text_encoder_dtype)
        text_encoder.dtype = text_encoder_dtype
        logger.info(f"Use amp level O2 for text encoder T5 with dtype={text_encoder_dtype}")

        tokenizer = text_encoder.tokenizer
    else:
        assert os.path.exists(
            args.text_embed_folder
        ), f"The provided text_embed_folder {args.text_embed_folder} is not existent!"
        text_encoder = None
        tokenizer = None

    # 2.3 ldm with loss
    diffusion = create_diffusion(timestep_respacing="")
    latent_diffusion_with_loss = DiffusionWithLoss(
        latte_model,
        diffusion,
        vae=vae,
        condition="text",
        text_encoder=text_encoder,
        cond_stage_trainable=False,
        text_emb_cached=use_text_embed,
        video_emb_cached=False,
        use_image_num=args.use_image_num,
        dtype=model_dtype,
    )

    # 3. create dataset
    assert args.dataset == "t2v", "Support t2v dataset only."
    ds_config = dict(
        data_file_path=args.data_path,
        video_folder=args.video_folder,
        text_emb_folder=args.text_embed_folder,
        return_text_emb=use_text_embed,
        vae_latent_folder=args.vae_latent_folder,
        return_vae_latent=train_with_vae_latent,
        vae_scale_factor=args.sd_scale_factor,
        sample_size=args.max_image_size,
        sample_stride=args.sample_rate,
        sample_n_frames=args.num_frames,
        tokenizer=tokenizer,
        video_column=args.video_column,
        caption_column=args.caption_column,
        disable_flip=not args.enable_flip,
        use_image_num=args.use_image_num,
        token_max_length=args.model_max_length,
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
    assert dataset_size > 0, "Incorrect dataset size. Please check your dataset size and your global batch size"

    # 4. build training utils: lr, optim, callbacks, trainer
    if args.max_train_steps is not None and args.max_train_steps > 0:
        args.epochs = args.max_train_steps // dataset_size
        logger.info(f"Forcing training epochs to {args.epochs} when using max_train_steps {args.max_train_steps}")
    if args.checkpointing_steps is not None and args.checkpointing_steps > 0:
        logger.info(f"Saving checkpoints every {args.checkpointing_steps} steps")
        args.step_mode = True
        args.ckpt_save_interval = args.checkpointing_steps

    # build learning rate scheduler
    if not args.lr_decay_steps:
        args.lr_decay_steps = args.epochs * dataset_size - args.lr_warmup_steps  # fix lr scheduling
        if args.lr_decay_steps <= 0:
            logger.warning(
                f"decay_steps is {args.lr_decay_steps}, please check epochs, dataset_size and warmup_steps. "
                f"Will force decay_steps to be set to 1."
            )
            args.lr_decay_steps = 1

    lr = create_scheduler(
        steps_per_epoch=dataset_size,
        name=args.lr_scheduler,
        lr=args.start_learning_rate,
        end_lr=args.end_learning_rate,
        warmup_steps=args.lr_warmup_steps,
        decay_steps=args.lr_decay_steps,
        num_epochs=args.epochs,
    )
    set_all_reduce_fusion(
        latent_diffusion_with_loss.trainable_params(),
        split_num=7,
        distributed=args.use_parallel,
        parallel_mode=args.parallel_mode,
    )

    # build optimizer
    assert args.optim.lower() == "adamw", f"Not support optimizer {args.optim}!"
    optimizer = AdamW(
        latent_diffusion_with_loss.trainable_params(),
        learning_rate=lr,
        beta1=args.betas[0],
        beta2=args.betas[1],
        eps=args.optim_eps,
        weight_decay=args.weight_decay,
    )

    loss_scaler = create_loss_scaler(args)
    # resume ckpt
    ckpt_dir = os.path.join(args.output_dir, "ckpt")
    start_epoch = 0
    if args.resume_from_checkpoint:
        resume_ckpt = (
            os.path.join(ckpt_dir, "train_resume.ckpt")
            if isinstance(args.resume_from_checkpoint, bool)
            else args.resume_from_checkpoint
        )

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

    if not args.global_bf16:
        model = Model(net_with_grads)
    else:
        model = Model(net_with_grads, amp_level="O0")
    # callbacks
    callback = [TimeMonitor(args.log_interval)]
    ofm_cb = OverflowMonitor()
    callback.append(ofm_cb)

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
            ckpt_save_policy="latest_k",
            ckpt_max_keep=ckpt_max_keep,
            step_mode=args.step_mode,
            ckpt_save_interval=args.ckpt_save_interval,
            log_interval=args.log_interval,
            start_epoch=start_epoch,
            model_name=args.model.replace("/", "-"),
            record_lr=False,
            integrated_save=integrated_save,
            save_training_resume=save_training_resume,
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
                f"Distributed mode: {args.use_parallel}" + f"\nParallel mode: {args.parallel_mode}"
                if args.use_parallel
                else "",
                f"Num params: {num_params:,} (latte: {num_params_latte:,}, vae: {num_params_vae:,})",
                f"Num trainable params: {num_params_trainable:,}",
                f"Use model dtype: {model_dtype}",
                f"AMP level: {args.amp_level}" if not args.global_bf16 else "Global BF16: True",
                f"Learning rate: {args.start_learning_rate}",
                f"Batch size: {args.batch_size}",
                f"Image size: {args.max_image_size}",
                f"Number of frames: {args.num_frames}",
                f"Use image num: {args.use_image_num}",
                f"Optimizer: {args.optim}",
                f"Optimizer epsilon: {args.optim_eps}",
                f"Weight decay: {args.weight_decay}",
                f"Grad accumulation steps: {args.gradient_accumulation_steps}",
                f"Num epochs: {args.epochs}",
                f"Loss scaler: {args.loss_scaler_type}",
                f"Init loss scale: {args.init_loss_scale}",
                f"Grad clipping: {args.clip_grad}",
                f"Max grad norm: {args.max_grad_norm}",
                f"EMA: {args.use_ema}",
                f"EMA decay: {args.ema_decay}",
                f"Enable flash attention: {args.enable_flash_attention}",
                f"Use recompute: {args.use_recompute}",
                f"Dataset sink: {args.dataset_sink_mode}",
            ]
        )
        key_info += "\n" + "=" * 50
        logger.info(key_info)

        logger.info("Start training...")

        with open(os.path.join(args.output_dir, "args.yaml"), "w") as f:
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


def parse_t2v_train_args(parser):
    parser.add_argument("--output_dir", default="outputs/", help="The directory where training results are saved.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument(
        "--text_embed_folder", type=str, default=None, help="the folder path to the t5 text embeddings and masks"
    )
    parser.add_argument("--vae_latent_folder", default=None, type=str, help="root dir for the vae latent data")
    parser.add_argument("--model", type=str, default="DiT-XL/122")
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--ae", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--ae_path", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--sample_rate", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--max_image_size", type=int, default=512)
    parser.add_argument("--compress_kv", action="store_true")
    parser.add_argument("--compress_kv_factor", type=int, default=1)
    parser.add_argument("--use_rope", action="store_true")
    parser.add_argument("--attention_mode", type=str, choices=["xformers", "math", "flash"], default="math")
    parser.add_argument("--pretrained", type=str, default=None)

    parser.add_argument("--tile_overlap_factor", type=float, default=0.25)
    parser.add_argument("--enable_tiling", action="store_true")

    parser.add_argument("--video_folder", type=str, default="")
    parser.add_argument("--text_encoder_name", type=str, default="DeepFloyd/t5-v1_1-xxl")
    parser.add_argument("--model_max_length", type=int, default=300)
    parser.add_argument("--multi_scale", action="store_true")

    # parser.add_argument("--enable_tracker", action="store_true")
    parser.add_argument("--use_image_num", type=int, default=0)
    parser.add_argument("--use_img_from_vid", action="store_true")
    parser.add_argument("--use_deepspeed", action="store_true")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=None,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--sd_scale_factor", type=float, default=0.18215, help="VAE scale factor of Stable Diffusion model."
    )
    parser.add_argument("--video_column", default="path", type=str, help="name of column for videos saved in csv file")
    parser.add_argument(
        "--caption_column", default="cap", type=str, help="name of column for captions saved in csv file"
    )
    parser.add_argument(
        "--enable_flip",
        action="store_true",
        help="enable random flip video (disable it to avoid motion direction and text mismatch)",
    )
    return parser


if __name__ == "__main__":
    logger.debug("process id:", os.getpid())
    args = parse_args(additional_parse_args=parse_t2v_train_args)
    main(args)
