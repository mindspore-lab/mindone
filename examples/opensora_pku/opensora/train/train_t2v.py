"""
STDiT training script
"""
import datetime
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
from opensora.dataset.t2v_dataset import create_dataloader
from opensora.models.ae import ae_channel_config, ae_stride_config, getae_model_config, getae_wrapper
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
from mindone.trainers.optim import create_optimizer
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
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    args.output_path = os.path.join(args.output_dir, time_str)

    # 1. init
    rank_id, device_num = init_env(
        args.mode,
        seed=args.seed,
        distributed=args.use_parallel,
        device_target=args.device,
        max_device_memory=args.max_device_memory,
        parallel_mode=args.parallel_mode,
        enable_dvm=args.enable_dvm,
    )
    set_logger(output_dir=args.output_path, rank=rank_id, log_level=eval(args.log_level))
    if args.use_deepspeed:
        raise NotImplementedError

    logger.info("vae init")
    vae = getae_wrapper(args.ae)(getae_model_config(args.ae), args.ae_path, subfolder="vae")
    if args.enable_tiling:
        raise NotImplementedError
        # vae.vae.enable_tiling()
        # vae.vae.tile_overlap_factor = args.tile_overlap_factor
    vae.set_train(False)

    vae = auto_mixed_precision(vae, amp_level="O2", dtype=ms.float16)
    logger.info("Use amp level O2 for causal 3D VAE.")
    for param in vae.get_parameters():  # freeze vae
        param.requires_grad = False

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
    )

    # mixed precision
    if args.precision == "fp32":
        model_dtype = get_precision(args.precision)
    else:
        model_dtype = get_precision(args.precision)
        latte_model = auto_mixed_precision(
            latte_model,
            amp_level=args.amp_level,
            dtype=model_dtype,
            custom_fp32_cells=[LayerNorm, Attention, nn.SiLU, nn.GELU],
        )
    # load checkpoint
    if len(args.pretrained) > 0:
        logger.info(f"Loading ckpt {args.pretrained}...")
        latte_model.load_from_checkpoint(args.pretrained)
    else:
        logger.info("Use random initialization for Latte")
    latte_model.set_train(True)

    logger.info("T5 init")
    text_encoder = T5Embedder(
        dir_or_name=args.text_encoder_name,
        cache_dir="./",
        model_max_length=args.model_max_length,
    )
    tokenizer = text_encoder.tokenizer

    # 2.3 ldm with loss
    diffusion = create_diffusion(timestep_respacing="")
    latent_diffusion_with_loss = DiffusionWithLoss(
        latte_model,
        diffusion,
        vae=vae,
        scale_factor=args.sd_scale_factor,
        condition="text",
        text_encoder=text_encoder,
        cond_stage_trainable=False,
        text_emb_cached=False,
        video_emb_cached=False,
        use_image_num=args.use_image_num,
        dtype=model_dtype,
    )

    # 3. create dataset
    ds_config = dict(
        data_file_path=args.data_path,
        video_folder=args.video_folder,
        text_emb_folder=None,
        return_text_emb=False,
        vae_latent_folder=None,
        return_vae_latent=False,
        vae_scale_factor=args.sd_scale_factor,
        sample_size=args.max_image_size,
        sample_stride=args.sample_rate,
        sample_n_frames=args.num_frames,
        tokenizer=tokenizer,
        video_column=args.video_column,
        caption_column=args.caption_column,
        disable_flip=False,  # use random flip
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
        warmup_steps=args.warmup_steps,
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
            model_name=args.model.replace("/", "-"),
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
                f"Distributed mode: {args.use_parallel}" + f"\nParallel mode: {args.parallel_mode}"
                if args.use_parallel
                else "",
                f"Num params: {num_params:,} (latte: {num_params_latte:,}, vae: {num_params_vae:,})",
                f"Num trainable params: {num_params_trainable:,}",
                f"Use model dtype: {model_dtype}",
                f"AMP level: {args.amp_level}",
                f"Learning rate: {args.start_learning_rate}",
                f"Batch size: {args.batch_size}",
                f"Image size: {args.image_size}",
                f"Number of frames: {args.num_frames}",
                f"Use image num: {args.use_image_num}",
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


def parse_t2v_train_args(parser):
    parser.add_argument("--output_dir", default="outputs/", help="The directory where training results are saved.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model", type=str, default="DiT-XL/122")
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--ae", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--ae_path", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--sample_rate", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--max_image_size", type=int, default=128)
    parser.add_argument("--dynamic_frames", action="store_true")
    parser.add_argument("--compress_kv", action="store_true")
    parser.add_argument("--attention_mode", type=str, choices=["xformers", "math", "flash"], default="math")
    parser.add_argument("--pretrained", type=str, default=None)

    parser.add_argument("--tile_overlap_factor", type=float, default=0.25)
    parser.add_argument("--enable_tiling", action="store_true")

    parser.add_argument("--video_folder", type=str, default="")
    parser.add_argument("--text_encoder_name", type=str, default="DeepFloyd/t5-v1_1-xxl")
    parser.add_argument("--model_max_length", type=int, default=120)

    parser.add_argument("--enable_tracker", action="store_true")
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
    return parser


if __name__ == "__main__":
    logger.debug("process id:", os.getpid())
    args = parse_args(additional_parse_args=parse_t2v_train_args)
    main(args)
