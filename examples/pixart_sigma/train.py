#!/usr/bin/env python
import argparse
import logging
import os
import sys

import yaml

import mindspore as ms
import mindspore.nn as nn
from mindspore import Model, Tensor
from mindspore.dataset import GeneratorDataset

# TODO: remove in future when mindone is ready for install
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)

from pixart.dataset import ImageDataset, bucket_split_function
from pixart.diffusers import AutoencoderKL
from pixart.diffusion.iddpm import create_diffusion
from pixart.modules.pixart import PixArt_XL_2, PixArtMS_XL_2
from pixart.pipelines import NetworkWithLoss, PixArtInferPipeline
from pixart.utils import (
    LossMonitor,
    SaveCkptCallback,
    TimeMonitor,
    TurnOffVAET5Train,
    Visualizer,
    auto_scale_lr,
    check_cfgs_in_parser,
    count_params,
    init_env,
    load_ckpt_params,
    str2bool,
)
from transformers import AutoTokenizer

from mindone.trainers.optim import create_optimizer
from mindone.trainers.train_step import TrainOneStepWrapper
from mindone.transformers import T5EncoderModel
from mindone.utils.amp import auto_mixed_precision
from mindone.utils.logger import set_logger

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="PixArt-Sigma Training script", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-c",
        "--config",
        help="Path to load a config yaml file that describes the setting which will override the default arguments.",
    )
    parser.add_argument("--json_path", required=True, help="path to json annotation file.")
    parser.add_argument("--image_dir", required=True, help="Directory storing the image directory.")
    parser.add_argument("--output_path", default="./output", help="Output directory to save the training result.")

    parser.add_argument("--sample_size", default=64, type=int, choices=[256, 128, 64, 32], help="Network sample size,")
    parser.add_argument("--batch_size", default=64, type=int, help="Training batch size.")
    parser.add_argument("--num_parallel_workers", default=4, type=int, help="Number of workers for data loading.")
    parser.add_argument(
        "--checkpoint", default="models/PixArt-Sigma-XL-2-512-MS.ckpt", help="The path to the PixArt checkpoint."
    )
    parser.add_argument("--vae_root", default="models/vae", help="Path storing the VAE checkpoint and configure file.")
    parser.add_argument(
        "--tokenizer_root", default="models/tokenizer", help="Path storing the T5 checkpoint and configure file."
    )
    parser.add_argument(
        "--text_encoder_root", default="models/text_encoder", help="Path storing the T5 tokenizer and configure file."
    )
    parser.add_argument("--t5_max_length", default=300, type=int, help="T5's embedded sequence length.")
    parser.add_argument(
        "--sd_scale_factor", default=0.13025, type=float, help="VAE scale factor of Stable Diffusion network."
    )
    parser.add_argument(
        "--real_prompt_ratio",
        default=0.5,
        type=float,
        help="The probability of using real text label instead of sharegpt4v.",
    )
    parser.add_argument("--class_dropout_prob", default=0.1, type=float, help="The probility of drop the text label.")
    parser.add_argument("--multi_scale", default=False, type=str2bool, help="Perform multi-scale training.")
    parser.add_argument("--visualize", default=False, type=str2bool, help="Perform visualization during training.")
    parser.add_argument("--visualize_interval", default=1, type=int, help="Visualization interval.")
    parser.add_argument("--validation_prompts", nargs="*", help="Validation prompts for visualziation.")
    parser.add_argument(
        "--validation_negative_prompts", nargs="*", help="Validation negative prompts for visualization."
    )

    parser.add_argument("--device_target", default="Ascend", choices=["CPU", "GPU", "Ascend"], help="Device target.")
    parser.add_argument("--mode", default=0, type=int, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1).")
    parser.add_argument("--jit_level", default="O1", choices=["O0", "O1"], help="Jit Level")
    parser.add_argument("--seed", default=42, type=int, help="Training seed.")

    parser.add_argument(
        "--enable_flash_attention", default=True, type=str2bool, help="whether to enable flash attention."
    )
    parser.add_argument("--kv_compress", default=False, type=str2bool, help="Do KV Compression if it is True.")
    parser.add_argument(
        "--kv_compress_sampling",
        default="conv",
        choices=["conv", "ave", "uniform"],
        help="Sampling method in KV compression.",
    )
    parser.add_argument("--kv_compress_scale_factor", default=1, type=int, help="Scaling value in KV compression.")
    parser.add_argument("--kv_compress_layer", nargs="*", type=int, help="Network layers performing KV compression.")
    parser.add_argument(
        "--dtype", default="fp16", choices=["bf16", "fp16", "fp32"], help="what data type to use for PixArt."
    )
    parser.add_argument("--scheduler", default="constant", choices=["constant"], help="LR scheduler.")
    parser.add_argument("--start_learning_rate", default=2e-5, type=float, help="The learning rate.")
    parser.add_argument("--warmup_steps", default=1000, type=int, help="Warmup steps.")
    parser.add_argument("--epochs", default=200, type=int, help="Number of total training epochs.")
    parser.add_argument("--optim", default="adamw", type=str, choices=["adamw", "came"], help="Optimizer name.")
    parser.add_argument(
        "--adamw_eps", default=1.0e-10, type=float, help="Specify the eps parameter for the AdamW optimizer."
    )
    parser.add_argument(
        "--came_betas",
        default=[0.9, 0.999, 0.9999],
        type=float,
        nargs="*",
        help="Specify the beta parameter for the CAME optimizer.",
    )
    parser.add_argument(
        "--came_eps",
        default=[1e-30, 1e-16],
        type=float,
        nargs="*",
        help="Specify the eps parameter for the CAME optimizer.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay.")
    parser.add_argument("--auto_lr", default="", choices=["", "sqrt", "linear"], help="Use auto-scaled learning rate")
    parser.add_argument(
        "--loss_scaler_type",
        default="dynamic",
        choices=["static", "dynamic"],
        help="Use dynamic or static loss scaler.",
    )
    parser.add_argument("--init_loss_scale", default=65536.0, type=float, help="Loss scale.")
    parser.add_argument("--scale_window", default=1000, type=int, help="Loss scale window.")
    parser.add_argument("--loss_scale_factor", default=2.0, type=float, help="Loss scale factor.")
    parser.add_argument("--use_ema", default=False, type=str2bool, help="Whether to use EMA")
    parser.add_argument("--ema_rate", default=0.9999, type=float, help="EMA Rate.")
    parser.add_argument("--drop_overflow_update", default=True, type=str2bool, help="Drop overflow update.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="Gradient accumulation steps.")
    parser.add_argument("--clip_grad", default=False, type=str2bool, help="Whether apply gradient clipping.")
    parser.add_argument(
        "--max_grad_norm",
        default=0.01,
        type=float,
        help="Max gradient norm for clipping, effective when `clip_grad` enabled.",
    )
    parser.add_argument("--ckpt_max_keep", default=3, type=int, help="Maximum number of checkpoints to keep")
    parser.add_argument("--ckpt_save_interval", default=1, type=int, help="Save checkpoint every this epochs or steps.")
    parser.add_argument("--log_loss_interval", default=1, type=int, help="Log interval of loss value.")
    parser.add_argument("--recompute", default=False, type=str2bool, help="Use recompute during training.")
    parser.add_argument("--use_parallel", default=False, type=str2bool, help="use parallel training.")
    parser.add_argument("--sampling_method", default="iddpm", choices=["iddpm", "ddim", "dpm"], help="Sampling method.")
    parser.add_argument("--sampling_steps", default=100, type=int, help="Diffusion Sampling Steps")
    parser.add_argument("--guidance_scale", default=4.5, type=float, help="Scale value for classifier-free guidance")
    default_args = parser.parse_args()
    abs_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ""))
    if default_args.config:
        logger.info(f"Overwrite default arguments with configuration file {default_args.config}")
        default_args.config = os.path.join(abs_path, default_args.config)
        with open(default_args.config, "r") as f:
            cfg = yaml.safe_load(f)
            check_cfgs_in_parser(cfg, parser)
            parser.set_defaults(**cfg)
    args = parser.parse_args()
    return args


def main(args):
    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)

    # 1. init env
    device_num, rank_id = init_env(args)
    set_logger(output_dir=os.path.join(args.output_path, "logs"), rank=rank_id)

    # 2. model initialize and weight loading
    # 2.1 PixArt
    image_size = args.sample_size * 8
    logger.info(f"{image_size}x{image_size} init")

    network_fn = PixArt_XL_2 if args.sample_size == 32 else PixArtMS_XL_2
    pe_interpolation = args.sample_size / 64
    sampling = args.kv_compress_sampling if args.kv_compress else None
    network = network_fn(
        input_size=args.sample_size,
        pe_interpolation=pe_interpolation,
        model_max_length=args.t5_max_length,
        sampling=sampling,
        scale_factor=args.kv_compress_scale_factor,
        kv_compress_layer=args.kv_compress_layer,
        class_dropout_prob=args.class_dropout_prob,
        recompute=args.recompute,
        block_kwargs={"enable_flash_attention": args.enable_flash_attention},
    )

    if args.dtype == "fp16":
        model_dtype = ms.float16
        network = auto_mixed_precision(network, amp_level="O2", dtype=model_dtype)
    elif args.dtype == "bf16":
        model_dtype = ms.bfloat16
        network = auto_mixed_precision(network, amp_level="O2", dtype=model_dtype)
    else:
        model_dtype = ms.float32

    if args.checkpoint:
        network = load_ckpt_params(network, args.checkpoint)
    else:
        logger.info("Initialize network randomly.")

    # 2.2 VAE
    logger.info("vae init")
    vae = AutoencoderKL.from_pretrained(args.vae_root, mindspore_dtype=model_dtype)

    # 2.3 T5
    logger.info("text encoder init")
    text_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_root, model_max_length=args.t5_max_length)
    text_encoder = T5EncoderModel.from_pretrained(args.text_encoder_root, mindspore_dtype=model_dtype)

    # 2.4 Sampling
    diffusion = create_diffusion(timestep_respacing="")

    # 3. build training network
    latent_diffusion_with_loss = NetworkWithLoss(
        network, diffusion, vae, text_encoder, scale_factor=args.sd_scale_factor
    )

    # 4. build dataset
    dataset = ImageDataset(
        args.json_path,
        args.image_dir,
        image_size,
        text_tokenizer,
        real_prompt_ratio=args.real_prompt_ratio,
        multi_scale=args.multi_scale,
    )
    data_generator = GeneratorDataset(
        dataset,
        column_names=["image", "text_emb", "text_mask"],
        column_types=[ms.float32, ms.int64, ms.bool_],
        shuffle=True,
        num_parallel_workers=args.num_parallel_workers,
        num_shards=device_num,
        shard_id=rank_id,
        max_rowsize=-1,
    )
    if args.multi_scale:
        element_length_func, bucket_boundaries, bucket_batch_size = bucket_split_function(
            dataset.ratio, args.batch_size
        )
        data_generator = data_generator.bucket_batch_by_length(
            ["image"], bucket_boundaries, bucket_batch_size, element_length_func, drop_remainder=True
        )
    else:
        data_generator = data_generator.batch(args.batch_size, drop_remainder=True)

    # 5. build training utils: lr, optim, callbacks, trainer
    # 5.1 learning rate
    if args.auto_lr:
        start_learning_rate = auto_scale_lr(
            args.batch_size * args.gradient_accumulation_steps * device_num, args.start_learning_rate, rule=args.auto_lr
        )
    else:
        start_learning_rate = args.start_learning_rate

    assert args.scheduler == "constant"
    # for bucket training, the total steps is unknown until it is real finished.
    lr = nn.WarmUpLR(learning_rate=start_learning_rate, warmup_steps=args.warmup_steps)

    # 5.2 optimizer
    optim = "adamw_re" if args.optim == "adamw" else args.optim
    eps = args.adamw_eps if args.optim == "adamw" else args.came_eps
    betas = None if args.optim == "adamw" else args.came_betas
    optimizer = create_optimizer(
        latent_diffusion_with_loss.trainable_params(),
        name=optim,
        lr=lr,
        weight_decay=args.weight_decay,
        betas=betas,
        eps=eps,
    )

    if args.loss_scaler_type == "dynamic":
        loss_scaler = nn.DynamicLossScaleUpdateCell(
            loss_scale_value=args.init_loss_scale, scale_factor=args.loss_scale_factor, scale_window=args.scale_window
        )
    else:
        loss_scaler = nn.FixedLossScaleUpdateCell(args.init_loss_scale)

    # 5.3 trainer (standalone and distributed)
    if args.use_ema:
        raise NotImplementedError("`EMA` does not support yet.")
        # ema = EMA(latent_diffusion_with_loss.network, ema_decay=args.ema_rate)
    else:
        ema = None

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

    if args.multi_scale:
        net_with_grads.set_inputs(
            Tensor(dtype=ms.float32, shape=(None, 3, None, None)),
            Tensor(dtype=ms.int64, shape=(None, args.t5_max_length)),
            Tensor(dtype=ms.bool_, shape=(None, args.t5_max_length)),
        )

    model = Model(net_with_grads)

    # 5.4 callbacks
    callbacks = [
        TurnOffVAET5Train(),
        TimeMonitor(),
        LossMonitor(log_interval=args.log_loss_interval),
        SaveCkptCallback(
            output_dir=os.path.join(args.output_path, "ckpt"),
            ckpt_max_keep=args.ckpt_max_keep,
            ckpt_save_interval=args.ckpt_save_interval,
            save_ema=args.use_ema,
            rank_id=rank_id,
        ),
    ]

    if args.visualize:
        # prepare validation
        infer_pipeline = PixArtInferPipeline(
            network,
            vae,
            text_encoder,
            text_tokenizer,
            scale_factor=args.sd_scale_factor,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.sampling_steps,
            sampling_method=args.sampling_method,
        )
        visualizer = Visualizer(
            infer_pipeline,
            args.sample_size,
            validation_prompts=args.validation_prompts,
            validation_negative_prompts=args.validation_negative_prompts,
            visualize_dir=os.path.join(args.output_path, "samples"),
            visualize_interval=args.visualize_interval,
            rank_id=rank_id,
        )
        callbacks.append(visualizer)

    if rank_id == 0:
        num_params_vae, num_params_trainable_vae = count_params(vae)
        num_params_network, num_params_trainable_network = count_params(network)
        num_params_text_encoder, num_params_trainable_text_encoder = count_params(text_encoder)
        num_params = num_params_vae + num_params_network + num_params_text_encoder
        num_params_trainable = (
            num_params_trainable_vae + num_params_trainable_network + num_params_trainable_text_encoder
        )
        key_info = "Key Settings:\n" + "=" * 50 + "\n"
        key_info += "\n".join(
            [
                f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.mode}",
                f"JIT level: {args.jit_level}",
                f"Distributed mode: {args.use_parallel}",
                f"Data path: {args.json_path}",
                f"Number of samples: {len(dataset)}",
                f"Num params: {num_params:,} (network: {num_params_network:,}, vae: {num_params_vae:,}, text_encoder: {num_params_text_encoder:,})",
                f"Num trainable params: {num_params_trainable:,}",
                f"Model type: {args.dtype}",
                f"Learning rate: {start_learning_rate:.7f}",
                f"Batch size: {args.batch_size}",
                f"Image size: {image_size}",
                f"Weight decay: {args.weight_decay}",
                f"Grad accumulation steps: {args.gradient_accumulation_steps}",
                f"Num epochs: {args.epochs}",
                f"Loss scaler: {args.loss_scaler_type}",
                f"Init loss scale: {args.init_loss_scale}",
                f"Grad clipping: {args.clip_grad}",
                f"Max grad norm: {args.max_grad_norm}",
                f"EMA: {args.use_ema}",
                f"Multi-Scale Training: {args.multi_scale}",
                f"Enable flash attention: {args.enable_flash_attention}",
                f"Visualization: {args.visualize}",
            ]
        )
        key_info += "\n" + "=" * 50
        print(key_info)

        with open(os.path.join(args.output_path, "args.yaml"), "w") as f:
            yaml.safe_dump(vars(args), stream=f, default_flow_style=False, sort_keys=False)

    # 6. train
    logger.info("Start training...")
    model.train(args.epochs, data_generator, callbacks=callbacks)


if __name__ == "__main__":
    args = parse_args()
    main(args)
