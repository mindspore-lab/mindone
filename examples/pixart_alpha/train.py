#!/usr/bin/env python
import argparse
import logging
import os
import sys
from typing import Tuple

import yaml

import mindspore as ms
import mindspore.nn as nn
from mindspore import Model
from mindspore.communication import get_group_size, get_rank, init
from mindspore.dataset import GeneratorDataset

# TODO: remove in future when mindone is ready for install
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)

from pixart.dataset import LatentDataset
from pixart.diffusion import create_diffusion
from pixart.modules.pixart import PixArt_XL_2, PixArtMS_XL_2
from pixart.pipelines import NetworkWithLoss
from pixart.utils import (
    EMA,
    LossMonitor,
    SaveCkptCallback,
    TimeMonitor,
    auto_scale_lr,
    check_cfgs_in_parser,
    count_params,
    load_ckpt_params,
    str2bool,
)

from mindone.trainers.callback import OverflowMonitor
from mindone.trainers.lr_schedule import create_scheduler
from mindone.trainers.optim import create_optimizer
from mindone.trainers.train_step import TrainOneStepWrapper
from mindone.utils.amp import auto_mixed_precision
from mindone.utils.logger import set_logger
from mindone.utils.seed import set_random_seed

logger = logging.getLogger(__name__)


def init_env(args) -> Tuple[int, int]:
    set_random_seed(args.seed)
    ms.set_context(mode=args.mode, device_target=args.device_target, jit_config=dict(jit_level="O0"))
    if args.use_parallel:
        init()
        device_num = get_group_size()
        rank_id = get_rank()
        ms.set_auto_parallel_context(
            parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True, device_num=device_num
        )
    else:
        device_num, rank_id = 1, 0

    return device_num, rank_id


def parse_args():
    parser = argparse.ArgumentParser(
        description="PixArt-Alpha Training script", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-c",
        "--config",
        help="path to load a config yaml file that describes the setting which will override the default arguments",
    )
    parser.add_argument("--csv_path", required=True, help="path to csv annotation file.")
    parser.add_argument("--latent_dir", required=True, help="directory storing the VAE latent.")
    parser.add_argument("--text_emb_dir", required=True, help="directory storing the text embedding")
    parser.add_argument("--path_column", default="dir", help="column name of image path in csv file.")
    parser.add_argument("--output_path", default="./output", help="output directory to save the training result.")

    parser.add_argument("--sample_size", default=64, type=int, choices=[128, 64, 32], help="network sample size")
    parser.add_argument("--batch_size", default=64, type=int, help="training batch size")
    parser.add_argument("--num_parallel_workers", default=4, type=int, help="num workers for data loading")
    parser.add_argument(
        "--checkpoint", default="models/PixArt-XL-2-512x512.ckpt", help="the path to the PixArt checkpoint."
    )
    parser.add_argument(
        "--sd_scale_factor", default=0.18215, type=float, help="VAE scale factor of Stable Diffusion network."
    )

    parser.add_argument("--device_target", default="Ascend", choices=["CPU", "GPU", "Ascend"], help="Device target")
    parser.add_argument("--mode", default=0, type=int, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1)")
    parser.add_argument("--seed", default=42, type=int, help="Inference seed")

    parser.add_argument(
        "--enable_flash_attention", default=True, type=str2bool, help="whether to enable flash attention."
    )
    parser.add_argument(
        "--dtype", default="fp16", choices=["bf16", "fp16", "fp32"], help="what data type to use for PixArt."
    )
    parser.add_argument("--scheduler", default="constant", type=str, help="scheduler.")
    parser.add_argument("--start_learning_rate", default=2e-5, type=float, help="The learning rate.")
    parser.add_argument("--warmup_steps", default=1000, type=int, help="Warmup steps")
    parser.add_argument("--epochs", default=200, type=int, help="number of total training epochs.")
    parser.add_argument("--optim", default="adamw", type=str, help="optimizer name.")
    parser.add_argument(
        "--optim_eps", default=1.0e-10, type=float, help="Specify the eps parameter for the AdamW optimizer."
    )
    parser.add_argument("--weight_decay", default=0.03, type=float, help="Weight decay.")
    parser.add_argument(
        "--auto_lr", default="sqrt", choices=["", "sqrt", "linear"], help="Use auto-scaled learning rate"
    )
    parser.add_argument(
        "--loss_scaler_type", default="dynamic", choices=["static", "dynamic"], help="dynamic or static"
    )
    parser.add_argument("--init_loss_scale", default=65536.0, type=float, help="loss scale")
    parser.add_argument("--scale_window", default=1000, type=int, help="loss scale window")
    parser.add_argument("--loss_scale_factor", default=2.0, type=float, help="loss scale factor")
    parser.add_argument("--use_ema", default=False, type=str2bool, help="whether to use EMA")
    parser.add_argument("--ema_rate", default=0.9999, type=float, help="EMA Rate.")
    parser.add_argument("--drop_overflow_update", default=True, type=str2bool, help="drop overflow update")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="gradient accumulation steps")
    parser.add_argument("--clip_grad", default=False, type=str2bool, help="whether apply gradient clipping")
    parser.add_argument(
        "--max_grad_norm",
        default=0.01,
        type=float,
        help="max gradient norm for clipping, effective when `clip_grad` enabled.",
    )
    parser.add_argument("--ckpt_max_keep", default=5, type=int, help="Maximum number of checkpoints to keep")
    parser.add_argument("--ckpt_save_interval", default=1, type=int, help="save checkpoint every this epochs or steps")
    parser.add_argument("--log_loss_interval", default=1, type=int, help="log interval of loss value")

    parser.add_argument("--use_parallel", default=False, type=str2bool, help="use parallel")
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
    set_logger(output_dir=os.path.join(args.output_path, "logs", f"rank_{rank_id}"))

    # 2. model initialize and weight loading
    # 2.1 PixArt
    image_size = args.sample_size * 8
    logger.info(f"{image_size}x{image_size} init")

    if args.sample_size == 128:
        network = PixArtMS_XL_2(block_kwargs={"enable_flash_attention": args.enable_flash_attention})
    else:
        network = PixArt_XL_2(
            input_size=args.sample_size, block_kwargs={"enable_flash_attention": args.enable_flash_attention}
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

    # 2.2 Sampling
    diffusion = create_diffusion(timestep_respacing="")

    # 3. build training network
    latent_diffusion_with_loss = NetworkWithLoss(
        network, diffusion, vae=None, text_encoder=None, scale_factor=args.sd_scale_factor
    )

    # 4. build dataset
    dataset = LatentDataset(args.csv_path, args.latent_dir, args.text_emb_dir, path_column=args.path_column)
    dataset = GeneratorDataset(
        dataset,
        column_names=["x", "text_emb", "text_mask"],
        shuffle=True,
        num_parallel_workers=args.num_parallel_workers,
    )
    dataset = dataset.batch(args.batch_size, drop_remainder=True)
    dataset_size = dataset.get_dataset_size()

    # 5. build training utils: lr, optim, callbacks, trainer
    # 5.1 learning rate
    if args.auto_lr:
        start_learning_rate = auto_scale_lr(
            args.batch_size * args.gradient_accumulation_steps * device_num, args.start_learning_rate, rule=args.auto_lr
        )
    else:
        start_learning_rate = args.start_learning_rate

    lr = create_scheduler(
        steps_per_epoch=dataset_size,
        name=args.scheduler,
        lr=start_learning_rate,
        warmup_steps=args.warmup_steps,
        num_epochs=args.epochs,
    )

    # 5.2 optimizer
    optimizer = create_optimizer(
        latent_diffusion_with_loss.trainable_params(),
        name=args.optim,
        lr=lr,
        weight_decay=args.weight_decay,
        eps=args.optim_eps,
        group_strategy="all",
    )

    if args.loss_scaler_type == "dynamic":
        loss_scaler = nn.DynamicLossScaleUpdateCell(
            loss_scale_value=args.init_loss_scale, scale_factor=args.loss_scale_factor, scale_window=args.scale_window
        )
    else:
        loss_scaler = nn.FixedLossScaleUpdateCell(args.init_loss_scale)

    # 5.3 trainer (standalone and distributed)
    if args.use_ema:
        ema = EMA(latent_diffusion_with_loss.network, ema_decay=args.ema_rate)
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

    model = Model(net_with_grads)

    # 5.4 callbacks
    callbacks = [
        TimeMonitor(),
        OverflowMonitor(),
        LossMonitor(log_interval=args.log_loss_interval),
        SaveCkptCallback(
            rank_id=rank_id,
            output_dir=os.path.join(args.output_path, "ckpt"),
            ckpt_max_keep=args.ckpt_max_keep,
            ckpt_save_interval=args.ckpt_save_interval,
            save_ema=args.use_ema,
        ),
    ]

    if rank_id == 0:
        num_params, num_params_trainable = count_params(latent_diffusion_with_loss)
        key_info = "Key Settings:\n" + "=" * 50 + "\n"
        key_info += "\n".join(
            [
                f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.mode}",
                f"Distributed mode: {args.use_parallel}",
                f"Data path: {args.csv_path}",
                f"Num params: {num_params:,} (PixArt: {num_params:,})",
                f"Num trainable params: {num_params_trainable:,}",
                f"Model type: {args.dtype}",
                f"Learning rate: {args.start_learning_rate}",
                f"Batch size: {args.batch_size}",
                f"Image size: {image_size}",
                f"Weight decay: {args.weight_decay}",
                f"Grad accumulation steps: {args.gradient_accumulation_steps}",
                f"Num epochs: {args.epochs}",
                f"Total training steps: {dataset_size * args.epochs:,}",
                f"Loss scaler: {args.loss_scaler_type}",
                f"Init loss scale: {args.init_loss_scale}",
                f"Grad clipping: {args.clip_grad}",
                f"Max grad norm: {args.max_grad_norm}",
                f"EMA: {args.use_ema}",
                f"Enable flash attention: {args.enable_flash_attention}",
            ]
        )
        key_info += "\n" + "=" * 50
        print(key_info)

        with open(os.path.join(args.output_path, "args.yaml"), "w") as f:
            yaml.safe_dump(vars(args), stream=f, default_flow_style=False, sort_keys=False)

    # 6. train
    logger.info("Start training...")
    model.train(args.epochs, dataset, callbacks=callbacks)


if __name__ == "__main__":
    args = parse_args()
    main(args)
