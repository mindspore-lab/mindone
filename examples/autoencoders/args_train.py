import argparse
import logging
import os

import yaml

from mindone.utils.config import str2bool

logger = logging.getLogger()


def _check_cfgs_in_parser(cfgs: dict, parser: argparse.ArgumentParser):
    actions_dest = [action.dest for action in parser._actions]
    defaults_key = parser._defaults.keys()
    cfg_keys = list(cfgs.keys())
    for k in cfg_keys:
        if k not in actions_dest and k not in defaults_key:
            raise KeyError(f"{k} does not exist in ArgumentParser!")
            cfgs.pop(k)
    return cfgs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="",
        type=str,
        help="path to load a config yaml file that describes the training recipes which will override the default arguments",
    )
    parser.add_argument(
        "--model_config",
        default="configs/train/autoencoder_kl_f8.yaml",
        type=str,
        help="model arch config",
    )
    parser.add_argument("--use_parallel", default=False, type=str2bool, help="use parallel")
    parser.add_argument(
        "--output_path", default="outputs/vae_train", type=str, help="output directory to save training results"
    )
    parser.add_argument(
        "--resume",
        default=False,
        type=str,
        help="resume training, can set True or path to resume checkpoint.(default=False)",
    )
    # ms
    parser.add_argument("--mode", default=0, type=int, help="Specify the mode: 0 for graph mode, 1 for pynative mode")
    parser.add_argument("--device_target", type=str, default="Ascend", help="Ascend or GPU")
    parser.add_argument(
        "--jit_level",
        default="O2",
        type=str,
        choices=["O0", "O1", "O2"],
        help="Used to control the compilation optimization level. Supports ['O0', 'O1', 'O2']."
        "O0: Except for optimizations that may affect functionality, all other optimizations are turned off, adopt KernelByKernel execution mode."
        "O1: Using commonly used optimizations and automatic operator fusion optimizations, adopt KernelByKernel execution mode."
        "O2: Ultimate performance optimization, adopt Sink execution mode.",
    )
    parser.add_argument("--profile", default=False, type=str2bool, help="Profile or not")
    # data
    parser.add_argument("--data_path", default="dataset", type=str, help="data path")
    parser.add_argument("--csv_path", default=None, type=str, help="path to csv annotation file")
    parser.add_argument("--dataset_sink_mode", default=False, type=str2bool, help="sink mode")
    parser.add_argument("--shuffle", default=True, type=str2bool, help="data shuffle")
    parser.add_argument("--num_parallel_workers", default=8, type=int, help="num workers for data loading")
    parser.add_argument("--size", default=384, type=int, help="image rescale size")
    parser.add_argument("--crop_size", default=256, type=int, help="image crop size")
    parser.add_argument("--random_crop", default=False, type=str2bool, help="random crop for data augmentation")
    parser.add_argument("--flip", default=False, type=str2bool, help="horizontal flip for data augmentation")

    # optim
    parser.add_argument(
        "--use_discriminator",
        default=False,
        type=str2bool,
        help="Phase 1 training does not use discriminator, set False to reduce memory cost in graph mode.",
    )
    parser.add_argument(
        "--dtype", default="fp32", type=str, choices=["fp32", "fp16", "bf16"], help="data type for mixed precision"
    )
    parser.add_argument("--optim", default="adam", type=str, help="optimizer")
    parser.add_argument(
        "--betas",
        type=float,
        default=(0.5, 0.9),  # [0.9, 0.999]
        help="Specify the [beta1, beta2] parameter for the Adam or AdamW optimizer.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay.")
    parser.add_argument(
        "--group_strategy",
        type=str,
        default="norm_and_bias",
        help="Grouping strategy for weight decay. If `norm_and_bias`, weight decay filter list is [beta, gamma, bias]. \
                If None, filter list is [layernorm, bias]. Default: norm_and_bias",
    )
    parser.add_argument("--seed", default=3407, type=int, help="data path")
    parser.add_argument("--warmup_steps", default=1000, type=int, help="warmup steps")
    parser.add_argument("--batch_size", default=4, type=int, help="batch size")
    parser.add_argument("--log_interval", default=1, type=int, help="log interval")
    parser.add_argument(
        "--base_learning_rate",
        default=4.5e-06,
        type=float,
        help="base learning rate, can be scaled by global batch size",
    )
    parser.add_argument("--end_learning_rate", default=1e-8, type=float, help="The end learning rate for Adam.")
    parser.add_argument(
        "--scale_lr", default=True, type=str2bool, help="scale base-lr by ngpu * batch_size * n_accumulate"
    )
    parser.add_argument("--decay_steps", default=0, type=int, help="lr decay steps.")
    parser.add_argument(
        "--scheduler", default="cosine_decay", type=str, help="scheduler. option: constant, cosine_decay, "
    )
    parser.add_argument("--epochs", default=10, type=int, help="epochs")
    parser.add_argument("--loss_scaler_type", default="static", type=str, help="dynamic or static")
    parser.add_argument("--init_loss_scale", default=1024, type=float, help="loss scale")
    parser.add_argument("--loss_scale_factor", default=2, type=float, help="loss scale factor")
    parser.add_argument("--scale_window", default=1000, type=float, help="scale window")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="gradient accumulation steps")
    parser.add_argument("--use_ema", default=False, type=str2bool, help="whether use EMA")
    parser.add_argument("--ema_decay", default=0.9999, type=float, help="EMA decay")
    parser.add_argument("--clip_grad", default=False, type=str2bool, help="whether apply gradient clipping")
    parser.add_argument("--drop_overflow_update", default=True, type=str2bool, help="drop overflow update")
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="max gradient norm for clipping, effective when `clip_grad` enabled.",
    )
    parser.add_argument("--ckpt_max_keep", default=10, type=int, help="Maximum number of checkpoints to keep")
    parser.add_argument("--ckpt_save_interval", default=1, type=int, help="save checkpoint every this epochs or steps")
    parser.add_argument(
        "--step_mode",
        default=False,
        type=str2bool,
        help="whether save ckpt by steps. If False, save ckpt by epochs.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="logging.INFO",
        help="log level, options: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR",
    )

    abs_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ""))
    default_args = parser.parse_args()
    if default_args.config:
        default_args.config = os.path.join(abs_path, default_args.config)
        with open(default_args.config, "r") as f:
            cfg = yaml.safe_load(f)
            cfg = _check_cfgs_in_parser(cfg, parser)
            parser.set_defaults(**cfg)
    args = parser.parse_args()
    # args.model_config = os.path.join(abs_path, args.model_config)

    logger.info(args)
    return args
