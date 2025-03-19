import argparse
import logging
import os
import sys

import yaml

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.insert(0, mindone_lib_path)

from mindone.utils.config import str2bool
from mindone.utils.misc import to_abspath

logger = logging.getLogger()


def _check_cfgs_in_parser(cfgs: dict, parser: argparse.ArgumentParser):
    actions_dest = [action.dest for action in parser._actions]
    defaults_key = parser._defaults.keys()
    for k in cfgs.keys():
        if k not in actions_dest and k not in defaults_key:
            raise KeyError(f"{k} does not exist in ArgumentParser!")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="",
        type=str,
        help="path to load a config yaml file that describes the training recipes which will override the default arguments",
    )
    # the following args's defualt value will be overrided if specified in config yaml
    parser.add_argument("--model_config", default="configs/training_1024_v1.0.yaml", type=str, help="model config path")
    parser.add_argument("--data_dir", default="dataset", type=str, help="path to video root folder")
    parser.add_argument("--text_emb_dir", default="text embedding", type=str, help="path to text embedding root folder")
    
    parser.add_argument(
        "--csv_path",
        default=None,
        type=str,
        help="path to csv annotation file. If None, video_caption.csv is expected to live under `data_path`",
    )
    parser.add_argument("--output_path", default="output/", type=str, help="output directory to save training results")
    parser.add_argument(
        "--pretrained_model_path", default="", type=str, help="Specify the pretrained model from this checkpoint"
    )
    # ms
    parser.add_argument("--debug", type=str2bool, default=False, help="Execute in pynative debug mode. (pynative_synchronize=True)")
    parser.add_argument("--device_target", type=str, default="Ascend", help="Ascend or GPU")
    parser.add_argument("--max_device_memory", type=str, default=None, help="e.g. `30GB` for 910a, `59GB` for 910b")
    parser.add_argument("--mode", default=0, type=int, help="Specify the mode: 0 for graph mode, 1 for pynative mode")
    parser.add_argument("--use_parallel", default=False, type=str2bool, help="use parallel")
    parser.add_argument(
        "--jit_level",
        default="O0",
        type=str,
        choices=["O0", "O1", "O2"],
        help="Used to control the compilation optimization level. Supports [“O0”, “O1”, “O2”]."
        "O0: Except for optimizations that may affect functionality, all other optimizations are turned off, adopt KernelByKernel execution mode."
        "O1: Using commonly used optimizations and automatic operator fusion optimizations, adopt KernelByKernel execution mode."
        "O2: Ultimate performance optimization, adopt Sink execution mode.",
    )
    parser.add_argument(
        "--amp_level",
        default="O0",
        type=str,
        help="mindspore amp level, O1: most fp32, only layers in whitelist compute in fp16 (dense, conv, etc); \
            O2: most fp16, only layers in blacklist compute in fp32 (batch norm etc)",
    )
    # modelarts
    parser.add_argument("--enable_modelarts", default=False, type=str2bool, help="run codes in ModelArts platform")
    parser.add_argument("--num_workers", default=1, type=int, help="the number of modelarts workers")
    parser.add_argument(
        "--resume",
        default=False,
        type=str,
        help="It can be a string for path to resume checkpoint, or a bool False for not resuming.(default=False)",
    )
    # training hyper-params
    parser.add_argument("--unet_initialize_random", default=False, type=str2bool, help="initialize unet randomly")
    parser.add_argument("--optim", default="adamw", type=str, help="optimizer")
    parser.add_argument(
        "--betas",
        type=float,
        default=[0.9, 0.999],
        help="Specify the [beta1, beta2] parameter for the AdamW optimizer.",
    )
    parser.add_argument(
        "--optim_eps", type=float, default=1e-6, help="Specify the eps parameter for the AdamW optimizer."
    )
    parser.add_argument(
        "--group_strategy",
        type=str,
        default="norm_and_bias",
        help="Grouping strategy for weight decay. If `norm_and_bias`, weight decay filter list is [beta, gamma, bias]. \
                If None, filter list is [layernorm, bias]. Default: norm_and_bias",
    )

    parser.add_argument("--weight_decay", default=1e-6, type=float, help="Weight decay.")
    parser.add_argument("--seed", default=3407, type=int, help="data path")
    parser.add_argument("--warmup_steps", default=1000, type=int, help="warmup steps")
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    parser.add_argument(
        "--log_interval",
        default=1,
        type=int,
        help="log interval in the unit of data sink size.. E.g. if data sink size = 10, log_inteval=2, log every 20 steps",
    )
    parser.add_argument("--start_learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--end_learning_rate", default=1e-7, type=float, help="The end learning rate for Adam.")
    parser.add_argument("--decay_steps", default=0, type=int, help="lr decay steps.")
    parser.add_argument("--scheduler", default="cosine_decay", type=str, help="scheduler.")

    parser.add_argument("--dataset_sink_mode", default=False, type=str2bool, help="sink mode")
    parser.add_argument("--sink_size", default=-1, type=int, help="dataset sink size. If -1, sink size = dataset size.")
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        help="iterate the whole dataset for this much epochs in training. If -1, apply `train_steps`",
    )
    parser.add_argument("--train_steps", default=-1, type=int, help="number of training steps")
    parser.add_argument("--init_loss_scale", default=65536, type=float, help="loss scale")
    parser.add_argument("--loss_scale_factor", default=2, type=float, help="loss scale factor")
    parser.add_argument("--scale_window", default=1000, type=float, help="scale window")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="gradient accumulation steps")
    # parser.add_argument("--cond_stage_trainable", default=False, type=str2bool, help="whether text encoder is trainable")
    parser.add_argument("--use_ema", default=False, type=str2bool, help="whether use EMA")
    parser.add_argument("--clip_grad", default=False, type=str2bool, help="whether apply gradient clipping")
    parser.add_argument(
        "--use_recompute",
        default=None,
        type=str2bool,
        help="whether use recompute. If None, controlled by unet config.",
    )
    parser.add_argument(
        "--recompute_strategy", default=None, type=str, help="options: down_blocks, down_mm, up_mm, down_up"
    )
    parser.add_argument(
        "--enable_flash_attention",
        default=None,
        type=str2bool,
        help="whether enable flash attention. If not None, it will overwrite the value in model config yaml.",
    )
    parser.add_argument("--drop_overflow_update", default=True, type=str2bool, help="drop overflow update")
    parser.add_argument("--loss_scaler_type", default="dynamic", type=str, help="dynamic or static")
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="max gradient norm for clipping, effective when `clip_grad` enabled.",
    )

    parser.add_argument("--ckpt_save_epochs", default=100, type=int, help="save checkpoint every this epochs")
    parser.add_argument("--ckpt_save_steps", default=-1, type=int, help="save checkpoint every this steps")
    parser.add_argument("--ckpt_max_keep", default=10, type=int, help="Maximum number of checkpoints to keep")
    parser.add_argument(
        "--step_mode",
        default=None,
        type=str2bool,
        help="whether save ckpt by steps. If False, save ckpt by epochs.If None, it will set True if ckpt_save_steps>0 and dataset sink mode is disabled",
    )
    parser.add_argument("--random_crop", default=False, type=str2bool, help="random crop")
    parser.add_argument("--filter_small_size", default=True, type=str2bool, help="filter small images")
    parser.add_argument("--image_filter_size", default=256, type=int, help="image filter size")

    parser.add_argument("--profile", default=False, type=str2bool, help="Profile or not")
    parser.add_argument(
        "--log_level",
        type=str,
        default="logging.INFO",
        help="log level, options: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR",
    )
    parser.add_argument(
        "--vae_fp16",
        default=None,
        type=str2bool,
        help="whether use fp16 precision in vae. If None, it will be set by the value in stable diffusion config yaml",
    )
    parser.add_argument(
        "--amp_dtype",
        default="fp16",
        type=str,
        choices=["bf16", "fp16"],
        help="what computation data type to use for amp setting. Default is `fp16`, which corresponds to ms.float16",
    )
    # parser.add_argument("--image_size", default=256, type=int, help="image size")
    parser.add_argument("--resolution", required=True, type=int, nargs="+", help="resolution")
    parser.add_argument("--num_frames", default=16, type=int, help="num frames")
    parser.add_argument("--frame_stride", default=6, type=int, help="frame sampling stride")
    parser.add_argument(
        "--snr_gamma",
        default=None,
        type=float,
        help="min-SNR weighting used to improve diffusion training convergence."
        "If not None, it will overwrite the value defined in config yaml(stable_diffusion/v1-train_xx.yaml)."
        "If use, 5.0 is a common choice. To disable min-SNR weighting, set it to 0",
    )
    parser.add_argument("--num_parallel_workers", default=12, type=int, help="num workers for data loading")
    parser.add_argument("--video_column", default="video", type=str, help="name of column for videos saved in csv file")
    parser.add_argument(
        "--caption_column", default="caption", type=str, help="name of column for captions saved in csv file"
    )

    abs_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    default_args = parser.parse_args()
    if default_args.config:
        default_args.config = to_abspath(abs_path, default_args.config)
        with open(default_args.config, "r") as f:
            cfg = yaml.safe_load(f)
            _check_cfgs_in_parser(cfg, parser)
            parser.set_defaults(**cfg)
    args = parser.parse_args()
    args.model_config = to_abspath(abs_path, args.model_config)
    args.data_dir = to_abspath(abs_path, args.data_dir)
    args.text_emb_dir = to_abspath(abs_path, args.text_emb_dir)
    args.csv_path = to_abspath(abs_path, args.csv_path)
    args.output_path = to_abspath(abs_path, args.output_path)
    args.pretrained_model_path = to_abspath(abs_path, args.pretrained_model_path)
    print(args)

    return args
