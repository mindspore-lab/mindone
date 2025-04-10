import argparse
import os
import sys

from .utils import str2bool

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.insert(0, mindone_lib_path)
from mindone.utils.config import parse_bool_str


def parse_train_args(parser):
    parser.add_argument(
        "--ms_mode", type=int, default=1, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=1)"
    )
    parser.add_argument(
        "--max_device_memory", type=str, default=None, help="e.g. `59GB` for Ascend Atlas 800T A2 machines"
    )
    parser.add_argument(
        "--jit_level",
        default="O0",
        type=str,
        choices=["O0", "O1", "O2"],
        help="Used to control the compilation optimization level. Supports ['O0', 'O1', 'O2']."
        "O0: Except for optimizations that may affect functionality, all other optimizations are turned off, adopt KernelByKernel execution mode."
        "O1: Using commonly used optimizations and automatic operator fusion optimizations, adopt KernelByKernel execution mode."
        "O2: Ultimate performance optimization, adopt Sink execution mode.",
    )
    parser.add_argument("--use_parallel", default=False, type=str2bool, help="use parallel")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--vae_checkpoint",
        type=parse_bool_str,
        default="model/vae-2972c247.ckpt",
        help="VAE checkpoint file path which is used to load vae weight.",
    )
    parser.add_argument(
        "--var_checkpoint",
        type=parse_bool_str,
        default="model/var-d16.ckpt",
        help="VAR checkpoint file path which is used to load var weight.",
    )
    parser.add_argument(
        "--dtype",
        default="fp16",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        help="what data type to use for latte. Default is `fp32`, which corresponds to ms.float16",
    )
    parser.add_argument("--output_path", default="outputs", type=str, help="output path to save training results")
    parser.add_argument(
        "--log_level",
        type=str,
        default="logging.INFO",
        help="log level, options: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR",
    )
    parser.add_argument(
        "--log_interval",
        default=1,
        type=int,
        help="log interval in the unit of data sink size. E.g. if data sink size = 10, log_interval=2, log every 20 steps",
    )
    parser.add_argument("--drop_overflow_update", default=True, type=str2bool, help="drop overflow update")
    parser.add_argument("--loss_scaler_type", default="dynamic", type=str, help="dynamic or static")
    parser.add_argument("--init_loss_scale", default=65536, type=float, help="loss scale")
    parser.add_argument("--loss_scale_factor", default=2, type=float, help="loss scale factor")
    parser.add_argument("--scale_window", default=2000, type=float, help="scale window")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="gradient accumulation steps")
    parser.add_argument("--use_ema", default=False, type=str2bool, help="whether use EMA")
    parser.add_argument(
        "--ema_decay",
        default=0.9999,
        type=float,
        help="EMA decay ratio, smaller value raises more importance to the current model weight.",
    )
    parser.add_argument("--clip_grad", default=False, type=str2bool, help="whether apply gradient clipping")
    parser.add_argument(
        "--max_grad_norm",
        default=2.0,
        type=float,
        help="max gradient norm for clipping, effective when `clip_grad` enabled.",
    )

    # data
    parser.add_argument(
        "--data_path",
        type=str,
        default="/path/to/imagenet",
        help="path to imagenet.",
    )
    parser.add_argument("--pn", default="1_2_3_4_5_6_8_10_13_16", type=str, help="patch_nums")
    parser.add_argument("--patch_size", default="16", type=int, help="patch size")
    parser.add_argument(
        "--mid_reso",
        default="1.125",
        type=float,
        help="aug: first resize to mid_reso = 1.125 * data_load_reso, then crop to data_load_reso, data_load_reso=max(patch_nums) * patch_size",
    )
    parser.add_argument("--hflip", default="False", type=str2bool, help="augmentation: horizontal flip")
    parser.add_argument(
        "--num_parallel_workers",
        default=8,
        type=int,
        help="The number of workers used for data transformations. Default is 8.",
    )
    parser.add_argument(
        "--num_classes",
        default=1000,
        type=int,
        help="dataset num classes. Default is 1000 for imagenet-1k.",
    )

    # VAR
    parser.add_argument("--depth", type=int, default=16, help="VAR depth")
    parser.add_argument("--ini", type=float, default=-1.0, help="automated model parameter initialization")
    parser.add_argument("--hd", type=float, default=0.02, help="head.w *= hd")
    parser.add_argument("--aln", type=float, default=0.5, help="the multiplier of ada_lin.w's initialization")
    parser.add_argument(
        "--alng", type=float, default=1e-5, help="the multiplier of ada_lin.w[gamma channels]'s initialization"
    )

    parser.add_argument("--batch_size", default=32, type=int, help="batch size")

    # optimizer
    parser.add_argument("--optim", default="adamw", type=str, help="optimizer")
    parser.add_argument(
        "--betas", type=float, default=[0.9, 0.95], help="Specify the [beta1, beta2] parameter for the Adam optimizer."
    )
    parser.add_argument("--weight_decay", default=0.05, type=float, help="Weight decay.")
    parser.add_argument("--ls", default=0.0, type=float, help="smooth.")

    # learning rate
    parser.add_argument("--epochs", default=250, type=int, help="epoch")
    parser.add_argument("--wp", default=0, type=int, help="warm up epoch")
    parser.add_argument("--wp0", default=0.005, type=float, help="initial lr ratio at the begging of lr warm up")
    parser.add_argument("--wpe", default=0.01, type=float, help="final lr ratio at the end of training")
    parser.add_argument("--scheduler", default="lin0", type=str, help="scheduler.")
    parser.add_argument("--tblr", default=1e-4, type=float, help="base lr")

    # other hps
    parser.add_argument(
        "--resume",
        default=False,
        type=parse_bool_str,
        help="string: path to resume checkpoint."
        "bool False: not resuming.(default=False)."
        "bool True: ModelArts auto resume training.",
    )
    parser.add_argument("--saln", default=False, type=bool, help="whether to use shared adaln")
    parser.add_argument("--anorm", default=True, type=bool, help="whether to use L2 normalized attention")
    parser.add_argument("--ckpt_save_interval", default=1, type=int, help="save checkpoint every this epochs")
    parser.add_argument(
        "--ckpt_save_steps",
        default=-1,
        type=int,
        help="save checkpoint every this steps. If -1, use ckpt_save_interval will be used.",
    )
    parser.add_argument("--ckpt_max_keep", default=10, type=int, help="Maximum number of checkpoints to keep")

    # progressive training
    parser.add_argument(
        "--pg", default=0.0, type=float, help=">0 for use progressive training during [0%, this] of training"
    )
    parser.add_argument(
        "--pg0",
        default=4,
        type=int,
        help="progressive initial stage, 0: from the 1st token map, 1: from the 2nd token map, etc",
    )
    parser.add_argument("--pgwp", default=0, type=float, help="num of warmup epochs at each progressive stage")

    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    parser = parse_train_args(parser)
    args = parser.parse_args()
    if args.pn == "256":
        args.pn = "1_2_3_4_5_6_8_10_13_16"
    elif args.pn == "512":
        args.pn = "1_2_3_4_6_9_13_18_24_32"
    elif args.pn == "1024":
        args.pn = "1_2_3_4_5_7_9_12_16_21_27_36_48_64"
    args.patch_nums = tuple(map(int, args.pn.replace("-", "_").split("_")))
    args.resos = tuple(pn * args.patch_size for pn in args.patch_nums)
    args.data_load_reso = max(args.resos)

    if args.wp == 0:
        args.wp = args.epochs * 1 / 50

    # update args: progressive training
    if args.pgwp == 0:
        args.pgwp = args.epochs * 1 / 300
    if args.pg > 0:
        args.scheduler = f"lin{args.pg:g}"

    return args
