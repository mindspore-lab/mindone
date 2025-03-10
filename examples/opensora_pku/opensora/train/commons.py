import argparse
import logging
import os

import yaml

from mindspore import nn
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell

from mindone.utils.config import str2bool

logger = logging.getLogger(__name__)


def _check_cfgs_in_parser(cfgs: dict, parser: argparse.ArgumentParser):
    actions_dest = [action.dest for action in parser._actions]
    defaults_key = parser._defaults.keys()
    for k in cfgs.keys():
        if k not in actions_dest and k not in defaults_key:
            raise KeyError(f"{k} does not exist in ArgumentParser!")


def parse_train_args(parser):
    parser.add_argument(
        "--config",
        "-c",
        default="",
        type=str,
        help="path to load a config yaml file that describes the training recipes which will override the default arguments",
    )
    # the following args's defualt value will be overrided if specified in config yaml
    #################################################################################
    #                      MindSpore Envs and Mode                                  #
    #################################################################################
    parser.add_argument("--device", type=str, default="Ascend", help="Ascend or GPU")
    parser.add_argument("--max_device_memory", type=str, default=None, help="e.g. `30GB` for 910a, `59GB` for 910b")
    parser.add_argument("--mode", default=1, type=int, help="Specify the mode: 0 for graph mode, 1 for pynative mode")
    parser.add_argument(
        "--jit_syntax_level", default="strict", type=str, help="Specify syntax level for graph mode: strict or lax"
    )
    parser.add_argument("--use_parallel", default=False, type=str2bool, help="use parallel")
    parser.add_argument(
        "--parallel_mode",
        default="data",
        type=str,
        choices=["data", "optim", "semi", "zero"],
        help="parallel mode: data, optim, zero",
    )
    parser.add_argument(
        "--zero_stage",
        default=2,
        type=int,
        help="run parallelism like deepspeed, supporting zero0, zero1, zero2, and zero3, if parallel_mode==zero",
    )
    parser.add_argument("--comm_fusion", default=True, type=str2bool)
    parser.add_argument("--seed", default=3407, type=int, help="data path")
    parser.add_argument(
        "--mempool_block_size",
        type=str,
        default="9GB",
        help="Set the size of the memory pool block in PyNative mode for devices. ",
    )
    parser.add_argument(
        "--optimizer_weight_shard_size",
        type=int,
        default=8,
        help="Set the size of the communication domain split by the optimizer weight. ",
    )

    #################################################################################
    #                                   Optimizers                                  #
    #################################################################################
    parser.add_argument("--optim", default="adamw_re", type=str, help="optimizer, use adamw from mindcv by default")
    parser.add_argument(
        "--betas",
        type=float,
        nargs="+",
        default=[0.9, 0.999],
        help="Specify the [beta1, beta2] parameter for the AdamW optimizer.",
    )
    parser.add_argument(
        "--optim_eps", type=float, default=1e-8, help="Specify the eps parameter for the AdamW optimizer."
    )
    parser.add_argument(
        "--group_strategy",
        type=str,
        default="norm_and_bias",
        help="Grouping strategy for weight decay. If `norm_and_bias`, weight decay filter list is [beta, gamma, bias]. \
                If None, filter list is [layernorm, bias]. Default: norm_and_bias",
    )
    parser.add_argument("--clip_grad", default=False, type=str2bool, help="whether apply gradient clipping")
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="max gradient norm for clipping, effective when `clip_grad` enabled.",
    )
    parser.add_argument("--use_ema", default=False, type=str2bool, help="whether use EMA")
    parser.add_argument("--ema_offload", default=True, type=str2bool, help="whether use EMA CPU offload")
    parser.add_argument("--ema_decay", default=0.999, type=float, help="EMA decay")

    #################################################################################
    #                                Learning Rate                                  #
    #################################################################################
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="gradient accumulation steps")
    parser.add_argument("--weight_decay", default=1e-2, type=float, help="Weight decay.")
    parser.add_argument("--lr_warmup_steps", default=1000, type=int, help="warmup steps")
    parser.add_argument(
        "--start_learning_rate", default=1e-5, type=float, help="The initial learning rate for the optimizer."
    )
    parser.add_argument(
        "--end_learning_rate", default=1e-7, type=float, help="The end learning rate for the optimizer."
    )
    parser.add_argument("--lr_decay_steps", default=0, type=int, help="lr decay steps.")
    parser.add_argument("--lr_scheduler", default="constant", type=str, help="scheduler.")
    parser.add_argument(
        "--scale_lr",
        default=False,
        type=str2bool,
        help="Specify whether to scale the learning rate based on the batch size, gradient accumulation steps, and n cards.",
    )

    #################################################################################
    #                           Dataset and DataLoader                              #
    #################################################################################
    parser.add_argument("--train_batch_size", default=10, type=int, help="train batch size")
    parser.add_argument("--val_batch_size", default=1, type=int, help="validation batch size")
    parser.add_argument("--dataset_sink_mode", default=False, type=str2bool, help="sink mode")
    parser.add_argument("--sink_size", default=-1, type=int, help="dataset sink size. If -1, sink size = dataset size.")
    parser.add_argument(
        "--num_train_epochs",
        default=10,
        type=int,
        help="epochs. When epochs is specified, the total number of training steps = epochs x num_batches",
    )
    parser.add_argument("--dataloader_num_workers", default=12, type=int, help="num workers for dataloder")
    parser.add_argument("--max_rowsize", default=32, type=int, help="max rowsize for data loading")

    #################################################################################
    #                         Mixed Precision: Loss scaler etc.                     #
    #################################################################################
    parser.add_argument("--init_loss_scale", default=65536, type=float, help="loss scale")
    parser.add_argument("--loss_scale_factor", default=2, type=float, help="loss scale factor")
    parser.add_argument("--scale_window", default=2000, type=float, help="scale window")
    parser.add_argument(
        "--precision",
        default="fp16",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        help="what data type to use for model. Default is `fp16`, which corresponds to ms.float16",
    )
    parser.add_argument(
        "--amp_level",
        default="O2",
        type=str,
        help="mindspore amp level, O1: most fp32, only layers in whitelist compute in fp16 (dense, conv, etc); \
            O2: most fp16, only layers in blacklist compute in fp32 (batch norm etc)",
    )
    parser.add_argument("--drop_overflow_update", default=True, type=str2bool, help="drop overflow update")
    parser.add_argument("--loss_scaler_type", default="dynamic", type=str, help="dynamic or static")
    #################################################################################
    #                                 Model Optimization                            #
    #################################################################################
    # parser.add_argument("--image_size", default=256, type=int, help="the image size used to initiate model")
    # parser.add_argument("--num_frames", default=16, type=int, help="the num of frames used to initiate model")
    # parser.add_argument("--frame_stride", default=3, type=int, help="frame sampling stride")
    # parser.add_argument(
    #     "--disable_flip",
    #     default=True,
    #     type=str2bool,
    #     help="disable random flip video (to avoid motion direction and text mismatch)",
    # )
    parser.add_argument(
        "--use_recompute",
        default=False,
        type=str2bool,
        help="whether use recompute.",
    )

    #################################################################################
    #                                Training Callbacks                            #
    #################################################################################
    parser.add_argument(
        "--resume_from_checkpoint",
        default=False,
        type=str,
        help="It can be a string for path to resume checkpoint, or a bool False for not resuming, or a bool True to use default train_resume.ckpt.",
    )
    parser.add_argument("--ckpt_save_interval", default=1, type=int, help="save checkpoint every this epochs or steps")
    parser.add_argument("--ckpt_max_keep", default=10, type=int, help="Maximum number of checkpoints to keep")
    parser.add_argument(
        "--step_mode",
        default=False,
        type=str2bool,
        help="whether save ckpt by steps. If False, save ckpt by epochs.",
    )
    parser.add_argument(
        "--save_ema_only",
        default=False,
        type=str2bool,
        help="whether save ema ckpt only. If False, and when ema during training is enabled, it will save both ema and non-ema.ckpt",
    )
    parser.add_argument(
        "--validate",
        default=False,
        type=str2bool,
        help="whether to compute the validation set loss during training",
    )
    parser.add_argument("--val_interval", default=1, type=int, help="Validation frequency in epochs")
    parser.add_argument("--profile", default=False, type=str2bool, help="Profile time analysis or not")
    parser.add_argument("--profile_memory", default=False, type=str2bool, help="Profile memory analysis or not")
    parser.add_argument(
        "--log_level",
        type=str,
        default="logging.INFO",
        help="log level, options: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR",
    )
    parser.add_argument("--log_interval", type=int, default=1, help="log interval")
    return parser


def parse_args(default_parse_args=parse_train_args, additional_parse_args=None):
    parser = argparse.ArgumentParser()
    parser = default_parse_args(parser)
    if additional_parse_args:
        parser = additional_parse_args(parser)
    abs_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ""))
    default_args = parser.parse_args()
    if default_args.config:
        default_args.config = os.path.join(abs_path, default_args.config)
        with open(default_args.config, "r") as f:
            cfg = yaml.safe_load(f)
            _check_cfgs_in_parser(cfg, parser)
            parser.set_defaults(**cfg)
    args = parser.parse_args()

    print(args)

    return args


def create_loss_scaler(args):
    if args.loss_scaler_type == "dynamic":
        loss_scaler = DynamicLossScaleUpdateCell(
            loss_scale_value=args.init_loss_scale, scale_factor=args.loss_scale_factor, scale_window=args.scale_window
        )
    elif args.loss_scaler_type == "static":
        loss_scaler = nn.FixedLossScaleUpdateCell(args.init_loss_scale)
    else:
        raise ValueError

    return loss_scaler
