import argparse
import logging
import os
from typing import Tuple

import yaml
from opensora.acceleration.parallel_states import get_sequence_parallel_state, initialize_sequence_parallel_state

import mindspore as ms
from mindspore import nn
from mindspore.communication.management import get_group_size, get_rank, init
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell

from mindone.utils.config import str2bool
from mindone.utils.seed import set_random_seed

logger = logging.getLogger()


def init_env(
    mode: int = ms.GRAPH_MODE,
    seed: int = 42,
    distributed: bool = False,
    max_device_memory: str = None,
    device_target: str = "Ascend",
    parallel_mode: str = "data",
    enable_dvm: bool = False,
    mempool_block_size: str = "9GB",
    global_bf16: bool = False,
    strategy_ckpt_save_file: str = "",
    optimizer_weight_shard_size: int = 8,
    sp_size: int = 1,
) -> Tuple[int, int, int]:
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
    ms.set_context(mempool_block_size=mempool_block_size)

    if max_device_memory is not None:
        ms.set_context(max_device_memory=max_device_memory)

    if distributed:
        ms.set_context(
            mode=mode,
            device_target=device_target,
            ascend_config={"precision_mode": "allow_fp32_to_fp16"},  # ms2.2.23 parallel needs
            # ascend_config={"precision_mode": "must_keep_origin_dtype"},  # TODO: tune
        )
        if parallel_mode == "optim":
            print("use optim parallel")
            ms.set_auto_parallel_context(
                parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL,
                parallel_optimizer_config={"optimizer_weight_shard_size": optimizer_weight_shard_size},
                enable_parallel_optimizer=True,
                strategy_ckpt_config={
                    "save_file": strategy_ckpt_save_file,
                    "only_trainable_params": False,
                },
            )
            init()
            device_num = get_group_size()
            rank_id = get_rank()
        elif parallel_mode == "data":
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

        var_info = ["device_num", "rank_id", "device_num / 8", "rank_id / 8"]
        var_value = [device_num, rank_id, int(device_num / 8), int(rank_id / 8)]
        logger.info(dict(zip(var_info, var_value)))

    else:
        device_num = 1
        rank_id = 0
        ms.set_context(
            mode=mode,
            device_target=device_target,
            ascend_config={"precision_mode": "allow_fp32_to_fp16"},  # TODO: tune for better precision
        )

    if enable_dvm:
        print("enable dvm")
        ms.set_context(enable_graph_kernel=True, graph_kernel_flags="--disable_cluster_ops=Pow,Select")
    if global_bf16:
        print("Using global bf16")
        ms.set_context(
            ascend_config={"precision_mode": "allow_mix_precision_bf16"}
        )  # reset ascend precison mode globally

    assert device_num >= sp_size and device_num % sp_size == 0, (
        f"unable to use sequence parallelism, " f"device num: {device_num}, sp size: {sp_size}"
    )
    initialize_sequence_parallel_state(sp_size)
    if get_sequence_parallel_state():
        assert (
            parallel_mode == "data"
        ), f"only support seq parallelism with parallel mode `data`, but got `{parallel_mode}`"

    return rank_id, device_num


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
    parser.add_argument("--mode", default=0, type=int, help="Specify the mode: 0 for graph mode, 1 for pynative mode")
    parser.add_argument("--use_parallel", default=False, type=str2bool, help="use parallel")
    parser.add_argument(
        "--parallel_mode",
        default="data",
        type=str,
        choices=["data", "optim", "semi"],
        help="parallel mode: data, optim",
    )
    parser.add_argument("--enable_dvm", default=False, type=str2bool, help="enable dvm mode")
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
    parser.add_argument("--optim", default="adamw", type=str, help="optimizer")
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
    parser.add_argument("--ema_decay", default=0.9999, type=float, help="EMA decay")

    #################################################################################
    #                                Learning Rate                                  #
    #################################################################################
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="gradient accumulation steps")
    parser.add_argument("--weight_decay", default=1e-2, type=float, help="Weight decay.")
    parser.add_argument("--lr_warmup_steps", default=1000, type=int, help="warmup steps")
    parser.add_argument("--start_learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--end_learning_rate", default=1e-7, type=float, help="The end learning rate for Adam.")
    parser.add_argument("--lr_decay_steps", default=0, type=int, help="lr decay steps.")
    parser.add_argument("--lr_scheduler", default="cosine_decay", type=str, help="scheduler.")
    parser.add_argument(
        "--scale_lr",
        default=False,
        type=str2bool,
        help="Specify whether to scale the learning rate based on the batch size, gradient accumulation steps, and n cards.",
    )

    #################################################################################
    #                           Dataset and DataLoader                              #
    #################################################################################
    parser.add_argument("--batch_size", default=10, type=int, help="batch size")
    parser.add_argument("--dataset_sink_mode", default=False, type=str2bool, help="sink mode")
    parser.add_argument("--sink_size", default=-1, type=int, help="dataset sink size. If -1, sink size = dataset size.")
    parser.add_argument(
        "--epochs",
        default=10,
        type=int,
        help="epochs. If dataset_sink_mode is on, epochs is with respect to dataset sink size. Otherwise, it's w.r.t the dataset size.",
    )
    parser.add_argument("--dataloader_num_workers", default=12, type=int, help="num workers for dataloder")
    parser.add_argument("--max_rowsize", default=64, type=int, help="max rowsize for data loading")

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
        default="O1",
        type=str,
        help="mindspore amp level, O1: most fp32, only layers in whitelist compute in fp16 (dense, conv, etc); \
            O2: most fp16, only layers in blacklist compute in fp32 (batch norm etc)",
    )
    parser.add_argument("--drop_overflow_update", default=True, type=str2bool, help="drop overflow update")
    parser.add_argument("--loss_scaler_type", default="dynamic", type=str, help="dynamic or static")
    parser.add_argument(
        "--global_bf16", action="store_true", help="whether to enable gloabal bf16 for diffusion model training."
    )
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
    parser.add_argument(
        "--enable_flash_attention",
        default=None,
        type=str2bool,
        help="whether to enable flash attention.",
    )
    #################################################################################
    #                                Training Callbacks                            #
    #################################################################################
    parser.add_argument(
        "--resume_from_checkpoint",
        default=False,
        type=str,
        help="It can be a string for path to resume checkpoint, or a bool False for not resuming.(default=False)",
    )
    parser.add_argument("--ckpt_save_interval", default=1, type=int, help="save checkpoint every this epochs or steps")
    parser.add_argument("--ckpt_max_keep", default=10, type=int, help="Maximum number of checkpoints to keep")
    parser.add_argument(
        "--step_mode",
        default=False,
        type=str2bool,
        help="whether save ckpt by steps. If False, save ckpt by epochs.",
    )
    parser.add_argument("--profile", default=False, type=str2bool, help="Profile or not")
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
