import argparse
from abc import abstractmethod
from logging import getLogger

from omegaconf import OmegaConf
from openlrm.runners.abstract import Runner
from openlrm.utils import str2bool

from mindspore import nn
from mindspore.experimental.optim.lr_scheduler import LRScheduler

logger = getLogger(__name__)


def parse_args(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser = parse_train_args(parser)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="resume from checkpoint with path",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train-sample.yaml",
        help="path to base configs",
    )
    parser.add_argument(
        "--log_interval",
        default=1,
        type=int,
        help="log interval in the unit of data sink size.. E.g. if data sink size = 10, log_inteval=2, log every 20 steps",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="When debugging, set it true. Dumping files will overlap to avoid trashing your storage.",
    )
    # args = parser.parse_args() # unrecognize runner=train.lrm
    args, unknown = parser.parse_known_args()
    return args


def parse_train_args(parser):
    parser.add_argument("--mode", default=1, type=int, help="Specify the mode: 0 for graph mode, 1 for pynative mode")
    parser.add_argument("--use_parallel", default=False, help="use parallel")
    parser.add_argument(
        "--parallel_mode", default="data", type=str, choices=["data", "optim"], help="parallel mode: data, optim"
    )
    parser.add_argument("--device_target", type=str, default="Ascend", help="Ascend or GPU")
    parser.add_argument("--max_device_memory", type=str, default=None, help="e.g. `30GB` for 910a, `59GB` for 910b")
    parser.add_argument(
        "--dtype",
        default="fp32",  # if amp level O0/1, must pass fp32; if amp level O2, pass bf16
        type=str,
        choices=["bf16", "fp16", "fp32"],
        help="what computation data type to use. Default is `fp32`, which corresponds to ms.float32",
    )
    parser.add_argument(
        "--amp_level",
        default="O0",
        type=str,
        help="mindspore amp level, O1: most fp32, only layers in whitelist compute in fp16 (dense, conv, etc); \
                        O2: most fp16, only layers in blacklist compute in fp32 (batch norm etc)",
    )
    parser.add_argument("--ckpt_save_interval", default=1000, type=int, help="save checkpoint every this epochs")
    parser.add_argument(
        "--profile",
        default=False,  # deactivate as profiler says NOT supporting PyNative
        type=str2bool,
        help="Profile or not",
    )
    parser.add_argument(
        "--loss_scaler_type", default=None, type=str, help="dynamic or static"  # loss scale only used in amp O1/O2
    )
    parser.add_argument("--init_loss_scale", default=16, type=float, help="loss scale")
    parser.add_argument("--loss_scale_factor", default=2, type=float, help="loss scale factor")
    parser.add_argument("--scale_window", default=10000, type=float, help="scale window")
    parser.add_argument("--ckpt_max_keep", default=10, type=int, help="Maximum number of checkpoints to keep")
    parser.add_argument("--output_path", default="outputs/", type=str, help="output directory to save training results")
    parser.add_argument(
        "--log_level",
        type=str,
        default="logging.INFO",
        help="log level, options: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR",
    )
    parser.add_argument("--num_parallel_workers", default=1, type=int, help="num workers for data loading")
    parser.add_argument(
        "--data_multiprocessing",
        default=False,
        help="If True, use multiprocessing for data processing. Default: multithreading.",
    )
    parser.add_argument("--max_rowsize", default=64, type=int, help="max rowsize for data loading")
    parser.add_argument(
        "--train_steps", default=-1, type=int, help="If not -1, limit the number of training steps to the set value"
    )
    parser.add_argument(
        "--epochs",
        default=100000,
        type=int,
        help="epochs. If dataset_sink_mode is on, epochs is with respect to dataset sink size. Otherwise, it's w.r.t the dataset size.",
    )
    parser.add_argument(
        "--ckpt_save_steps",
        default=-1,
        type=int,
        help="save checkpoint every this steps. If -1, use ckpt_save_interval will be used.",
    )
    parser.add_argument("--step_mode", default=False, help="whether save ckpt by steps. If False, save ckpt by epochs.")
    # optimizer param
    parser.add_argument("--use_ema", default=False, help="whether use EMA")
    parser.add_argument("--drop_overflow_update", default=True, type=str2bool, help="drop overflow update")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="gradient accumulation steps")
    parser.add_argument("--clip_grad", default=True, type=str2bool, help="whether apply gradient clipping")
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="max gradient norm for clipping, effective when `clip_grad` enabled.",
    )
    parser.add_argument(
        "--use_recompute",
        default=False,
        type=str2bool,
        help="whether use recompute.",
    )
    parser.add_argument("--start_learning_rate", default=4e-4, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--end_learning_rate", default=4e-5, type=float, help="The end learning rate for Adam.")
    parser.add_argument("--decay_steps", default=5e5, type=int, help="lr decay steps.")  # 5 data * epochs
    parser.add_argument("--scheduler", default="cosine_annealing_warm_restarts_lr", type=str, help="scheduler.")
    parser.add_argument("--optim", default="adamw", type=str, help="optimizer")
    parser.add_argument(
        "--betas",
        type=float,
        nargs="+",
        default=[0.9, 0.95],
        help="Specify the [beta1, beta2] parameter for the AdamW optimizer.",
    )
    parser.add_argument(
        "--optim_eps", type=float, default=1e-8, help="Specify the eps parameter for the AdamW optimizer."
    )
    parser.add_argument(
        "--group_strategy",
        type=str,
        default="not_grouping",
        help="Grouping strategy for weight decay. If `norm_and_bias`, weight decay filter list is [beta, gamma, bias]. \
                                If None, filter list is [layernorm, bias]. Default: norm_and_bias",
    )
    parser.add_argument("--weight_decay", default=0.05, type=float, help="Weight decay.")
    parser.add_argument("--seed", default=42, type=int, help="data path")
    parser.add_argument("--warmup_steps", default=0, type=int, help="warmup steps")
    # dataloader param
    parser.add_argument("--dataset_sink_mode", default=False, help="sink mode")
    parser.add_argument("--sink_size", default=-1, type=int, help="dataset sink size. If -1, sink size = dataset size.")

    return parser


class Trainer(Runner):
    def __init__(self):
        super().__init__()
        # read configs
        self.args = parse_args()
        self.cfg = OmegaConf.load(self.args.config)

        # attributes with defaults
        self.model: nn.Cell = None
        self.optimizer: nn.optim.Optimizer = None
        self.scheduler: LRScheduler = None
        self.train_loader = None
        self.val_loader = None
        self.N_max_global_steps: int = None
        self.N_global_steps_per_epoch: int = None
        self.global_step: int = 0
        self.current_epoch: int = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @abstractmethod
    def _build_model(self):
        pass

    @abstractmethod
    def _build_utils(self):
        pass

    @abstractmethod
    def _build_dataloader(self):
        pass

    @abstractmethod
    def train(self):
        pass

    def run(self):
        self.train(self.args, self.cfg)
