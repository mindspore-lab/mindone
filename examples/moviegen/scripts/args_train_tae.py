import logging
import os
import sys

from jsonargparse import ActionConfigFile, ArgumentParser

# TODO: remove in future when mindone is ready for install
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.append(mindone_lib_path)
sys.path.append(os.path.join(__dir__, ".."))

from mg.dataset.tae_dataset import BatchTransform, VideoDataset
from mg.models.tae import TemporalAutoencoder

from mindone.data import create_dataloader
from mindone.utils import init_train_env
from mindone.utils.misc import to_abspath

logger = logging.getLogger()


def parse_train_args():
    parser = ArgumentParser(description="Temporal Autoencoder training script.")
    parser.add_argument(
        "-c",
        action=ActionConfigFile,
        help="Path to load a config yaml file that describes the setting which will override the default arguments.",
    )
    parser.add_function_arguments(
        init_train_env, skip={"ascend_config", "num_workers", "json_data_path", "enable_modelarts"}
    )
    parser.add_class_arguments(TemporalAutoencoder, instantiate=False)
    parser.add_class_arguments(VideoDataset, skip={"output_columns"}, instantiate=False)
    parser.add_class_arguments(BatchTransform, instantiate=False)
    parser.add_function_arguments(
        create_dataloader,
        skip={"dataset", "transforms", "batch_transforms", "device_num", "rank_id", "debug", "enable_modelarts"},
    )
    parser.add_argument("--output_path", default="output/", type=str, help="output directory to save training results")
    parser.add_argument(
        "--add_datetime", default=True, type=str, help="If True, add datetime subfolder under output_path"
    )
    # model
    parser.add_argument("--perceptual_loss_weight", default=0.1, type=float, help="perceptual (lpips) loss weight")
    parser.add_argument("--kl_loss_weight", default=1.0e-6, type=float, help="KL loss weight")
    parser.add_argument(
        "--use_outlier_penalty_loss",
        default=False,
        type=bool,
        help="use outlier penalty loss",
    )
    # training hyper-params
    parser.add_argument(
        "--resume",
        default=False,
        type=str,
        help="It can be a string for path to resume checkpoint, or a bool False for not resuming.(default=False)",
    )
    parser.add_argument("--optim", default="adamw_re", type=str, help="optimizer")
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
        default=None,
        help="Grouping strategy for weight decay. If `norm_and_bias`, weight decay filter list is [beta, gamma, bias]. \
                If None, filter list is [layernorm, bias], Default: None",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay.")
    parser.add_argument("--warmup_steps", default=1000, type=int, help="warmup steps")
    parser.add_argument("--start_learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--end_learning_rate", default=1e-7, type=float, help="The end learning rate for Adam.")
    parser.add_argument(
        "--scale_lr", default=False, type=bool, help="scale base-lr by ngpu * batch_size * n_accumulate"
    )
    parser.add_argument("--decay_steps", default=0, type=int, help="lr decay steps.")
    parser.add_argument("--scheduler", default="cosine_decay", type=str, help="scheduler.")
    parser.add_argument("--pre_patchify", default=False, type=bool, help="Training with patchified latent.")

    # dataloader params
    parser.add_argument("--dataset_sink_mode", default=False, type=bool, help="sink mode")
    parser.add_argument("--sink_size", default=-1, type=int, help="dataset sink size. If -1, sink size = dataset size.")
    parser.add_argument(
        "--epochs",
        default=10,
        type=int,
        help="epochs. If dataset_sink_mode is on, epochs is with respect to dataset sink size. Otherwise, it's w.r.t the dataset size.",
    )
    parser.add_argument(
        "--train_steps", default=-1, type=int, help="If not -1, limit the number of training steps to the set value"
    )
    parser.add_argument("--init_loss_scale", default=65536, type=float, help="loss scale")
    parser.add_argument("--loss_scale_factor", default=2, type=float, help="loss scale factor")
    parser.add_argument("--scale_window", default=2000, type=float, help="scale window")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="gradient accumulation steps")
    # parser.add_argument("--cond_stage_trainable", default=False, type=bool, help="whether text encoder is trainable")
    parser.add_argument("--use_ema", default=False, type=bool, help="whether use EMA")
    parser.add_argument("--ema_decay", default=0.9999, type=float, help="ema decay ratio")
    parser.add_argument("--clip_grad", default=False, type=bool, help="whether apply gradient clipping")
    parser.add_argument(
        "--vae_keep_gn_fp32",
        default=True,
        type=bool,
        help="whether keep GroupNorm in fp32.",
    )
    parser.add_argument(
        "--vae_keep_updown_fp32",
        default=True,
        type=bool,
        help="whether keep spatial/temporal upsample and downsample in fp32.",
    )
    parser.add_argument(
        "--enable_flash_attention",
        default=None,
        type=bool,
        help="whether to enable flash attention.",
    )
    parser.add_argument("--drop_overflow_update", default=True, type=bool, help="drop overflow update")
    parser.add_argument("--loss_scaler_type", default="dynamic", type=str, help="dynamic or static")
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="max gradient norm for clipping, effective when `clip_grad` enabled.",
    )
    parser.add_argument("--ckpt_save_interval", default=1, type=int, help="save checkpoint every this epochs")
    parser.add_argument(
        "--ckpt_save_steps",
        default=-1,
        type=int,
        help="save checkpoint every this steps. If -1, use ckpt_save_interval will be used.",
    )
    parser.add_argument("--ckpt_max_keep", default=10, type=int, help="Maximum number of checkpoints to keep")
    parser.add_argument(
        "--step_mode",
        default=False,
        type=bool,
        help="whether save ckpt by steps. If False, save ckpt by epochs.",
    )
    parser.add_argument("--profile", default=False, type=bool, help="Profile or not")
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
        help="log interval in the unit of data sink size.. E.g. if data sink size = 10, log_inteval=2, log every 20 steps",
    )
    return parser


def parse_args():
    parser = parse_train_args()
    args = parser.parse_args()

    __dir__ = os.path.dirname(os.path.abspath(__file__))
    abs_path = os.path.abspath(os.path.join(__dir__, ".."))

    # convert to absolute path, necessary for modelarts
    args.csv_path = to_abspath(abs_path, args.csv_path)
    args.video_folder = to_abspath(abs_path, args.video_folder)
    args.output_path = to_abspath(abs_path, args.output_path)
    args.pretrained_model_path = to_abspath(abs_path, args.pretrained_model_path)
    args.vae_checkpoint = to_abspath(abs_path, args.vae_checkpoint)
    print(args)

    return args
