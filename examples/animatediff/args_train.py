import argparse
import logging
import os

import yaml

from mindone.utils.config import str2bool

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
    parser.add_argument("--model_config", default="configs/v1-train-chinese.yaml", type=str, help="model config path")
    parser.add_argument("--data_path", default="dataset", type=str, help="path to video root folder")
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
    parser.add_argument("--device_target", type=str, default="Ascend", help="Ascend or GPU")
    parser.add_argument("--max_device_memory", type=str, default=None, help="e.g. `30GB` for 910a, `59GB` for 910b")
    parser.add_argument("--mode", default=0, type=int, help="Specify the mode: 0 for graph mode, 1 for pynative mode")
    parser.add_argument("--use_parallel", default=False, type=str2bool, help="use parallel")
    parser.add_argument(
        "--replace_small_images",
        default=True,
        type=str2bool,
        help="replace the small-size images with other training samples",
    )
    # modelarts
    parser.add_argument("--enable_modelarts", default=False, type=str2bool, help="run codes in ModelArts platform")
    parser.add_argument("--num_workers", default=1, type=int, help="the number of modelarts workers")
    parser.add_argument(
        "--json_data_path",
        default="mindone/examples/stable_diffusion_v2/ldm/data/num_samples_64_part.json",
        type=str,
        help="the path of num_samples.json containing a dictionary with 64 parts. "
        "Each part is a large dictionary containing counts of samples of 533 tar packages.",
    )
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
    parser.add_argument("--train_batch_size", default=10, type=int, help="batch size")
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
        "--save_mm_only",
        default=False,
        type=str2bool,
        help="If True, save motion module params only to reduce checkpoint size. Otherwise, save the whole ldm model (vae, clip, unet, mm) in one checkpoint",
    )
    # video
    parser.add_argument(
        "--image_finetune",
        default=True,
        type=str2bool,
        help="True for image finetune. False for motion module training.",
    )
    parser.add_argument(
        "--force_motion_module_amp_O2",
        default=False,
        type=str2bool,
        help="if True, set mixed precision O2 for MM. Otherwise, use manually defined precision according to use_fp16 flag",
    )
    parser.add_argument(
        "--vae_fp16",
        default=None,
        type=bool,
        help="whether use fp16 precision in vae. If None, it will be set by the value in stable diffusion config yaml",
    )
    parser.add_argument("--image_size", default=256, type=int, help="image size")
    parser.add_argument("--num_frames", default=16, type=int, help="num frames")
    parser.add_argument("--frame_stride", default=4, type=int, help="frame sampling stride")
    parser.add_argument(
        "--random_drop_text", default=True, type=str2bool, help="set caption to empty string randomly if enabled"
    )
    parser.add_argument("--random_drop_text_ratio", default=0.1, type=float, help="drop ratio")
    parser.add_argument(
        "--disable_flip",
        default=True,
        type=str2bool,
        help="disable random flip video (to avoid motion direction and text mismatch)",
    )
    parser.add_argument("--num_parallel_workers", default=12, type=int, help="num workers for data loading")
    parser.add_argument(
        "--motion_module_path", default="", type=str, help="path to pretrained motion mdule. Load it if not empty"
    )
    parser.add_argument(
        "--train_data_type",
        default="video_file",
        type=str,
        choices=["video_file", "npz", "mindrecord"],
        help="type of data for training",
    )
    parser.add_argument("--motion_lora_finetune", default=False, type=str2bool, help="True if finetune motion lora.")
    parser.add_argument("--motion_lora_rank", default=64, type=int, help="motion lora rank.")
    parser.add_argument(
        "--motion_lora_alpha", default=1.0, type=int, help="alpha: the strength of LoRA, typically in range [0, 1]"
    )

    # For embedding cache
    parser.add_argument("--video_column", default="video", type=str, help="name of column for videos saved in csv file")
    parser.add_argument(
        "--caption_column", default="caption", type=str, help="name of column for captions saved in csv file"
    )
    parser.add_argument(
        "--save_data_type",
        default="float32",
        type=str,
        choices=["float16", "float32"],
        help="data type when saving embedding cache",
    )

    parser.add_argument("--cache_folder", default="", type=str, help="directory to save embedding cache")

    abs_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ""))
    default_args = parser.parse_args()
    if default_args.config:
        default_args.config = os.path.join(abs_path, default_args.config)
        with open(default_args.config, "r") as f:
            cfg = yaml.safe_load(f)
            _check_cfgs_in_parser(cfg, parser)
            parser.set_defaults(**cfg)
    args = parser.parse_args()
    args.model_config = os.path.join(abs_path, args.model_config)

    print(args)

    return args
