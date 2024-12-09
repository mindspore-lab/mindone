import argparse
import logging
import os
import sys

import yaml
from opensora.datasets.aspect import ASPECT_RATIO_MAP, ASPECT_RATIOS
from opensora.utils.model_utils import _check_cfgs_in_parser, str2bool

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.insert(0, mindone_lib_path)
from mindone.utils.config import parse_bool_str
from mindone.utils.misc import to_abspath

logger = logging.getLogger()


def parse_train_args(parser):
    parser.add_argument(
        "--config",
        "-c",
        default="",
        type=str,
        help="path to load a config yaml file that describes the training recipes which will override the default arguments",
    )
    # the following args's defualt value will be overrided if specified in config yaml

    # data
    parser.add_argument("--dataset_name", default="", type=str, help="dataset name")
    parser.add_argument(
        "--csv_path",
        required=True,
        type=str,
        help="path to csv annotation file. columns: video, caption. \
        video indicates the relative path of video file in video_folder. caption - the text caption for video",
    )
    parser.add_argument("--video_column", default="video", type=str, help="name of column for videos saved in csv file")
    parser.add_argument(
        "--caption_column", default="caption", type=str, help="name of column for captions saved in csv file"
    )
    parser.add_argument("--video_folder", required=True, type=str, help="root dir for the video data")
    parser.add_argument("--text_embed_folder", type=str, help="root dir for the text embeding data")
    parser.add_argument("--vae_latent_folder", type=str, help="root dir for the vae latent data")
    parser.add_argument("--filter_data", default=False, type=str2bool, help="Filter non-existing videos.")
    parser.add_argument("--output_path", default="output/", type=str, help="output directory to save training results")
    parser.add_argument(
        "--add_datetime", default=True, type=str2bool, help="If True, add datetime subfolder under output_path"
    )
    # model
    parser.add_argument(
        "--model_version", default="v1", type=str, choices=["v1", "v1.1"], help="OpenSora model version."
    )
    parser.add_argument(
        "--pretrained_model_path",
        default="",
        type=str,
        help="Specify the pretrained model path, either a pretrained " "DiT model or a pretrained Latte model.",
    )
    parser.add_argument("--space_scale", default=0.5, type=float, help="stdit model space scalec")
    parser.add_argument("--time_scale", default=1.0, type=float, help="stdit model time scalec")
    parser.add_argument("--model_max_length", type=int, default=120, help="T5's embedded sequence length.")
    parser.add_argument("--freeze_y_embedder", type=str2bool, default=False, help="Do not train Caption Embedder.")
    parser.add_argument(
        "--patchify",
        type=str,
        default="conv2d",
        choices=["conv3d", "conv2d", "linear"],
        help="patchify_conv3d_replace, conv2d - equivalent conv2d to replace conv3d patchify, linear - equivalent linear layer to replace conv3d patchify  ",
    )
    parser.add_argument(
        "--manual_pad",
        type=str2bool,
        default=False,
        help="whether pad independently for conv2d patchify. \
            If True, pad_mode in conv will be set to 'valid' and padding is done before conv. If False, pad_mode is 'same' in conv. \
            Set True for bucket config training in graph mode. Default: False",
    )
    parser.add_argument(
        "--vae_type",
        type=str,
        choices=["OpenSora-VAE-v1.2", "VideoAutoencoderKL"],
        help="If None, use VideoAutoencoderKL, which is a spatial VAE from SD, for opensora v1.0 and v1.1. \
                If OpenSora-VAE-v1.2, will use 3D VAE (spatial + temporal), typically for opensora v1.2",
    )
    parser.add_argument(
        "--noise_scheduler", type=str, default="ddpm", choices=["ddpm", "rflow"], help="Diffusion noise scheduler."
    )
    parser.add_argument(
        "--sample_method",
        type=str,
        default="uniform",
        choices=["discrete-uniform", "uniform", "logit-normal"],
        help="[RFlow only] Noise sampling method.",
    )
    parser.add_argument(
        "--use_timestep_transform",
        type=str2bool,
        default=True,
        help="[RFlow only] Apply resolution and video length aware timestep sampling.",
    )
    # ms
    parser.add_argument("--debug", type=str2bool, default=False, help="Execute inference in debug mode.")
    parser.add_argument("--device_target", type=str, default="Ascend", help="Ascend or GPU")
    parser.add_argument("--max_device_memory", type=str, default=None, help="e.g. `30GB` for 910a, `59GB` for 910b")
    parser.add_argument("--mode", default=0, type=int, help="Specify the mode: 0 for graph mode, 1 for pynative mode")
    parser.add_argument(
        "--save_graphs",
        type=int,
        default=0,
        help="save IR graphs in different level for debugging, 0 - not save, 1 - save intermediate graphs, 2 - save more information",
    )
    parser.add_argument("--use_parallel", default=False, type=str2bool, help="use parallel")
    parser.add_argument(
        "--parallel_mode", default="data", type=str, choices=["data", "optim"], help="parallel mode: data, optim"
    )
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

    # training hyper-params
    parser.add_argument(
        "--resume",
        default=False,
        type=parse_bool_str,
        help="string: path to resume checkpoint."
        "bool False: not resuming.(default=False)."
        "bool True: ModelArts auto resume training.",
    )
    parser.add_argument("--optim", default="adamw", type=str, help="optimizer")
    parser.add_argument("--zero_stage", default=0, type=int, help="ZeRO stage")
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
                If None, filter list is [layernorm, bias]. Default: None",
    )

    parser.add_argument("--weight_decay", default=1e-6, type=float, help="Weight decay.")
    parser.add_argument("--seed", default=3407, type=int, help="global random seed")
    parser.add_argument("--warmup_steps", default=1000, type=int, help="warmup steps")
    parser.add_argument("--batch_size", default=10, type=int, help="batch size")
    parser.add_argument(
        "--vae_micro_batch_size",
        type=int,
        default=None,
        help="If not None, split batch_size*num_frames into smaller ones for VAE encoding to reduce memory limitation",
    )
    parser.add_argument(
        "--vae_micro_frame_size",
        type=int,
        default=17,
        help="If not None, split batch_size*num_frames into smaller ones for VAE encoding to reduce memory limitation. Used by temporal vae",
    )
    parser.add_argument("--start_learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--end_learning_rate", default=1e-7, type=float, help="The end learning rate for Adam.")
    parser.add_argument("--decay_steps", default=0, type=int, help="lr decay steps.")
    parser.add_argument("--scheduler", default="cosine_decay", type=str, help="scheduler.")
    parser.add_argument("--pre_patchify", default=False, type=str2bool, help="Training with patchified latent.")
    parser.add_argument(
        "--max_image_size", default=512, type=int, help="Max image size for patchified latent training."
    )

    # dataloader params
    parser.add_argument("--dataset_sink_mode", default=False, type=str2bool, help="sink mode")
    parser.add_argument(
        "--video_backend",
        default="cv2",
        type=str,
        choices=["cv2", "decord"],
        help="select video reading backend. if decord, use decord to read video frames, which may lead to memory leak for high-resolution videos. \
                if cv2, use cv2 video capture. only valid for opensora v1.2. Default: cv2",
    )
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
    # parser.add_argument("--cond_stage_trainable", default=False, type=str2bool, help="whether text encoder is trainable")
    parser.add_argument("--use_ema", default=False, type=str2bool, help="whether use EMA")
    parser.add_argument(
        "--ema_decay",
        default=0.9999,
        type=float,
        help="EMA decay ratio, smaller value raises more importance to the current model weight.",
    )
    parser.add_argument("--clip_grad", default=False, type=str2bool, help="whether apply gradient clipping")
    parser.add_argument(
        "--use_recompute",
        default=False,
        type=str2bool,
        help="whether use recompute.",
    )
    parser.add_argument(
        "--num_recompute_blocks",
        default=None,
        type=int,
        help="If None, all stdit blocks will be applied with recompute (gradient checkpointing). If int, the first N blocks will be applied with recompute",
    )
    parser.add_argument(
        "--dtype",
        default="fp16",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        help="what computation data type to use for latte. Default is `fp16`, which corresponds to ms.float16",
    )
    parser.add_argument(
        "--vae_dtype",
        default="fp32",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        help="what compuatation data type to use for vae. Default is `fp32`, which corresponds to ms.float32",
    )
    parser.add_argument(
        "--vae_keep_gn_fp32",
        default=True,
        type=str2bool,
        help="whether keep GroupNorm in fp32.",
    )
    parser.add_argument(
        "--global_bf16",
        default=False,
        type=str2bool,
        help="Experimental. If True, dtype will be overrided, operators will be computered in bf16 if they are supported by CANN",
    )
    parser.add_argument(
        "--vae_param_dtype",
        default="fp32",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        help="what param data type to use for vae. Default is `fp32`, which corresponds to ms.float32",
    )
    parser.add_argument(
        "--amp_level",
        default="O2",
        type=str,
        help="mindspore amp level, O1: most fp32, only layers in whitelist compute in fp16 (dense, conv, etc); \
            O2: most fp16, only layers in blacklist compute in fp32 (batch norm etc)",
    )
    parser.add_argument("--vae_amp_level", default="O2", type=str, help="O2 or O3")
    parser.add_argument("--t5_model_dir", default=None, type=str, help="the T5 cache folder path")
    parser.add_argument(
        "--vae_checkpoint",
        type=str,
        default="models/sd-vae-ft-ema.ckpt",
        help="VAE checkpoint file path which is used to load vae weight.",
    )
    parser.add_argument(
        "--sd_scale_factor", type=float, default=0.18215, help="VAE scale factor of Stable Diffusion model."
    )
    parser.add_argument("--image_size", type=int, nargs="+", help="the image size used to initiate model")
    parser.add_argument("--resolution", type=str, help=f"Supported video resolutions: {list(ASPECT_RATIOS.keys())}")
    parser.add_argument(
        "--aspect_ratio", type=str, help=f"Supported video aspect ratios: {list(ASPECT_RATIO_MAP.keys())}"
    )
    parser.add_argument("--num_frames", default=16, type=int, help="the num of frames used to initiate model")
    parser.add_argument("--frame_stride", default=1, type=int, help="frame sampling stride")
    parser.add_argument("--mask_ratios", type=dict, help="Masking ratios")
    parser.add_argument("--bucket_config", type=dict, help="Multi-resolution bucketing configuration")
    parser.add_argument(
        "--num_parallel_workers",
        default=12,
        type=int,
        help="The number of workers used for data transformations. Default is 12.",
    )
    parser.add_argument(
        "--num_workers_dataset",
        default=4,
        type=int,
        help="The number of workers used for reading data from the dataset. Default is 4.",
    )
    parser.add_argument(
        "--num_workers_batch",
        default=2,
        type=int,
        help="The number of workers used for batch aggregation. Default is 2.",
    )
    parser.add_argument(
        "--prefetch_size", default=16, type=int, help="The number of samples to prefetch (per device). Default is 16."
    )
    parser.add_argument(
        "--data_multiprocessing",
        default=False,
        type=str2bool,
        help="If True, use multiprocessing for data processing. Default: multithreading.",
    )
    parser.add_argument("--max_rowsize", default=64, type=int, help="max rowsize for data loading")
    parser.add_argument(
        "--disable_flip",
        default=True,
        type=str2bool,
        help="disable random flip video (to avoid motion direction and text mismatch)",
    )
    parser.add_argument(
        "--enable_flash_attention",
        default=None,
        type=str2bool,
        help="whether to enable flash attention.",
    )
    parser.add_argument(
        "--enable_sequence_parallelism",
        default=False,
        type=str2bool,
        help="whether to enable sequence parallelism. Default is False",
    )
    parser.add_argument(
        "--sequence_parallel_shards",
        default=1,
        type=int,
        help="The number of shards in sequence parallel. Default is 1.",
    )
    parser.add_argument("--dsp", default=False, type=str2bool, help="Use DSP instead of SP in sequence parallel.")
    parser.add_argument("--drop_overflow_update", default=True, type=str2bool, help="drop overflow update")
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
        default=None,
        type=str2bool,
        help="whether save ckpt by steps. If False, save ckpt by epochs. If None, will be determined by train_steps and dataset_sink_mode automatically",
    )
    parser.add_argument(
        "--custom_train", default=False, type=str2bool, help="Use custom train process instead of model.train"
    )
    parser.add_argument("--profile", default=False, type=str2bool, help="Profile or not")
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

    # ---------- Validation ----------
    parser.add_argument(
        "--validate", type=str2bool, default=False, help="Whether to perform validation during training."
    )
    parser.add_argument("--val_interval", default=1, type=int, help="Validation frequency in epochs")
    parser.add_argument(
        "--num_eval_timesteps",
        type=int,
        default=10,
        help="The number of timesteps to evaluate on (sampled equidistantly).",
    )
    parser.add_argument("--val_csv_path", type=str, help="the validation csv path")
    parser.add_argument("--val_video_folder", type=str, help="the validation video folder path")
    parser.add_argument("--val_text_embed_folder", type=str, help="the validation text embedding folder path")
    parser.add_argument("--val_vae_latent_folder", type=str, help="the validation vae latent folder path")
    parser.add_argument("--val_batch_size", default=1, type=int, help="the validation vae latent folder path")
    parser.add_argument("--val_bucket_config", type=dict, help="Multi-resolution bucketing configuration.")
    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    parser = parse_train_args(parser)

    __dir__ = os.path.dirname(os.path.abspath(__file__))
    abs_path = os.path.abspath(os.path.join(__dir__, ".."))
    default_args = parser.parse_args()
    if default_args.config:
        default_args.config = to_abspath(abs_path, default_args.config)
        with open(default_args.config, "r") as f:
            cfg = yaml.safe_load(f)
            _check_cfgs_in_parser(cfg, parser)
            parser.set_defaults(**cfg)
    args = parser.parse_args()
    # convert to absolute path, necessary for modelarts
    args.csv_path = to_abspath(abs_path, args.csv_path)
    args.video_folder = to_abspath(abs_path, args.video_folder)
    args.text_embed_folder = to_abspath(abs_path, args.text_embed_folder)
    args.vae_latent_folder = to_abspath(abs_path, args.vae_latent_folder)
    args.output_path = to_abspath(abs_path, args.output_path)
    args.pretrained_model_path = to_abspath(abs_path, args.pretrained_model_path)
    args.t5_model_dir = to_abspath(abs_path, args.t5_model_dir)
    args.vae_checkpoint = to_abspath(abs_path, args.vae_checkpoint)
    print(args)

    return args
