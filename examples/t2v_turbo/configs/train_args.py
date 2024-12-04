import argparse
import os

from mindone.utils.config import str2bool


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    # ----------Model Checkpoint Loading Arguments----------
    parser.add_argument(
        "--pretrained_model_cfg",
        type=str,
        default="configs/inference_t2v_512_v2.0.yaml",
        help="Pretrained Model Config.",
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="PATH_TO_VC2_model.ckpt",
        help="Path to the pretrained model.",
    )
    parser.add_argument(
        "--pretrained_enc_path",
        type=str,
        default="PATH_TO_encoder_model.ckpt",
        help="Path to the pretrained t2v encoder model.",
    )
    parser.add_argument("--pretrained_lora_path", type=str, default=None, help="Path to pretrained lora weights.")
    # ----------MS environment args----------
    parser.add_argument("--device_target", type=str, default="Ascend", help="Ascend or GPU")
    parser.add_argument(
        "--mode",
        type=int,
        default=0,
        help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)",
    )
    parser.add_argument("--use_parallel", default=False, type=str2bool, help="use parallel")
    parser.add_argument(
        "--parallel_mode",
        default="data",
        type=str,
        choices=["data", "optim", "pipeline"],
        help="parallel mode: data, optim",
    )
    parser.add_argument(
        "--optimizer_weight_shard_size",
        type=int,
        default=8,
        help="Set the size of the communication domain split by the optimizer weight. ",
    )
    parser.add_argument("--debug", type=str2bool, default=False, help="Execute inference in debug mode.")
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
    parser.add_argument("--max_device_memory", type=str, default=None, help="e.g. `30GB` for 910a, `59GB` for 910b")
    # ----------Training Arguments----------
    # ----General Training Arguments----
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/t2v-turbo-vc2",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=453645634, help="A seed for reproducible training.")
    # ----Logging----
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument("--log_interval", type=int, default=1, help="log interval")
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    # ----Checkpointing----
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=5,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="latest",
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument("--ckpt_save_interval", default=1, type=int, help="save checkpoint every this epochs or steps")
    # ----Image Processing----
    parser.add_argument(
        "--train_shards_path_or_url",
        type=str,
        default="PATH_TO_WEBVID_DATA_DIR",
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument("--data_path", default="dataset", type=str, help="path to video root folder")
    parser.add_argument(
        "--csv_path",
        default=None,
        type=str,
        help="path to csv annotation file. If None, video_caption.csv is expected to live under `data_path`",
    )
    # ----Dataloader----
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    # ----Batch Size and Training Steps----
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--reward_batch_size",
        type=int,
        default=5,
        help="Batch size (per device) for optimizing the text-image RM.",
    )
    parser.add_argument(
        "--video_rm_batch_size",
        type=int,
        default=8,
        help="Num frames for inputing to the text-video RM.",
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=16,
        help="Number of frames to sample from a video.",
    )

    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=10000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=400000,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    # ----Learning Rate----
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--end_learning_rate", default=1e-7, type=float, help="The end learning rate for the optimizer."
    )
    parser.add_argument("--decay_steps", default=0, type=int, help="lr decay steps.")
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument("--scheduler", default="constant", type=str, help="scheduler.")
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--drop_overflow_update", default=True, type=str2bool, help="drop overflow update")
    parser.add_argument("--loss_scaler_type", default="dynamic", type=str, help="dynamic or static")
    parser.add_argument("--init_loss_scale", default=65536, type=float, help="loss scale")
    parser.add_argument("--loss_scale_factor", default=2, type=float, help="loss scale factor")
    parser.add_argument("--scale_window", default=2000, type=float, help="scale window")
    parser.add_argument("--use_recompute", default=False, type=str2bool, help="if use recompute")
    # ----Optimizer (Adam)----
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument("--adam_weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--clip_grad", default=False, type=str2bool, help="whether apply gradient clipping")
    # ----Diffusion Training Arguments----
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    # ---- Latent Consistency Distillation (LCD) Specific Arguments ----
    parser.add_argument(
        "--w_min",
        type=float,
        default=5.0,
        required=False,
        help=(
            "The minimum guidance scale value for guidance scale sampling. Note that we are using the Imagen CFG"
            " formulation rather than the LCM formulation, which means all guidance scales have 1 added to them as"
            " compared to the original paper."
        ),
    )
    parser.add_argument(
        "--w_max",
        type=float,
        default=15.0,
        required=False,
        help=(
            "The maximum guidance scale value for guidance scale sampling. Note that we are using the Imagen CFG"
            " formulation rather than the LCM formulation, which means all guidance scales have 1 added to them as"
            " compared to the original paper."
        ),
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help=("Eta for solving the DDIM step."),
    )
    parser.add_argument(
        "--no_scale_pred_x0",
        action="store_true",
        default=False,
        help=("Whether to scale the pred_x0 in DDIM step."),
    )
    parser.add_argument(
        "--num_ddim_timesteps",
        type=int,
        default=50,
        help="Num timesteps for DDIM sampling",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help="1000 (Num Train timesteps) // 50 (Num timesteps for DDIM sampling)",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="huber",
        choices=["l2", "huber", "none"],
        help="The type of loss to use for the LCD loss.",
    )
    parser.add_argument(
        "--huber_c",
        type=float,
        default=0.001,
        help="The huber loss parameter. Only used if `--loss_type=huber`.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="The rank of the LoRA projection matrix.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="The dropout probability for the dropout layer added before applying the LoRA to each layer input.",
    )
    parser.add_argument(
        "--unet_time_cond_proj_dim",
        type=int,
        default=256,
        help=(
            "The dimension of the guidance scale embedding in the U-Net, which will be used if the teacher U-Net"
            " does not have `time_cond_proj_dim` set."
        ),
    )
    parser.add_argument(
        "--vae_encode_batch_size",
        type=int,
        default=8,
        required=False,
        help=(
            "The batch size used when encoding images to latents using the VAE."
            " Encoding the whole batch at once may run into OOM issues."
        ),
    )
    parser.add_argument(
        "--vae_decode_batch_size",
        type=int,
        default=16,
        required=False,
        help=(
            "The batch size used when decoding images to latents using the VAE."
            " Decoding the whole batch at once may run into OOM issues."
        ),
    )

    parser.add_argument(
        "--timestep_scaling_factor",
        type=float,
        default=10.0,
        help=(
            "The multiplicative timestep scaling factor used when calculating the boundary scalings for LCM. The"
            " higher the scaling is, the lower the approximation error, but the default value of 10.0 should typically"
            " suffice."
        ),
    )
    # ----Exponential Moving Average (EMA)----
    parser.add_argument("--use_ema", default=False, type=str2bool, help="whether use EMA")
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.95,
        required=False,
        help="The exponential moving average (EMA) rate or decay factor.",
    )
    # ----Mixed Precision----
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--global_bf16",
        default=False,
        type=str2bool,
        help="Experimental. If True, dtype will be overrided, operators will be computered in bf16 if they are supported by CANN",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--cast_teacher_unet",
        action="store_true",
        help="Whether to cast the teacher U-Net to the precision specified by `--mixed_precision`.",
    )
    # ----Training Optimizations----
    # ----Distributed Training----
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    # ----------Validation Arguments----------
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=500,
        help="Run validation every X steps.",
    )
    # ----------Accelerate Arguments----------
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="t2v-turbo",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--reward_fn_name",
        type=str,
        default="hpsv2",
        help="Image Reward function name",
    )
    parser.add_argument(
        "--reward_scale",
        type=float,
        default=1.0,
        help="The scale of the reward loss",
    )
    parser.add_argument(
        "--image_rm_ckpt_dir",
        type=str,
        default="PATH/TO/HPS_v2_compressed.ckpt",
        help="Image-Text reward model path",
    )
    parser.add_argument(
        "--video_rm_name",
        type=str,
        default="vi_clip2",
        help="Video Reward function name",
    )
    parser.add_argument(
        "--video_rm_ckpt_dir",
        type=str,
        # default="PATH/TO/ViClip-InternVid-10M-FLT.pth",
        default="PATH/TO/InternVideo2-stage2_1b-224p-f4.pt",
        help="Video-Text reward model path",
    )
    parser.add_argument(
        "--video_reward_scale",
        type=float,
        default=1.0,
        help="The scale of the viclip reward loss",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.video_rm_name == "vi_clip":
        assert args.video_rm_batch_size == 8
    elif args.video_rm_name == "vi_clip2":
        assert args.video_rm_batch_size in [4, 8]
    else:
        raise ValueError(f"Unsupported viclip reward function: {args.video_rm_name}")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    return args
