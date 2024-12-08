import argparse
import os

from mindone.utils.config import str2bool


def add_model_checkpoint_loading_arguments(parser):
    """Add model checkpoint loading arguments."""
    model_group = parser.add_argument_group("Model Checkpoint Loading")
    model_group.add_argument(
        "--pretrained_model_cfg",
        type=str,
        default="configs/inference_t2v_512_v2.0.yaml",
        help="Pretrained Model Config.",
    )
    model_group.add_argument(
        "--pretrained_model_path",
        type=str,
        default="PATH_TO_VC2_model.ckpt",
        help="Path to the pretrained model.",
    )
    model_group.add_argument(
        "--pretrained_enc_path",
        type=str,
        default="PATH_TO_encoder_model.ckpt",
        help="Path to the pretrained t2v encoder model.",
    )
    model_group.add_argument("--pretrained_lora_path", type=str, default=None, help="Path to pretrained lora weights.")


def add_ms_environment_arguments(parser):
    """Add MindSpore environment arguments."""
    ms_env_group = parser.add_argument_group("MS Environment")
    ms_env_group.add_argument("--device_target", type=str, default="Ascend", help="Ascend or GPU")
    ms_env_group.add_argument(
        "--mode", type=int, default=0, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)"
    )
    ms_env_group.add_argument("--use_parallel", type=str2bool, default=False, help="Use parallel processing.")
    ms_env_group.add_argument("--debug", type=str2bool, default=False, help="Execute inference in debug mode.")
    ms_env_group.add_argument(
        "--jit_level", type=str, default="O0", choices=["O0", "O1", "O2"], help="Compilation optimization level."
    )
    ms_env_group.add_argument(
        "--max_device_memory", type=str, default=None, help="e.g. `30GB` for 910a, `59GB` for 910b"
    )


def add_training_arguments(parser):
    """Add training-related arguments."""
    training_group = parser.add_argument_group("Training Arguments")
    training_group.add_argument(
        "--output_dir",
        type=str,
        default="output/t2v-turbo-vc2",
        help="Output directory for model predictions and checkpoints.",
    )
    training_group.add_argument("--seed", type=int, default=453645634, help="Seed for reproducible training.")
    training_group.add_argument("--logging_dir", type=str, default="logs", help="TensorBoard log directory.")
    training_group.add_argument("--log_interval", type=int, default=1, help="Log interval.")

    # Recompute
    training_group.add_argument("--use_recompute", default=False, type=str2bool, help="if use recompute")

    # Checkpointing
    training_group.add_argument(
        "--checkpoints_total_limit", type=int, default=5, help="Max number of checkpoints to store."
    )
    training_group.add_argument(
        "--ckpt_save_interval", type=int, default=1, help="Save checkpoint every this many epochs or steps."
    )

    # Data paths
    training_group.add_argument("--data_path", type=str, default="dataset", help="Path to video root folder.")
    training_group.add_argument("--csv_path", type=str, default=None, help="Path to CSV annotation file.")

    # Dataloader settings
    training_group.add_argument(
        "--dataloader_num_workers", type=int, default=8, help="Number of subprocesses for data loading."
    )
    training_group.add_argument("--train_batch_size", type=int, default=16, help="Batch size for training dataloader.")
    training_group.add_argument(
        "--reward_batch_size", type=int, default=5, help="Batch size for optimizing the text-image RM."
    )
    training_group.add_argument(
        "--video_rm_batch_size", type=int, default=8, help="Num frames for inputting to the text-video RM."
    )
    training_group.add_argument("--n_frames", type=int, default=16, help="Number of frames to sample from a video.")
    training_group.add_argument("--num_train_epochs", type=int, default=100, help="Number of training epochs.")
    training_group.add_argument("--max_train_steps", type=int, default=10000, help="Total number of training steps.")
    training_group.add_argument(
        "--max_train_samples", type=int, default=400000, help="Limit on training examples for debugging."
    )

    # Model
    training_group.add_argument(
        "--unet_time_cond_proj_dim",
        type=int,
        default=256,
        help=(
            "The dimension of the guidance scale embedding in the U-Net, which will be used if the teacher U-Net"
            " does not have `time_cond_proj_dim` set."
        ),
    )


def add_learning_rate_arguments(parser):
    """Add learning rate arguments."""
    lr_group = parser.add_argument_group("Learning Rate")
    lr_group.add_argument("--learning_rate", type=float, default=1e-5, help="Initial learning rate.")
    lr_group.add_argument("--end_learning_rate", type=float, default=1e-7, help="End learning rate for the optimizer.")
    lr_group.add_argument("--decay_steps", type=int, default=0, help="Learning rate decay steps.")
    lr_group.add_argument(
        "--scale_lr", action="store_true", default=False, help="Scale learning rate by various factors."
    )
    lr_group.add_argument("--scheduler", type=str, default="constant", help="Learning rate scheduler.")
    lr_group.add_argument("--lr_warmup_steps", type=int, default=500, help="Warmup steps for the lr scheduler.")
    lr_group.add_argument(
        "--gradient_accumulation_steps", type=int, default=1, help="Steps to accumulate before updating."
    )
    lr_group.add_argument("--drop_overflow_update", type=str2bool, default=True, help="Drop overflow update.")
    lr_group.add_argument("--loss_scaler_type", default="dynamic", type=str, help="dynamic or static")
    lr_group.add_argument("--init_loss_scale", default=65536, type=float, help="loss scale")
    lr_group.add_argument("--loss_scale_factor", default=2, type=float, help="loss scale factor")
    lr_group.add_argument("--scale_window", default=2000, type=float, help="scale window")


def add_optimizer_arguments(parser):
    """Add optimizer arguments."""
    opt_group = parser.add_argument_group("Optimizer (Adam)")
    opt_group.add_argument("--adam_beta1", type=float, default=0.9, help="Beta1 parameter for Adam optimizer.")
    opt_group.add_argument("--adam_beta2", type=float, default=0.999, help="Beta2 parameter for Adam optimizer.")
    opt_group.add_argument("--adam_weight_decay", type=float, default=0.0, help="Weight decay for Adam.")
    opt_group.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for Adam optimizer.")
    opt_group.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm.")
    opt_group.add_argument("--clip_grad", type=str2bool, default=False, help="Apply gradient clipping.")


def add_lcd_arguments(parser):
    """Add Latent Consistency Distillation specific arguments."""
    lcd_group = parser.add_argument_group("Latent Consistency Distillation")
    lcd_group.add_argument("--w_min", type=float, default=5.0, help="Minimum guidance scale value.")
    lcd_group.add_argument("--w_max", type=float, default=15.0, help="Maximum guidance scale value.")
    lcd_group.add_argument("--ddim_eta", type=float, default=0.0, help="Eta for DDIM step.")
    lcd_group.add_argument("--no_scale_pred_x0", action="store_true", default=False, help="Scale pred_x0 in DDIM step.")
    lcd_group.add_argument("--num_ddim_timesteps", type=int, default=50, help="Num timesteps for DDIM sampling.")
    lcd_group.add_argument("--topk", type=int, default=20, help="Top K for training timesteps.")
    lcd_group.add_argument(
        "--loss_type", type=str, default="huber", choices=["l2", "huber", "none"], help="Type of loss for LCD."
    )
    lcd_group.add_argument("--huber_c", type=float, default=0.001, help="Huber loss parameter.")
    lcd_group.add_argument(
        "--timestep_scaling_factor",
        type=float,
        default=10.0,
        help=(
            "The multiplicative timestep scaling factor used when calculating the boundary scalings for LCM. The"
            " higher the scaling is, the lower the approximation error, but the default value of 10.0 should typically"
            " suffice."
        ),
    )


def add_lora_arguments(parser):
    """Add LoRA specific arguments."""
    lora_group = parser.add_argument_group("LoRA Specific")
    lora_group.add_argument("--lora_rank", type=int, default=64, help="Rank of the LoRA projection matrix.")
    lora_group.add_argument("--lora_dropout", type=float, default=0.1, help="Dropout probability for LoRA.")


def add_vae_arguments(parser):
    """Add VAE arguments."""
    vae_group = parser.add_argument_group("VAE Arguments")
    vae_group.add_argument("--vae_encode_batch_size", type=int, default=8, help="Batch size for encoding images.")
    vae_group.add_argument("--vae_decode_batch_size", type=int, default=16, help="Batch size for decoding images.")


def add_ema_arguments(parser):
    """Add Exponential Moving Average arguments."""
    ema_group = parser.add_argument_group("Exponential Moving Average")
    ema_group.add_argument("--use_ema", type=str2bool, default=False, help="Use EMA.")
    ema_group.add_argument("--ema_decay", type=float, default=0.95, help="EMA decay factor.")


def add_mixed_precision_arguments(parser):
    """Add mixed precision arguments."""
    mixed_prec_group = parser.add_argument_group("Mixed Precision")
    mixed_prec_group.add_argument(
        "--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"], help="Use mixed precision."
    )
    mixed_prec_group.add_argument(
        "--global_bf16", type=str2bool, default=False, help="Override dtype to bf16 if supported."
    )
    mixed_prec_group.add_argument(
        "--cast_teacher_unet", action="store_true", help="Cast teacher U-Net to specified precision."
    )


def add_reward_arguments(parser):
    """Add reward function arguments."""
    reward_group = parser.add_argument_group("Reward Arguments")
    reward_group.add_argument("--reward_fn_name", type=str, default="hpsv2", help="Image Reward function name.")
    reward_group.add_argument("--reward_scale", type=float, default=1.0, help="Scale of the reward loss.")
    reward_group.add_argument(
        "--image_rm_ckpt_dir", type=str, default="PATH/TO/HPS_v2_compressed.ckpt", help="Image-Text reward model path."
    )
    reward_group.add_argument("--video_rm_name", type=str, default="vi_clip2", help="Video Reward function name.")
    reward_group.add_argument(
        "--video_rm_ckpt_dir",
        type=str,
        default="PATH/TO/InternVideo2-stage2_1b-224p-f4.pt",
        help="Video-Text reward model path.",
    )
    reward_group.add_argument("--video_reward_scale", type=float, default=1.0, help="Scale of the viclip reward loss.")


def parse_args():
    """Parse command-line arguments for the training script."""
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    # Add argument groups
    add_model_checkpoint_loading_arguments(parser)
    add_ms_environment_arguments(parser)
    add_training_arguments(parser)
    add_learning_rate_arguments(parser)
    add_optimizer_arguments(parser)
    add_lcd_arguments(parser)
    add_lora_arguments(parser)
    add_vae_arguments(parser)
    add_ema_arguments(parser)
    add_mixed_precision_arguments(parser)
    add_reward_arguments(parser)

    # Parse arguments
    args = parser.parse_args()

    # Handle local rank environment variable
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Validate reward function settings
    if args.video_rm_name == "vi_clip":
        assert args.video_rm_batch_size == 8
    elif args.video_rm_name == "vi_clip2":
        assert args.video_rm_batch_size in [4, 8]
    else:
        raise ValueError(f"Unsupported video reward function: {args.video_rm_name}")

    return args
