#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The Huawei Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
from copy import deepcopy
import functools
import gc
import logging
import math
import os
import sys
import random
import shutil
import yaml
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import mindspore as ms
from mindspore import ops, mint, nn
from mindspore.amp import StaticLossScaler

from tqdm.auto import tqdm

from mindone.diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from mindone.diffusers.optimization import get_scheduler
from mindone.diffusers.utils import check_min_version, is_wandb_available
from mindone.diffusers.training_utils import (
    set_seed,
    TrainStep,
    cast_training_params,
)
from mindone.diffusers.utils import convert_state_dict_to_diffusers, convert_unet_state_dict_to_peft
from mindone.diffusers.loaders import LoraLoaderMixin
from mindone.diffusers._peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from mindone.diffusers._peft.tuners.tuners_utils import BaseTunerLayer
from mindone.utils.logger import set_logger
from mindone.utils.config import str2bool

from data.dataset import create_dataloader
from utils.env import init_env
from utils.lora import save_lora_weight
from utils.lora_handler import LoraHandler
from ode_solver import DDIMSolver
from reward_fn import get_reward_fn
from scheduler.t2v_turbo_scheduler import T2VTurboScheduler
from pipeline.t2v_turbo_vc2_pipeline import T2VTurboVC2Pipeline
from utils.common_utils import (
    append_dims,
    create_optimizer_params,
    get_predicted_noise,
    get_predicted_original_sample,
    guidance_scale_embedding,
    handle_trainable_modules,
    huber_loss,
    log_validation_video,
    param_optim,
    scalings_for_boundary_conditions,
    tuple_type,
    load_model_checkpoint,
)
from utils.utils import instantiate_from_config, freeze_params

sys.path.append("./mindone/examples/stable_diffusion_xl")
from gm.modules.embedders.open_clip.tokenizer import tokenize

MAX_SEQ_LENGTH = 77

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.26.0.dev0")

logger = logging.getLogger(__name__)


def _to_abspath(rp):
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(__dir__, rp)


def log_validation(pretrained_t2v, unet, scheduler, model_config, args, trackers):
    logger.info("Running validation... ")
    pretrained_t2v.model.diffusion_model = unet
    pipeline = T2VTurboVC2Pipeline(pretrained_t2v, scheduler, model_config)

    log_validation_video(pipeline, args, trackers, save_fps=16)
    gc.collect()


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
        default="PATH_TO_VC2_model.pt",
        help="Path to the pretrained model.",
    )
    # ----------MS environment args----------
    parser.add_argument(
        "--device_target", type=str, default="Ascend", help="Ascend or GPU"
    )
    parser.add_argument(
        "--mode",
        type=int,
        default=0,
        help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)",
    )
    parser.add_argument(
        "--use_parallel", default=False, type=str2bool, help="use parallel"
    )
    parser.add_argument(
        "--debug", type=str2bool, default=False, help="Execute inference in debug mode."
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
    parser.add_argument(
        "--seed", type=int, default=453645634, help="A seed for reproducible training."
    )
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
        "--vlcd_processes",
        type=tuple_type,
        default=(0, 1, 2, 3, 4, 5),
        help="Process idx that are used to perform consistency distillation.",
    )
    parser.add_argument(
        "--reward_train_processes",
        type=tuple_type,
        default=(0, 1, 2, 3, 4, 5),
        help="Process idx that are used to maximize text-img reward fn.",
    )
    parser.add_argument(
        "--video_rm_train_processes",
        type=tuple_type,
        default=(6, 7),
        help="Process idx that are used to maximize text-video reward fn.",
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
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
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
    parser.add_argument(
        "--adam_weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
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
        choices=["l2", "huber"],
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
        help="Reward function name",
    )
    parser.add_argument(
        "--reward_scale",
        type=float,
        default=1.0,
        help="The scale of the reward loss",
    )
    parser.add_argument(
        "--video_rm_name",
        type=str,
        default="vi_clip2",
        help="Reward function name",
    )
    parser.add_argument(
        "--video_rm_ckpt_dir",
        type=str,
        # default="PATH/TO/ViClip-InternVid-10M-FLT.pth",
        default="PATH/TO/InternVideo2-stage2_1b-224p-f4.pt",
        help="Reward function name",
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


# Adapted from pipelines.StableDiffusionPipeline.encode_prompt
def encode_prompt(prompt_batch, text_encoder, is_train=True):
    captions = []
    for caption in prompt_batch:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with ms._no_grad():
        prompt_embeds = text_encoder(prompt_batch)

    return prompt_embeds


def compute_embeddings(prompt_batch, text_encoder, is_train=True):
    prompt_embeds = encode_prompt(prompt_batch, text_encoder, is_train)
    return {"prompt_embeds": prompt_embeds}


def main(args):
    args = parse_args()
    ms.set_context(mode=args.mode, jit_syntax_level=ms.STRICT)
    rank_id, device_num = init_env(
        args.mode,
        args.seed,
        args.use_parallel,
        device_target=args.device_target,
        jit_level=args.jit_level,
        global_bf16=args.global_bf16,
        debug=args.debug,
    )

    logging_dir = Path(args.output_dir, args.logging_dir)
    set_logger(name="", output_dir=logging_dir)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # 5. Load teacher Model
    config = OmegaConf.load(args.pretrained_model_cfg)
    model_config = config.pop("model", OmegaConf.create())
    pretrained_t2v = instantiate_from_config(model_config)
    pretrained_t2v = load_model_checkpoint(
        pretrained_t2v,
        args.pretrained_model_path,
    )

    vae = pretrained_t2v.first_stage_model
    vae_scale_factor = model_config["params"]["scale_factor"]
    text_encoder = pretrained_t2v.cond_stage_model
    teacher_unet = pretrained_t2v.model.diffusion_model

    # 6. Freeze teacher vae, text_encoder, and teacher_unet
    freeze_params(vae)
    freeze_params(text_encoder)
    freeze_params(teacher_unet)

    # 7. Create online student U-Net. This will be updated by the optimizer (e.g. via backpropagation.)
    # Add `time_cond_proj_dim` to the student U-Net if `teacher_unet.config.time_cond_proj_dim` is None
    time_cond_proj_dim = (
        teacher_unet.time_cond_proj_dim
        if teacher_unet.time_cond_proj_dim is not None
        else args.unet_time_cond_proj_dim
    )
    unet_config = model_config["params"]["unet_config"]
    unet_config["params"]["time_cond_proj_dim"] = time_cond_proj_dim
    unet = instantiate_from_config(unet_config)
    # load teacher_unet weights into unet
    ms.load_param_into_net(unet, teacher_unet.parameters_dict(), strict_load=False)
    freeze_params(unet)
    unet.set_train(True)

    use_unet_lora = True
    lora_manager = LoraHandler(
        version="cloneofsimo",
        use_unet_lora=use_unet_lora,
        save_for_webui=True,
        unet_replace_modules=["UNetModel"],
    )

    unet_lora_params, unet_negation = lora_manager.add_lora_to_model(
        use_unet_lora,
        unet,
        lora_manager.unet_replace_modules,
        dropout=args.lora_dropout,
        r=args.lora_rank,
    )

    if args.reward_scale > 0:
        reward_fn = get_reward_fn(args.reward_fn_name, precision=args.mixed_precision)
    else:
        reward_fn = None
    if args.video_reward_scale > 0:
        video_rm_fn = get_reward_fn(
            args.video_rm_name,
            precision=args.mixed_precision,
            rm_ckpt_dir=args.video_rm_ckpt_dir,
            n_frames=args.video_rm_batch_size,
        )
    else:
        video_rm_fn = None

    # 1. Create the noise scheduler and the desired noise schedule.
    noise_scheduler = T2VTurboScheduler(
        linear_start=model_config["params"]["linear_start"],
        linear_end=model_config["params"]["linear_end"],
    )

    # DDPMScheduler calculates the alpha and sigma noise schedules (based on the alpha bars) for us
    alpha_schedule = ops.sqrt(noise_scheduler.alphas_cumprod)
    sigma_schedule = ops.sqrt(1 - noise_scheduler.alphas_cumprod)
    # Initialize the DDIM ODE solver for distillation.

    if args.no_scale_pred_x0:
        use_scale = False
    else:
        use_scale = model_config["params"]["use_scale"]

    assert not use_scale
    scale_b = model_config["params"]["scale_b"]
    solver = DDIMSolver(
        noise_scheduler.alphas_cumprod.numpy(),
        ddim_timesteps=args.num_ddim_timesteps,
        use_scale=use_scale,
        scale_b=scale_b,
        ddim_eta=args.ddim_eta,
    )

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if unet.dtype != ms.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {unet.dtype}. {low_precision_error_string}"
        )

    # 9. Handle mixed precision and device placement
    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = ms.float32
    if args.mixed_precision == "fp16":
        weight_dtype = ms.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = ms.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    vae.to_float(weight_dtype)
    text_encoder.to_float(weight_dtype)

    # Move teacher_unet to device, optionally cast to weight_dtype
    if args.cast_teacher_unet:
        teacher_unet.to_float(weight_dtype)

    # 10. Handle saving and loading of checkpoints

    def save_model_hook(models, weights, output_dir):
        if rank_id == 0:
            unet_ = deepcopy(unet)
            save_lora_dir = os.path.join(output_dir, "unet_lora.pt")
            save_lora_weight(unet_, save_lora_dir, ["UNetModel"])
            for model in models:
                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()
            del unet_

    def load_model_hook(models, input_dir):
        unet_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unet)):
                unet_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir)

        unet_state_dict = {
            f'{k.replace("unet.", "")}': v
            for k, v in lora_state_dict.items()
            if k.startswith("unet.")
        }
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(
            unet_, unet_state_dict, adapter_name="default"
        )
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            models = [unet_]
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models)

    # Make sure the trainable params are in float32.
    models = [unet]
    if args.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=ms.float32)

    # Print trainable parameters statistics
    for peft_model in models:
        all_params = sum(p.numel() for p in peft_model.get_parameters())
        trainable_params = sum(p.numel() for p in peft_model.trainable_params())
        logger.info(
            f"{peft_model.__class__.__name__:<30s} ==> Trainable params: {trainable_params:<10,d} || "
            f"All params: {all_params:<16,d} || Trainable ratio: {trainable_params / all_params:.8%}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    optimizer_class = nn.AdamWeightDecay

    # Create parameters to optimize over with a condition (if "condition" is true, optimize it)
    extra_unet_params = {}
    trainable_modules_available = False
    optim_params = [
        param_optim(
            unet,
            trainable_modules_available,
            extra_params=extra_unet_params,
            negation=unet_negation,
        ),
        param_optim(
            unet_lora_params,
            use_unet_lora,
            is_lora=True,
            extra_params={**{"lr": args.learning_rate}, **extra_unet_params},
        ),
    ]
    params = create_optimizer_params(optim_params, args.learning_rate)

    # 12. Optimizer creation
    optimizer = optimizer_class(
        params,
        learning_rate=args.learning_rate,
        beta1=args.adam_beta1,
        beta2=args.adam_beta2,
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # 13. Dataset creation and data processing
    # Here, we compute not just the text embeddings but also the additional embeddings
    # needed for the SD XL UNet to operate.

    # decoder_kwargs = {
    #     "n_frames": args.n_frames,  # get 16 frames from each video
    #     "fps": 16,
    #     "num_threads": 12,  # use 16 threads to decode the video
    # }
    # resolution = tuple([s * 8 for s in model_config["params"]["image_size"]])
    # dataset = get_video_dataset(
    #     urls=args.train_shards_path_or_url,
    #     batch_size=args.train_batch_size,
    #     shuffle=1000,
    #     decoder_kwargs=decoder_kwargs,
    #     resize_size=resolution,
    #     crop_size=resolution,
    # )
    num_workers = args.dataloader_num_workers
    # train_dataloader = WebLoader(dataset, batch_size=None, num_workers=num_workers)

    resolution = tuple([s * 8 for s in model_config["params"]["image_size"]])
    csv_path = args.csv_path if args.csv_path is not None else os.path.join(args.data_path, "video_caption.csv")
    data_config = dict(
        video_folder=_to_abspath(args.data_path),
        csv_path=_to_abspath(csv_path),
        sample_size=resolution,
        sample_stride=1, #args.frame_stride,
        sample_n_frames=args.n_frames,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_parallel_workers=args.dataloader_num_workers,
        max_rowsize=64,
        random_drop_text=False,
    )

    train_dataloader = create_dataloader(
        data_config, is_image=False, device_num=device_num, rank_id=rank_id
    )

    num_train_examples = args.max_train_samples
    global_batch_size = args.train_batch_size * device_num
    num_worker_batches = math.ceil(
        num_train_examples / (global_batch_size * num_workers)
    )

    train_dataloader.num_batches = num_worker_batches * args.dataloader_num_workers
    train_dataloader.num_samples = train_dataloader.num_batches * global_batch_size

    # 14. LR Scheduler creation
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        train_dataloader.num_batches / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        # optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # 15. Prepare for training
    # Prepare everything with our `accelerator`.
    # unet, optimizer, lr_scheduler = accelerator.prepare(unet, optimizer, lr_scheduler)
    for peft_model in models:
        for _, module in peft_model.cells_and_names():
            if isinstance(module, BaseTunerLayer):
                for layer_name in module.adapter_layer_names:
                    module_dict = getattr(module, layer_name)
                    for key, layer in module_dict.items():
                        if key in module.active_adapters and isinstance(layer, nn.Cell):
                            layer.to_float(weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        train_dataloader.num_batches / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if rank_id==0:
        with open(logging_dir / "hparams.yml", "w") as f:
            yaml.dump(vars(args), f, indent=4)
    trackers = dict()
    for tracker_name in args.report_to.split(","):
        if tracker_name == "tensorboard":
            from tensorboardX import SummaryWriter

            trackers[tracker_name] = SummaryWriter(
                str(logging_dir), write_to_disk=(rank_id==0)
            )
        else:
            logger.warning(f"Tracker {tracker_name} is not implemented, omitting...")

    uncond_prompt, _ = tokenize([""] * args.train_batch_size)
    uncond_prompt = ms.Tensor(np.array(uncond_prompt, dtype=np.int32))
    uncond_prompt_embeds = text_encoder(uncond_prompt)
    if isinstance(uncond_prompt_embeds, DiagonalGaussianDistribution):
        uncond_prompt_embeds = uncond_prompt_embeds.mode()

    train_step = TrainStepForTurbo(
        vae=vae,
        text_encoder=text_encoder,
        teacher_unet=teacher_unet,
        unet=unet,
        optimizer=optimizer,
        tokenizer=tokenize,
        noise_scheduler=noise_scheduler,
        alpha_schedule=alpha_schedule,
        sigma_schedule=sigma_schedule,
        weight_dtype=weight_dtype,
        length_of_dataloader=len(train_dataloader),
        vae_scale_factor=vae_scale_factor,
        time_cond_proj_dim=time_cond_proj_dim,
        args=args,
        solver=solver,
        uncond_prompt_embeds=uncond_prompt_embeds,
        reward_fn=reward_fn,
        video_rm_fn=video_rm_fn,
    ).set_train()

    # 16. Train!
    total_batch_size = (
        args.train_batch_size * device_num * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num batches each epoch = {train_dataloader.num_batches}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    handle_trainable_modules(unet, None, is_enabled=True, negation=unet_negation)

    initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not (rank_id==0),
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.set_train(True)
        for step, batch in enumerate(train_dataloader):

            loss, distill_loss, reward_loss, video_rm_loss = train_step(*batch)

        # 11. Backpropagate on the online student model (`unet`)
        # if accelerator.sync_gradients: # TODO!!!
        #     accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)

        # Checks if the accelerator has performed an optimization step behind the scenes
        if train_step.sync_gradients:
            progress_bar.update(1)
            global_step += 1

            if rank_id == 0:
                if global_step % args.checkpointing_steps == 0:
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [
                            d
                            for d in checkpoints
                            if d.startswith("checkpoint") and not "rm" in d
                        ]
                        checkpoints = sorted(
                            checkpoints, key=lambda x: int(x.split("-")[1])
                        )

                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = (
                                len(checkpoints) - args.checkpoints_total_limit + 1
                            )
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(
                                f"removing checkpoints: {', '.join(removing_checkpoints)}"
                            )

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(
                                    args.output_dir, removing_checkpoint
                                )
                                shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(
                        args.output_dir, f"checkpoint-{global_step}"
                    )
                    os.makedirs(save_path, exist_ok=True)
                    save_model_hook(models, save_path)
                    output_model_file = os.path.join(save_path, "model.ckpt")
                    ms.save_checkpoint(unet, output_model_file)
                    logger.info(f"Saved state to {save_path}")
                    logger.info(f"Saved state to {save_path}")

                if (
                    global_step % args.validation_steps == 0
                    and args.report_to == "wandb"
                ):
                    log_validation(
                        pretrained_t2v,
                        unet,
                        noise_scheduler,
                        model_config,
                        args,
                        trackers,
                    )

            # Gather losses from all processes
            # distill_loss_list = accelerator.gather(distill_loss.detach())
            # reward_loss_list = accelerator.gather(reward_loss.detach().float())
            # video_rm_loss_list = accelerator.gather(
            #     video_rm_loss.detach().float()
            # )

            if rank_id == 0:
                logs = {
                    "distillation loss": distill_loss,
                    "image reward loss": reward_loss,
                    "video reward loss": video_rm_loss,
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                progress_bar.set_postfix(**logs)
                logger.info(logs, step=global_step)
                del distill_loss, reward_loss, video_rm_loss
                gc.collect()

        if global_step >= args.max_train_steps:
            break

    # End of training
    for tracker_name, tracker in trackers.items():
        if tracker_name == "tensorboard":
            tracker.close()


class TrainStepForTurbo(TrainStep):
    def __init__(
        self,
        vae: nn.Cell,
        text_encoder: nn.Cell,
        teacher_unet: nn.Cell,
        unet: nn.Cell,
        optimizer: nn.Optimizer,
        tokenizer,
        noise_scheduler,
        alpha_schedule,
        sigma_schedule,
        weight_dtype,
        length_of_dataloader,
        vae_scale_factor,
        time_cond_proj_dim,
        args,
        solver,
        uncond_prompt_embeds,
        reward_fn,
        video_rm_fn,
    ):
        super().__init__(
            unet,
            optimizer,
            StaticLossScaler(65536),
            args.max_grad_norm,
            args.gradient_accumulation_steps,
            gradient_accumulation_kwargs=dict(
                length_of_dataloader=length_of_dataloader
            ),
        )
        self.unet = unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.teacher_unet = teacher_unet
        self.noise_scheduler = noise_scheduler
        self.alpha_schedule = alpha_schedule
        self.sigma_schedule = sigma_schedule
        self.weight_dtype = weight_dtype
        self.vae_scale_factor = vae_scale_factor
        self.time_cond_proj_dim = time_cond_proj_dim
        self.args = args
        self.solver = solver
        self.uncond_prompt_embeds = uncond_prompt_embeds

        self.compute_embeddings_fn = functools.partial(
            compute_embeddings,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        self.reward_fn = reward_fn
        self.video_rm_fn = video_rm_fn

    def forward(self, pixel_values, text):

        text = text.numpy().astype(str)

        # 1. Load and process the image and text conditioning
        # video = ((video / 255.0).clamp(0.0, 1.0) - 0.5) / 0.5

        # Convert video from (b, t, h, w, c) to (b, t, c, h, w)
        # video = video.permute(0, 1, 4, 2, 3) # FIXME!!!
        # pixel_values = video.to(dtype=self.weight_dtype)
        pixel_values = pixel_values.to(dtype=self.weight_dtype)
        b, t = pixel_values.shape[:2]
        pixel_values_flatten = pixel_values.view(b * t, *pixel_values.shape[2:])
        # encode pixel values with batch size of at most args.vae_encode_batch_size
        latents = []
        for i in range(
            0, pixel_values_flatten.shape[0], self.args.vae_encode_batch_size
        ):
            latents.append(
                self.vae.encode(
                    pixel_values_flatten[i : i + self.args.vae_encode_batch_size]
                ).sample()
            )
        latents = mint.cat(latents, dim=0)
        latents = latents.view(b, t, *latents.shape[1:])
        # Convert latents from (b, t, c, h, w) to (b, c, t, h, w)
        latents = latents.permute(0, 2, 1, 3, 4)
        latents = latents * self.vae_scale_factor

        # assert not pretrained_t2v.scale_by_std
        latents = latents.to(self.weight_dtype)
        encoded_text = self.compute_embeddings_fn(text)
        bsz = latents.shape[0]

        # 2. Sample a random timestep for each image t_n from the ODE solver timesteps without bias.
        # For the DDIM solver, the timestep schedule is [T - 1, T - k - 1, T - 2 * k - 1, ...]
        index = ops.randint(0, args.num_ddim_timesteps, (bsz,))
        start_timesteps = self.solver.ddim_timesteps[index]
        timesteps = start_timesteps - args.topk
        timesteps = mint.where(timesteps < 0, mint.zeros_like(timesteps), timesteps)

        # 3. Get boundary scalings for start_timesteps and (end) timesteps.
        c_skip_start, c_out_start = scalings_for_boundary_conditions(
            start_timesteps, timestep_scaling=args.timestep_scaling_factor
        )
        c_skip_start, c_out_start = [
            append_dims(x, latents.ndim) for x in [c_skip_start, c_out_start]
        ]
        c_skip, c_out = scalings_for_boundary_conditions(
            timesteps, timestep_scaling=args.timestep_scaling_factor
        )
        c_skip, c_out = [append_dims(x, latents.ndim) for x in [c_skip, c_out]]

        # 4. Sample noise from the prior and add it to the latents according to the noise magnitude at each
        # timestep (this is the forward diffusion process) [z_{t_{n + k}} in Algorithm 1]
        noise = ops.randn_like(latents)
        noisy_model_input = self.noise_scheduler.add_noise(
            latents, noise, start_timesteps
        )

        # 5. Sample a random guidance scale w from U[w_min, w_max] and embed it
        w = (args.w_max - args.w_min) * ops.rand((bsz,)) + args.w_min
        w_embedding = guidance_scale_embedding(w, embedding_dim=self.time_cond_proj_dim)
        w = w.reshape(bsz, 1, 1, 1, 1)
        # Move to U-Net device and dtype
        w = w.to(dtype=latents.dtype)
        w_embedding = w_embedding.to(dtype=latents.dtype)

        # 6. Prepare prompt embeds and unet_added_conditions
        prompt_embeds = encoded_text.pop("prompt_embeds")

        # 7. Get online LCM prediction on z_{t_{n + k}} (noisy_model_input), w, c, t_{n + k} (start_timesteps)
        context = {"context": mint.cat([prompt_embeds.float()], 1), "fps": 16}
        noise_pred = self.unet(
            noisy_model_input,
            start_timesteps,
            **context,
            timestep_cond=w_embedding,
        )
        pred_x_0 = get_predicted_original_sample(
            noise_pred,
            start_timesteps,
            noisy_model_input,
            "epsilon",
            self.alpha_schedule,
            self.sigma_schedule,
        )

        model_pred = c_skip_start * noisy_model_input + c_out_start * pred_x_0

        distill_loss = mint.zeros_like(model_pred).mean()
        reward_loss = mint.zeros_like(model_pred).mean()
        video_rm_loss = mint.zeros_like(model_pred).mean()
        # if (
        #     accelerator.process_index in args.reward_train_processes
        #     and args.reward_scale > 0
        # ):
        if args.reward_scale > 0:
            # sample args.reward_batch_size frames
            assert args.train_batch_size == 1
            idx = ops.randint(0, t, (args.reward_batch_size,))

            selected_latents = (
                model_pred[:, :, idx].to(self.vae.dtype) / self.vae_scale_factor
            )
            num_images = args.train_batch_size * args.reward_batch_size
            selected_latents = selected_latents.permute(0, 2, 1, 3, 4)
            selected_latents = selected_latents.reshape(
                num_images, *selected_latents.shape[2:]
            )
            decoded_imgs = self.vae.decode(selected_latents)
            decoded_imgs = (decoded_imgs / 2 + 0.5).clamp(0, 1)
            expert_rewards = self.reward_fn(decoded_imgs, text)
            reward_loss = -expert_rewards.mean() * args.reward_scale
        # if (
        #     accelerator.process_index in args.video_rm_train_processes
        #     and args.video_reward_scale > 0
        # ):
        if args.video_reward_scale > 0:
            assert args.train_batch_size == 1
            assert t > args.video_rm_batch_size

            skip_frames = t // args.video_rm_batch_size
            start_id = ops.randint(0, skip_frames, (1,))[0].item()
            idx = mint.arange(start_id, t, skip_frames)[: args.video_rm_batch_size]
            assert len(idx) == args.video_rm_batch_size

            selected_latents = (
                model_pred[:, :, idx].to(self.vae.dtype) / self.vae_scale_factor
            )
            num_images = args.train_batch_size * args.video_rm_batch_size
            selected_latents = selected_latents.permute(0, 2, 1, 3, 4)
            selected_latents = selected_latents.reshape(
                num_images, *selected_latents.shape[2:]
            )
            decoded_imgs = self.vae.decode(selected_latents)
            decoded_imgs = (decoded_imgs / 2 + 0.5).clamp(0, 1)
            decoded_imgs = decoded_imgs.reshape(
                args.train_batch_size,
                args.video_rm_batch_size,
                *decoded_imgs.shape[1:],
            )
            video_rewards = self.video_rm_fn(decoded_imgs, text)
            video_rm_loss = -video_rewards.mean() * args.video_reward_scale
        # if accelerator.process_index in args.vlcd_processes:
        # 8. Compute the conditional and unconditional teacher model predictions to get CFG estimates of the
        # predicted noise eps_0 and predicted original sample x_0, then run the ODE solver using these
        # estimates to predict the data point in the augmented PF-ODE trajectory corresponding to the next ODE
        # solver timestep.
        with ms._no_grad():
            # 8.1. Get teacher model prediction on noisy_model_input z_{t_{n + k}} and conditional embedding c
            cond_teacher_output = self.teacher_unet(
                noisy_model_input.to(self.weight_dtype),
                start_timesteps,
                **context,
            )
            cond_pred_x0 = get_predicted_original_sample(
                cond_teacher_output,
                start_timesteps,
                noisy_model_input,
                "epsilon",
                self.alpha_schedule,
                self.sigma_schedule,
            )
            cond_pred_noise = get_predicted_noise(
                cond_teacher_output,
                start_timesteps,
                noisy_model_input,
                "epsilon",
                self.alpha_schedule,
                self.sigma_schedule,
            )

            # 8.2. Get teacher model prediction on noisy_model_input z_{t_{n + k}} and unconditional embedding 0
            uncond_teacher_output = self.teacher_unet(
                noisy_model_input.to(self.weight_dtype),
                start_timesteps,
                context=self.uncond_prompt_embeds.to(self.weight_dtype),
            )
            uncond_pred_x0 = get_predicted_original_sample(
                uncond_teacher_output,
                start_timesteps,
                noisy_model_input,
                "epsilon",
                self.alpha_schedule,
                self.sigma_schedule,
            )
            uncond_pred_noise = get_predicted_noise(
                uncond_teacher_output,
                start_timesteps,
                noisy_model_input,
                "epsilon",
                self.alpha_schedule,
                self.sigma_schedule,
            )

            # 8.3. Calculate the CFG estimate of x_0 (pred_x0) and eps_0 (pred_noise)
            # Note that this uses the LCM paper's CFG formulation rather than the Imagen CFG formulation
            pred_x0 = cond_pred_x0 + w * (cond_pred_x0 - uncond_pred_x0)
            pred_noise = cond_pred_noise + w * (cond_pred_noise - uncond_pred_noise)
            # 8.4. Run one step of the ODE solver to estimate the next point x_prev on the
            # augmented PF-ODE trajectory (solving backward in time)
            # Note that the DDIM step depends on both the predicted x_0 and source noise eps_0.
            x_prev = self.solver.ddim_step(pred_x0, pred_noise, index)

            # 9. Get target LCM prediction on x_prev, w, c, t_n (timesteps)
            with ms._no_grad():
                target_noise_pred = self.unet(
                    x_prev.float(),
                    timesteps,
                    **context,
                    timestep_cond=w_embedding,
                )
                pred_x_0 = get_predicted_original_sample(
                    target_noise_pred,
                    timesteps,
                    x_prev,
                    "epsilon",  
                    self.alpha_schedule,
                    self.sigma_schedule,
                ) 
                target = c_skip * x_prev + c_out * pred_x_0

            # 10. Calculate loss
            if args.loss_type == "l2":
                distill_loss = ops.mse_loss(
                    model_pred.float(), target.float(), reduction="mean"
                )
            elif args.loss_type == "huber":
                distill_loss = huber_loss(model_pred, target, args.huber_c)

        # accelerator.backward(distill_loss + reward_loss + video_rm_loss)
        loss = distill_loss + reward_loss + video_rm_loss
        return loss, distill_loss, reward_loss, video_rm_loss


if __name__ == "__main__":
    args = parse_args()
    main(args)
