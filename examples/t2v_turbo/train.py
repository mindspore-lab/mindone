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

from train_args import parse_args
from data.dataset import create_dataloader
from utils.env import init_env
from utils.lora import save_lora_weight
from utils.lora_handler import LoraHandler
from ode_solver import DDIMSolver
from reward_fn import get_reward_fn
from scheduler.t2v_turbo_scheduler import T2VTurboScheduler
from pipeline.lcd_with_loss import LCDWithLoss
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

    lcd_with_loss = LCDWithLoss(
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
        use_recompute=True,
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


    # End of training
    for tracker_name, tracker in trackers.items():
        if tracker_name == "tensorboard":
            tracker.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
