#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import copy
import functools
import logging
import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
import yaml
from datasets import load_dataset
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoTokenizer

import mindspore as ms
from mindspore import mint, nn, ops
from mindspore.amp import auto_mixed_precision
from mindspore.dataset import GeneratorDataset, transforms, vision

from mindone.diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
from mindone.diffusers.models.controlnet_flux import FluxControlNetModel
from mindone.diffusers.optimization import get_scheduler
from mindone.diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline
from mindone.diffusers.training_utils import (
    AttrJitWrapper,
    compute_density_for_timestep_sampling,
    init_distributed_device,
    is_master,
    prepare_train_network,
    pynative_no_grad,
    set_seed,
)
from mindone.transformers import CLIPTextModel, T5EncoderModel
from mindone.utils.config import str2bool

logger = logging.getLogger(__name__)


def do_ckpt_combine_online(net_to_save, optimizer_parallel_group):
    """
    Combine the model parameters when saving weighs during zero3 training.
    """
    new_net_to_save = []
    all_gather_op = ops.AllGather(optimizer_parallel_group)

    #  net_to_save is a dict with elements as {"name":name, "data": data}
    for param in net_to_save:
        if param["data"].parallel_optimizer:
            new_data = ms.Tensor(all_gather_op(param["data"]).asnumpy())
        else:
            new_data = ms.Tensor(param["data"].asnumpy())
        new_net_to_save.append({"name": param["name"], "data": new_data})
    return new_net_to_save


def log_validation(pipeline, args, step, trackers, logging_dir, is_final_validation=False):
    logger.info("Running validation... ")

    if args.seed is None:
        generator = None
    else:
        generator = np.random.Generator(np.random.PCG64(seed=args.seed))

    if len(args.validation_image) == len(args.validation_prompt):
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt
    elif len(args.validation_image) == 1:
        validation_images = args.validation_image * len(args.validation_prompt)
        validation_prompts = args.validation_prompt
    elif len(args.validation_prompt) == 1:
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt * len(args.validation_image)
    else:
        raise ValueError(
            "number of `args.validation_image` and `args.validation_prompt` should be checked in `parse_args`"
        )

    image_logs = []
    for validation_prompt, validation_image in zip(validation_prompts, validation_images):
        from mindone.diffusers.utils import load_image

        validation_image = load_image(validation_image)
        # maybe need to inference on 1024 to get a good image
        validation_image = validation_image.resize((args.resolution, args.resolution))

        images = []

        # pre calculate  prompt embeds, pooled prompt embeds, text ids because t5 does not support autocast
        prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
            validation_prompt, prompt_2=validation_prompt
        )
        for _ in range(args.num_validation_images):
            image = pipeline(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                control_image=validation_image,
                num_inference_steps=28,
                controlnet_conditioning_scale=0.7,
                guidance_scale=3.5,
                generator=generator,
            )[0][0]
            image = image.resize((args.resolution, args.resolution))
            images.append(image)
        image_logs.append(
            {"validation_image": validation_image, "images": images, "validation_prompt": validation_prompt}
        )

    def get_valid_filename(name):
        import re

        s = str(name).strip().replace(" ", "_")
        s = re.sub(r"(?u)[^-\w.]", "", s)
        if s in {"", ".", ".."}:
            raise ValueError(f"Cannot get valid filename from '{name}'")
        return s

    tracker_key = "test" if is_final_validation else "validation"
    if is_master(args):
        validation_logging_dir = os.path.join(logging_dir, tracker_key, f"step{step}")
        os.makedirs(validation_logging_dir, exist_ok=True)
        for log in image_logs:
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_logging_dir_sub = os.path.join(validation_logging_dir, get_valid_filename(validation_prompt))
            os.makedirs(validation_logging_dir_sub, exist_ok=True)
            validation_image.save(os.path.join(validation_logging_dir_sub, "validation_image.jpg"))
            for idx, image in enumerate(images):
                image.save(os.path.join(validation_logging_dir_sub, f"{idx:04d}.jpg"))

    for tracker_name, tracker_writer in trackers.items():
        if tracker_name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]
                formatted_images = []
                formatted_images.append(np.asarray(validation_image))
                for image in images:
                    formatted_images.append(np.asarray(image))
                formatted_images = np.stack(formatted_images)
                tracker_writer.add_images(
                    get_valid_filename(validation_prompt), formatted_images, step, dataformats="NHWC"
                )
        else:
            logger.warning(f"image logging not implemented for {tracker_name}")

    logger.info("Validation done.")
    return image_logs


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to an improved VAE to stabilize training. For more details check out: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--crops_coords_top_left_h",
        type=int,
        default=0,
        help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
    )
    parser.add_argument(
        "--crops_coords_top_left_w",
        type=int,
        default=0,
        help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
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
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help=("Number of subprocesses to use for data loading."),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
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
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--distributed", default=False, action="store_true", help="Enable distributed training")
    parser.add_argument(
        "--enable_mindspore_data_sink",
        action="store_true",
        help=(
            "Whether or not to enable `Data Sinking` feature from MindData which boosting data "
            "fetching and transferring from host to device. For more information, see "
            "https://www.mindspore.cn/tutorials/experts/en/r2.2/optimize/execution_opt.html#data-sinking. "
            "Note: To avoid breaking the iteration logic of the training, the size of data sinking is set to 1."
        ),
    )
    parser.add_argument(
        "--mindspore_mode",
        type=int,
        default=0,
        choices=[0, 1],
        help="Forms of MindSpore programming execution, 0 means static graph mode and 1 means dynamic graph mode.",
    )
    parser.add_argument(
        "--jit_level",
        type=str,
        default="O1",
        choices=["O0", "O1", "O2"],
        help=(
            "Used to control the compilation optimization level, supports [O0, O1, O2]. The framework automatically "
            "selects the execution method. O0: All optimizations except those necessary for functionality are "
            "disabled, using an operator-by-operator execution method. O1: Enables common optimizations and automatic "
            "operator fusion optimizations, using an operator-by-operator execution method. This is an experimental "
            "optimization level, which is continuously being improved. O2: Enables extreme performance optimization, "
            "using a sinking execution method. Only effective when args.mindspore_mode is 0"
        ),
    )
    parser.add_argument(
        "--zero_stage",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="ZeRO-Stage in data parallel.",
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_iterator_no_copy",
        default=True,
        type=str2bool,
        help="dataset iterator optimization strategy. Whether dataset iterator creates a Tensor without copy.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the controlnet conditioning image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_double_layers",
        type=int,
        default=4,
        help="Number of double layers in the controlnet (default: 4).",
    )
    parser.add_argument(
        "--num_single_layers",
        type=int,
        default=4,
        help="Number of single layers in the controlnet (default: 4).",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=2,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="flux_train_controlnet",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--jsonl_for_train",
        type=str,
        default=None,
        help="Path to the jsonl file containing the training data.",
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="the guidance scale used for transformer.",
    )

    parser.add_argument(
        "--save_weight_dtype",
        type=str,
        default="fp32",
        choices=[
            "fp16",
            "bf16",
            "fp32",
        ],
        help=("Preserve precision type according to selected weight"),
    )

    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="logit_normal",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    # parser.add_argument(
    #     "--enable_model_cpu_offload",
    #     action="store_true",
    #     help="Enable model cpu offload and save memory.",
    # )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.jsonl_for_train is None:
        raise ValueError("Specify either `--dataset_name` or `--jsonl_for_train`")

    if args.dataset_name is not None and args.jsonl_for_train is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--jsonl_for_train`")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if (
        args.validation_image is not None
        and args.validation_prompt is not None
        and len(args.validation_image) != 1
        and len(args.validation_prompt) != 1
        and len(args.validation_image) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s"
        )

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args


def get_train_dataset(args):
    dataset = None
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    if args.jsonl_for_train is not None:
        # load from json
        dataset = load_dataset("json", data_files=args.jsonl_for_train, cache_dir=args.cache_dir)
        dataset = dataset.flatten_indices()
    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    if args.image_column is None:
        image_column = column_names[0]
        logger.info(f"image column defaulting to {image_column}")
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"`--image_column` value '{args.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.caption_column is None:
        caption_column = column_names[1]
        logger.info(f"caption column defaulting to {caption_column}")
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"`--caption_column` value '{args.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.conditioning_image_column is None:
        conditioning_image_column = column_names[2]
        logger.info(f"conditioning image column defaulting to {conditioning_image_column}")
    else:
        conditioning_image_column = args.conditioning_image_column
        if conditioning_image_column not in column_names:
            raise ValueError(
                f"`--conditioning_image_column` value '{args.conditioning_image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"  # noqa E501
            )

    train_dataset = dataset["train"].shuffle(seed=args.seed)
    if args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(args.max_train_samples))
    return train_dataset


def _get_t5_prompt_embeds(
    tokenizer,
    text_encoder,
    prompt,
    num_images_per_prompt=1,
    max_sequence_length=512,
):
    batch_size = len(prompt)
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="np",
    )
    text_input_ids = text_inputs.input_ids

    prompt_embeds = text_encoder(ms.Tensor.from_numpy(text_input_ids), output_hidden_states=False)[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.tile((1, num_images_per_prompt, 1))
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _get_clip_prompt_embeds(
    tokenizer,
    text_encoder,
    prompt,
    num_images_per_prompt=1,
):
    batch_size = len(prompt)
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_overflowing_tokens=False,
        return_length=False,
        return_tensors="np",
    )

    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(ms.Tensor.from_numpy(text_input_ids), output_hidden_states=False)

    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds[1]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.tile((1, num_images_per_prompt))
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds


# Adapted from pipelines.FluxControlNetPipeline.encode_prompt
def encode_prompt(prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train=True):
    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    # We only use the pooled prompt output from the CLIPTextModel
    pooled_prompt_embeds = _get_clip_prompt_embeds(
        tokenizer=tokenizers[0],
        text_encoder=text_encoders[0],
        prompt=captions,
    )
    prompt_embeds = _get_t5_prompt_embeds(
        tokenizer=tokenizers[1],
        text_encoder=text_encoders[1],
        prompt=captions,
        max_sequence_length=512,
    )
    dtype = text_encoders[0].dtype
    text_ids = mint.zeros((prompt_embeds.shape[1], 3), dtype=dtype)

    return prompt_embeds, pooled_prompt_embeds, text_ids


def prepare_train_dataset(dataset, args):
    image_transforms = transforms.Compose(
        [
            vision.Resize(args.resolution, interpolation=vision.Inter.BILINEAR),
            vision.CenterCrop(args.resolution),
            vision.ToTensor(),
            vision.Normalize([0.5], [0.5], is_hwc=False),
        ]
    )

    conditioning_image_transforms = transforms.Compose(
        [
            vision.Resize(args.resolution, interpolation=vision.Inter.BILINEAR),
            vision.CenterCrop(args.resolution),
            vision.ToTensor(),
            vision.Normalize([0.5], [0.5], is_hwc=False),
        ]
    )

    def preprocess_train(examples):
        images = [
            (image.convert("RGB") if not isinstance(image, str) else Image.open(image).convert("RGB"))
            for image in examples[args.image_column]
        ]
        images = [image_transforms(image)[0] for image in images]

        conditioning_images = [
            (image.convert("RGB") if not isinstance(image, str) else Image.open(image).convert("RGB"))
            for image in examples[args.conditioning_image_column]
        ]
        conditioning_images = [conditioning_image_transforms(image)[0] for image in conditioning_images]
        examples["pixel_values"] = images
        examples["conditioning_pixel_values"] = conditioning_images

        return examples

    dataset = dataset.with_transform(preprocess_train)

    return dataset


def collate_fn(examples):
    pixel_values = np.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.astype(np.float32)

    conditioning_pixel_values = np.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.astype(np.float32)

    prompt_ids = np.stack([example["prompt_embeds"] for example in examples]).astype(np.float32)

    pooled_prompt_embeds = np.stack([example["pooled_prompt_embeds"] for example in examples]).astype(np.float32)
    text_ids = np.stack([example["text_ids"] for example in examples]).astype(np.float32)

    return pixel_values, conditioning_pixel_values, prompt_ids, pooled_prompt_embeds, text_ids


def main():
    args = parse_args()
    ms.set_context(
        mode=ms.GRAPH_MODE,
        jit_config={"jit_level": args.jit_level},
    )

    init_distributed_device(args)  # read attr distributed, writer attrs rank/local_rank/world_size

    logging_out_dir = Path(args.output_dir, args.logging_dir)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        # DEBUG, INFO, WARNING, ERROR, CRITICAL
        level=logging.INFO,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if is_master(args):
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(logging_out_dir, exist_ok=True)

    # Load the tokenizers
    # load clip tokenizer
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    # load t5 tokenizer
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )
    # load clip text encoder
    text_encoder_one = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    # load t5 text encoder
    text_encoder_two = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    flux_transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
    )
    if args.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        flux_controlnet = FluxControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
    else:
        logger.info("Initializing controlnet weights from transformer")
        # we can define the num_layers, num_single_layers,
        flux_controlnet = FluxControlNetModel.from_transformer(
            flux_transformer,
            attention_head_dim=flux_transformer.config["attention_head_dim"],
            num_attention_heads=flux_transformer.config["num_attention_heads"],
            num_layers=args.num_double_layers,
            num_single_layers=args.num_single_layers,
        )
    logger.info("all models loaded successfully")

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    def freeze_params(m: nn.Cell):
        for p in m.get_parameters():
            p.requires_grad = False

    freeze_params(vae)
    freeze_params(flux_transformer)
    freeze_params(text_encoder_one)
    freeze_params(text_encoder_two)
    vae.set_grad(requires_grad=False)
    flux_transformer.set_grad(requires_grad=False)
    text_encoder_one.set_grad(requires_grad=False)
    text_encoder_two.set_grad(requires_grad=False)

    if args.gradient_checkpointing:
        flux_transformer.enable_gradient_checkpointing()
        flux_controlnet.enable_gradient_checkpointing()

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = ms.float32
    if args.mixed_precision == "fp16":
        weight_dtype = ms.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = ms.bfloat16

    vae.to(dtype=weight_dtype)
    flux_transformer.to(dtype=weight_dtype)

    # Make sure the trainable params are in float32. and do AMP wrapper manually
    if weight_dtype != ms.float32:
        flux_controlnet = auto_mixed_precision(flux_controlnet, amp_level="auto", dtype=weight_dtype)

    def compute_embeddings(
        batch,
        proportion_empty_prompts,
        text_encoders,
        tokenizers,
        weight_dtype,
        is_train=True,
    ):
        prompt_batch = batch[args.caption_column]

        prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
            prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train
        )

        # text_ids [512,3] to [bs,512,3]
        text_ids = text_ids.unsqueeze(0).broadcast_to((prompt_embeds.shape[0], -1, -1))
        return {
            "prompt_embeds": prompt_embeds.numpy(),
            "pooled_prompt_embeds": pooled_prompt_embeds.numpy(),
            "text_ids": text_ids.numpy(),
        }

    train_dataset = get_train_dataset(args)
    text_encoders = [text_encoder_one, text_encoder_two]
    tokenizers = [tokenizer_one, tokenizer_two]
    compute_embeddings_fn = functools.partial(
        compute_embeddings,
        text_encoders=text_encoders,
        tokenizers=tokenizers,
        proportion_empty_prompts=args.proportion_empty_prompts,
        weight_dtype=weight_dtype,
    )
    from datasets.fingerprint import Hasher

    # fingerprint used by the cache for the other processes to load the result
    # details: https://github.com/huggingface/diffusers/pull/4038#discussion_r1266078401
    new_fingerprint = Hasher.hash(args)
    train_dataset = train_dataset.map(
        compute_embeddings_fn, batched=True, new_fingerprint=new_fingerprint, batch_size=50
    )

    # Then get the training dataset ready to be passed to the dataloader.
    train_dataset = prepare_train_dataset(train_dataset, args)

    class DatasetForMindData:
        def __init__(self, data):
            self.data = data

        def __getitem__(self, idx):
            idx = idx.item() if isinstance(idx, np.integer) else idx
            return self.data[idx]

        def __len__(self):
            return len(self.data)

    train_dataloader = GeneratorDataset(
        DatasetForMindData(train_dataset),
        column_names=["example"],
        shuffle=True,
        shard_id=args.rank,
        num_shards=args.world_size,
        num_parallel_workers=args.dataloader_num_workers,
    ).batch(
        batch_size=args.train_batch_size,
        per_batch_map=lambda examples, batch_info: collate_fn(examples),
        input_columns=["example"],
        output_columns=[
            "c1",
            "c2",
            "c3",
            "c4",
            "c5",
        ],  # pixel_values, conditioning_pixel_values, prompt_ids, pooled_prompt_embeds, text_ids
        num_parallel_workers=args.dataloader_num_workers,
    )

    del text_encoder_one, text_encoder_two, text_encoders
    del tokenizer_one, tokenizer_two, tokenizers

    # Print trainable parameters statistics
    flux_controlnet_trainable = sum(p.numel() for p in flux_controlnet.trainable_params())
    logger.info(f"{flux_controlnet.__class__.__name__:<30s} ==> Trainable params: {flux_controlnet_trainable:<10,d}")

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * args.world_size
        )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        args.learning_rate,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Optimizer creation
    params_to_optimize = flux_controlnet.trainable_params()
    optimizer = nn.AdamWeightDecay(
        params_to_optimize,
        learning_rate=lr_scheduler,
        beta1=args.adam_beta1,
        beta2=args.adam_beta2,
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # create train_step for training
    network_with_loss = FluxControlNetWithLoss(
        vae=vae,
        flux_transformer=flux_transformer,
        flux_controlnet=flux_controlnet,
        noise_scheduler=noise_scheduler_copy,
        weight_dtype=weight_dtype,
        args=args,
    ).set_train(True)

    loss_scaler = nn.FixedLossScaleUpdateCell(loss_scale_value=2**12)

    train_step = prepare_train_network(
        network_with_loss,
        optimizer=optimizer,
        scale_sense=loss_scaler,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        clip_grad=True,
        clip_norm=args.max_grad_norm,
        zero_stage=args.zero_stage,
    )

    if args.enable_mindspore_data_sink:
        sink_process = ms.data_sink(train_step, train_dataloader)
    else:
        sink_process = None

    # create pipeline for validation
    if args.validation_prompt is not None:
        pipeline = FluxControlNetPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            controlnet=flux_controlnet,
            transformer=flux_transformer,
            mindspore_dtype=ms.bfloat16,
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if is_master(args):
        with open(logging_out_dir / "hparams.yml", "w") as f:
            yaml.dump(vars(args), f, indent=4)
    trackers = dict()
    for tracker_name in args.report_to.split(","):
        if tracker_name == "tensorboard":
            from tensorboardX import SummaryWriter

            trackers[tracker_name] = SummaryWriter(str(logging_out_dir), write_to_disk=is_master(args))
        else:
            logger.warning(f"Tracker {tracker_name} is not implemented, omitting...")

    # Train!
    total_batch_size = args.train_batch_size * args.world_size * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.zero_stage == 3:
            raise NotImplementedError(
                "currently we save combined checkpoint during zero3 training. resume not implemented yet"
            )

        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            if is_master(args):
                logger.info(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            if is_master(args):
                logger.info(f"Resuming from checkpoint {path}")
            # TODO: load optimizer & grad scaler etc. like accelerator.load_state
            input_model_file = os.path.join(args.output_dir, path, "diffusion_pytorch_model.safetensors")
            ms.load_param_into_net(
                flux_controlnet, ms.load_checkpoint(input_model_file, format="safetensors"), strict_load=True
            )
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not is_master(args),
    )
    # do_copy=False enables the dataset iterator to not do copy when creating a tensor which takes less time.
    # Currently the default value of do_copy is True,
    # it is expected that the default value of do_copy will be changed to False in MindSpore 2.7.0.
    train_dataloader_iter = train_dataloader.create_tuple_iterator(
        num_epochs=args.num_train_epochs - first_epoch,
        do_copy=not args.dataset_iterator_no_copy,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        flux_controlnet.set_train(True)
        for step, batch in (
            ((_, None) for _ in range(len(train_dataloader)))  # dummy iterator
            if args.enable_mindspore_data_sink
            else enumerate(train_dataloader_iter)
        ):
            if args.enable_mindspore_data_sink:
                loss, _, _ = sink_process()
            else:
                loss, _, _ = train_step(*batch)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if train_step.accum_steps == 1 or train_step.cur_accum_step.item() == 0:
                progress_bar.update(1)
                global_step += 1
                prefix = "flux_controlnet."

                if global_step % args.checkpointing_steps == 0:
                    net_to_save = [
                        {"name": p.name[len(prefix) :], "data": p} for p in flux_controlnet.trainable_params()
                    ]
                    net_to_save = (
                        net_to_save
                        if args.zero_stage != 3
                        else do_ckpt_combine_online(net_to_save, train_step.zero_helper.optimizer_parallel_group)
                    )

                    if is_master(args):
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        # TODO: save optimizer & grad scaler etc. like accelerator.save_state
                        os.makedirs(save_path, exist_ok=True)
                        output_model_file = os.path.join(save_path, "diffusion_pytorch_model.safetensors")
                        flux_controlnet.save_config(save_path)
                        ms.save_checkpoint(net_to_save, output_model_file, format="safetensors")
                        logger.info(f"Saved state to {save_path}")

                    if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                        log_validation(
                            pipeline=pipeline,
                            args=args,
                            trackers=trackers,
                            weight_dtype=weight_dtype,
                            step=global_step,
                        )
            logs = {"loss": loss.numpy().item(), "lr": optimizer.get_lr().numpy().item()}
            progress_bar.set_postfix(**logs)
            for tracker_name, tracker in trackers.items():
                if tracker_name == "tensorboard":
                    tracker.add_scalars("train", logs, global_step)
            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    save_weight_dtype = ms.float32
    if args.save_weight_dtype == "fp16":
        save_weight_dtype = ms.float16
    elif args.save_weight_dtype == "bf16":
        save_weight_dtype = ms.bfloat16
    flux_controlnet.to(save_weight_dtype)

    if args.zero_stage != 3:
        if is_master(args):
            if args.save_weight_dtype != "fp32":
                flux_controlnet.save_pretrained(args.output_dir, variant=args.save_weight_dtype)
            else:
                flux_controlnet.save_pretrained(args.output_dir)

    else:
        prefix = "flux_controlnet."
        net_to_save = [{"name": p.name[len(prefix) :], "data": p} for p in flux_controlnet.trainable_params()]
        net_to_save = do_ckpt_combine_online(net_to_save, train_step.zero_helper.optimizer_parallel_group)
        if is_master(args):
            flux_controlnet.save_config(args.output_dir)
            output_model_file = os.path.join(args.output_dir, "diffusion_pytorch_model.safetensors")
            ms.save_checkpoint(net_to_save, output_model_file, format="safetensors")

    if is_master(args):
        # Run a final round of validation.
        # Setting `vae`, `unet`, and `controlnet` to None to load automatically from `args.output_dir`.
        if args.validation_prompt is not None:
            log_validation(
                pipeline=pipeline,
                args=args,
                weight_dtype=weight_dtype,
                step=global_step,
                is_final_validation=True,
            )

    # End of training
    for tracker_name, tracker in trackers.items():
        if tracker_name == "tensorboard":
            tracker.close()


class FluxControlNetWithLoss(nn.Cell):
    def __init__(
        self,
        vae: nn.Cell,
        flux_transformer: nn.Cell,
        flux_controlnet: nn.Cell,
        noise_scheduler,
        weight_dtype,
        args,
    ):
        super().__init__()
        self.flux_controlnet = flux_controlnet
        self.flux_transformer = flux_transformer
        self.flux_transformer_config_guidance_embeds = flux_transformer.config.guidance_embeds
        self.vae = vae
        self.vae_dtype = vae.dtype
        self.vae_config_scaling_factor = vae.config.scaling_factor
        self.vae_config_shift_factor = vae.config.shift_factor
        self.vae_config_block_out_channels = vae.config.block_out_channels
        self.vae_scale_factor = 2 ** (len(self.vae_config_block_out_channels))

        self.noise_scheduler = noise_scheduler
        self.noise_scheduler_num_train_timesteps = noise_scheduler.config.num_train_timesteps
        self.weight_dtype = weight_dtype
        self.args = AttrJitWrapper(**vars(args))

    def get_sigmas(self, indices, n_dim=4, dtype=ms.float32):
        """
        origin `get_sigmas` which uses timesteps to get sigmas might be not supported
        in mindspore Graph mode, thus we rewrite `get_sigmas` to get sigma directly
        from indices which calls less ops and could run in mindspore Graph mode.
        """
        sigma = self.noise_scheduler.sigmas[indices].to(dtype=dtype)
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def construct(self, pixel_values, conditioning_pixel_values, prompt_ids, pooled_prompt_embeds, text_ids):
        # Convert images to latent space
        # vae encode
        with pynative_no_grad():
            pixel_latents_tmp = self.vae.diag_gauss_dist.sample(self.vae.encode(pixel_values.to(self.vae_dtype))[0])
            pixel_latents_tmp = (pixel_latents_tmp - self.vae_config_shift_factor) * self.vae_config_scaling_factor

        pixel_latents_tmp = pixel_latents_tmp.to(self.weight_dtype)
        pixel_latents = FluxControlNetPipeline._pack_latents(
            pixel_latents_tmp,
            pixel_values.shape[0],
            pixel_latents_tmp.shape[1],
            pixel_latents_tmp.shape[2],
            pixel_latents_tmp.shape[3],
        )

        with pynative_no_grad():
            control_values = conditioning_pixel_values.to(dtype=self.weight_dtype)
            control_latents = self.vae.diag_gauss_dist.sample(self.vae.encode(control_values.to(self.vae_dtype))[0])

        control_latents = (control_latents - self.vae_config_shift_factor) * self.vae_config_scaling_factor
        control_image = FluxControlNetPipeline._pack_latents(
            control_latents,
            control_values.shape[0],
            control_latents.shape[1],
            control_latents.shape[2],
            control_latents.shape[3],
        )

        latent_image_ids = FluxControlNetPipeline._prepare_latent_image_ids(
            batch_size=pixel_latents_tmp.shape[0],  # no use
            height=pixel_latents_tmp.shape[2] // 2,
            width=pixel_latents_tmp.shape[3] // 2,
            dtype=self.weight_dtype,
        )

        bsz = pixel_latents.shape[0]
        noise = mint.randn_like(pixel_latents).to(dtype=self.weight_dtype)
        # Sample a random timestep for each image
        # for weighting schemes where we sample timesteps non-uniformly
        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.args.weighting_scheme,
            batch_size=bsz,
            logit_mean=self.args.logit_mean,
            logit_std=self.args.logit_std,
            mode_scale=self.args.mode_scale,
        )
        indices = (u * self.noise_scheduler_num_train_timesteps).long()
        timesteps = self.noise_scheduler.timesteps[indices]

        # Add noise according to flow matching.
        sigmas = self.get_sigmas(indices, n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype)
        noisy_model_input = sigmas * noise + (1.0 - sigmas) * pixel_latents

        # handle guidance
        if self.flux_transformer_config_guidance_embeds:
            guidance_vec = mint.full(
                (noisy_model_input.shape[0],),
                self.args.guidance_scale,
                dtype=self.weight_dtype,
            )
        else:
            guidance_vec = None

        controlnet_block_samples, controlnet_single_block_samples = self.flux_controlnet(
            hidden_states=noisy_model_input,
            controlnet_cond=control_image,
            timestep=timesteps / 1000,
            guidance=guidance_vec,
            pooled_projections=pooled_prompt_embeds.to(dtype=self.weight_dtype),
            encoder_hidden_states=prompt_ids.to(dtype=self.weight_dtype),
            txt_ids=text_ids[0].to(dtype=self.weight_dtype),  # TODO:check if have [0]
            img_ids=latent_image_ids,
            return_dict=False,
        )

        noise_pred = self.flux_transformer(
            hidden_states=noisy_model_input,
            timestep=timesteps / 1000,
            guidance=guidance_vec,
            pooled_projections=pooled_prompt_embeds.to(dtype=self.weight_dtype),
            encoder_hidden_states=prompt_ids.to(dtype=self.weight_dtype),
            controlnet_block_samples=[sample.to(dtype=self.weight_dtype) for sample in controlnet_block_samples]
            if controlnet_block_samples is not None
            else None,
            controlnet_single_block_samples=[
                sample.to(dtype=self.weight_dtype) for sample in controlnet_single_block_samples
            ]
            if controlnet_single_block_samples is not None
            else None,
            txt_ids=text_ids[0].to(dtype=self.weight_dtype),  # TODO:check if have [0]
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]

        loss = mint.nn.functional.mse_loss(noise_pred.float(), (noise - pixel_latents).float(), reduction="mean")

        return loss


if __name__ == "__main__":
    main()
