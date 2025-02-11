#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Hacked together by / Copyright 2024 Genius Patrick @ MindSpore Team.
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
import logging
import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTokenizer

import mindspore as ms
from mindspore import nn, ops
from mindspore.amp import StaticLossScaler
from mindspore.dataset import GeneratorDataset, vision

from mindone.diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
from mindone.diffusers.optimization import get_scheduler
from mindone.diffusers.training_utils import AttrJitWrapper, TrainStep, init_distributed_device, is_master, set_seed
from mindone.diffusers.utils import PIL_INTERPOLATION
from mindone.safetensors.mindspore import save_file as safe_save_file
from mindone.transformers import CLIPTextModel, CLIPTextModelWithProjection

logger = logging.getLogger(__name__)


def unwrap_model(model, prefix=""):
    for name, param in model.parameters_and_names(name_prefix=prefix):
        param.name = name
    return model


def log_validation(pipeline, args, trackers, logging_dir, global_step, is_final_validation=False):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    # run inference
    generator = None if args.seed is None else np.random.Generator(np.random.PCG64(seed=args.seed))
    images = []
    for _ in range(args.num_validation_images):
        image = pipeline(args.validation_prompt, num_inference_steps=25, generator=generator)[0][0]
        images.append(image)

    tracker_key = "test" if is_final_validation else "validation"
    if is_master(args):
        validation_logging_dir = os.path.join(logging_dir, tracker_key, f"step{global_step}")
        os.makedirs(validation_logging_dir, exist_ok=True)
        for idx, img in enumerate(images):
            img.save(os.path.join(validation_logging_dir, f"{idx:04d}.jpg"))
    for tracker_name, tracker_writer in trackers.items():
        if tracker_name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker_writer.add_images(tracker_key, np_images, global_step, dataformats="NHWC")
        else:
            logger.warning(f"image logging not implemented for {tracker_name}")

    logger.info("Validation done.")
    return images


def save_progress(text_encoder, placeholder_token_ids, args, save_path, safe_serialization=True):
    logger.info("Saving embeddings")
    learned_embeds = text_encoder.get_input_embeddings().embedding_table[
        min(placeholder_token_ids) : max(placeholder_token_ids) + 1
    ]
    learned_embeds_dict = {args.placeholder_token: learned_embeds}

    if safe_serialization:
        safe_save_file(learned_embeds_dict, save_path, metadata={"format": "np"})
    else:
        ms.save_checkpoint(learned_embeds_dict, save_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save learned_embeds.bin every X updates steps.",
    )
    parser.add_argument(
        "--save_as_full_pipeline",
        action="store_true",
        help="Save the complete stable diffusion pipeline.",
    )
    parser.add_argument(
        "--num_vectors",
        type=int,
        default=1,
        help="How many textual inversion vectors shall be used to learn the concept.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--train_data_dir", type=str, default=None, required=True, help="A folder containing the training data."
    )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default=None,
        required=True,
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--initializer_token", type=str, default=None, required=True, help="A token to use as initializer word."
    )
    parser.add_argument("--learnable_property", type=str, default="object", help="Choose between 'object' and 'style'")
    parser.add_argument("--repeats", type=int, default=100, help="How many times to repeat the training data.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
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
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution."
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=5000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
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
        default=1e-4,
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
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of subprocesses to use for data loading.",
    )
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
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
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
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
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
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
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
    parser.add_argument("--distributed", default=False, action="store_true", help="Enable distributed training")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
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
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    args = parser.parse_args()

    if args.train_data_dir is None:
        raise ValueError("You must specify a train data directory.")

    # Limitations for NOW.
    if args.push_to_hub is True:
        raise ValueError(
            "You cannot use --push_to_hub due to a security risk of uploading your data to huggingface-hub. "
            "If you know what you are doing, just delete this line and try again."
        )

    return args


imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]


class TextualInversionDataset(object):
    def __init__(
        self,
        data_root,
        tokenizer_1,
        tokenizer_2,
        learnable_property="object",  # [object, style]
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        center_crop=False,
    ):
        self.data_root = data_root
        self.tokenizer_1 = tokenizer_1
        self.tokenizer_2 = tokenizer_2
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        self.templates = imagenet_style_templates_small if learnable_property == "style" else imagenet_templates_small
        self.flip_transform = vision.RandomHorizontalFlip(prob=self.flip_p)
        self.crop = vision.CenterCrop(size) if center_crop else vision.RandomCrop(size)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        text = random.choice(self.templates).format(placeholder_string)

        original_size = (image.height, image.width)

        image = image.resize((self.size, self.size), resample=self.interpolation)

        if self.center_crop:
            y1 = max(0, int(round((image.height - self.size) / 2.0)))
            x1 = max(0, int(round((image.width - self.size) / 2.0)))
            image = self.crop(image)
        else:
            h, w = image.height, image.width
            th, tw = self.size, self.size
            if h < th or w < tw:
                raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")
            y1 = np.random.randint(0, h - th + 1, size=(1,)).item()
            x1 = np.random.randint(0, w - tw + 1, size=(1,)).item()
            image = image.crop((x1, y1, x1 + tw, y1 + th))

        crop_top_left = (y1, x1)
        add_time_id = original_size + crop_top_left + (self.size, self.size)

        input_ids_1 = self.tokenizer_1(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer_1.model_max_length,
            return_tensors="np",
        ).input_ids[0]

        input_ids_2 = self.tokenizer_2(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer_2.model_max_length,
            return_tensors="np",
        ).input_ids[0]

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        image = Image.fromarray(img)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        pixel_values = image.transpose((2, 0, 1))
        return pixel_values, input_ids_1, input_ids_2, add_time_id


def main():
    args = parse_args()
    ms.set_context(mode=ms.GRAPH_MODE, jit_syntax_level=ms.STRICT)
    init_distributed_device(args)  # read attr distributed, writer attrs rank/local_rank/world_size

    # tensorboard, mindinsight, wandb logging stuff into logging_dir
    logging_dir = Path(args.output_dir, args.logging_dir)

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
    if is_master(args):
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(logging_dir, exist_ok=True)

    # Load tokenizer
    tokenizer_1 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder_1 = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )
    # set sample_size of unet
    unet.register_to_config(sample_size=args.resolution // (2 ** (len(vae.config.block_out_channels) - 1)))

    # Add the placeholder token in tokenizer_1
    placeholder_tokens = [args.placeholder_token]

    if args.num_vectors < 1:
        raise ValueError(f"--num_vectors has to be larger or equal to 1, but is {args.num_vectors}")

    # add dummy tokens for multi-vector
    additional_tokens = []
    for i in range(1, args.num_vectors):
        additional_tokens.append(f"{args.placeholder_token}_{i}")
    placeholder_tokens += additional_tokens

    num_added_tokens = tokenizer_1.add_tokens(placeholder_tokens)
    if num_added_tokens != args.num_vectors:
        raise ValueError(
            f"The tokenizer already contains the token {args.placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )

    # Convert the initializer_token, placeholder_token to ids
    token_ids = tokenizer_1.encode(args.initializer_token, add_special_tokens=False)
    # Check if initializer_token is a single token or a sequence of tokens
    if len(token_ids) > 1:
        raise ValueError("The initializer token must be a single token.")

    initializer_token_id = token_ids[0]
    placeholder_token_ids = tokenizer_1.convert_tokens_to_ids(placeholder_tokens)

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder_1.resize_token_embeddings(len(tokenizer_1))

    # Initialise the newly added placeholder token with the embeddings of the initializer token
    token_embeds = text_encoder_1.get_input_embeddings().embedding_table.data
    for token_id in placeholder_token_ids:
        token_embeds[token_id] = token_embeds[initializer_token_id].copy()

    # Freeze vae and unet
    def freeze_params(m: nn.Cell):
        for p in m.get_parameters():
            p.requires_grad = False

    freeze_params(vae)
    freeze_params(unet)
    freeze_params(text_encoder_2)
    # Freeze all parameters except for the token embeddings in text encoder
    freeze_params(text_encoder_1.text_model.encoder)
    freeze_params(text_encoder_1.text_model.final_layer_norm)
    freeze_params(text_encoder_1.text_model.embeddings.position_embedding)
    text_encoder_1.set_train(True)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to
    # half-precision as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = ms.float32
    if args.mixed_precision == "fp16":
        weight_dtype = ms.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = ms.bfloat16

    # Move vae and unet and text_encoder_2 to device and cast to weight_dtype
    unet.to(dtype=weight_dtype)
    vae.to(dtype=weight_dtype)
    text_encoder_2.to(dtype=weight_dtype)

    if args.gradient_checkpointing:
        text_encoder_1.gradient_checkpointing_enable()

    if args.enable_xformers_memory_efficient_attention:
        unet.enable_xformers_memory_efficient_attention()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        raise NotImplementedError("TF32 data type is not yet supported, please do not set --allow_tf32")

    # Dataset and DataLoaders creation:
    train_dataset = TextualInversionDataset(
        data_root=args.train_data_dir,
        tokenizer_1=tokenizer_1,
        tokenizer_2=tokenizer_2,
        size=args.resolution,
        placeholder_token=" ".join(tokenizer_1.convert_ids_to_tokens(placeholder_token_ids)),
        repeats=args.repeats,
        learnable_property=args.learnable_property,
        center_crop=args.center_crop,
        set="train",
    )
    train_dataloader = GeneratorDataset(
        train_dataset,
        column_names=["pixel_values", "input_ids_1", "input_ids_2", "add_time_ids"],
        shuffle=True,
        shard_id=args.rank,
        num_shards=args.world_size,
        num_parallel_workers=args.dataloader_num_workers,
    ).batch(
        batch_size=args.train_batch_size,
        num_parallel_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
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
    )

    # Initialize the optimizer
    optimizer = nn.AdamWeightDecay(
        text_encoder_1.get_input_embeddings().trainable_params(),  # only optimize the embeddings
        learning_rate=lr_scheduler,
        beta1=args.adam_beta1,
        beta2=args.adam_beta2,
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Prepare everything with our `accelerator`.
    text_encoder_1.to(weight_dtype)
    text_encoder_1.get_input_embeddings().embedding_table.set_dtype(ms.float32)
    # TODO: We will update the training methods during mixed precision training to ensure the performance and strategies during the training process.
    if args.mixed_precision and args.mixed_precision != "no":
        text_encoder_1.get_input_embeddings().to_float(weight_dtype)
    # In `StableDiffusionPipeline.encode_prompt`, `prompt_embeds` is cast according to `text_encoder.dtype`
    # This unsafe and ugly patch affects all instances of `MSPreTrainedModel`. Improve it in the future!
    from mindone.transformers import MSPreTrainedModel

    MSPreTrainedModel.dtype = weight_dtype

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if is_master(args):
        with open(logging_dir / "hparams.yml", "w") as f:
            yaml.dump(vars(args), f, indent=4)
    trackers = dict()
    for tracker_name in args.report_to.split(","):
        if tracker_name == "tensorboard":
            from tensorboardX import SummaryWriter

            trackers[tracker_name] = SummaryWriter(str(logging_dir), write_to_disk=is_master(args))
        else:
            logger.warning(f"Tracker {tracker_name} is not implemented, omitting...")

    train_step = TrainStepForTI(
        vae=vae,
        text_encoder_1=text_encoder_1,
        text_encoder_2=text_encoder_2,
        unet=unet,
        optimizer=optimizer,
        noise_scheduler=noise_scheduler,
        weight_dtype=weight_dtype,
        length_of_dataloader=len(train_dataloader),
        args=args,
    ).set_train()

    if args.enable_mindspore_data_sink:
        sink_process = ms.data_sink(train_step, train_dataloader)
    else:
        sink_process = None

    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=text_encoder_1,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer_1,
        tokenizer_2=tokenizer_2,
        unet=unet,
        vae=vae,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        mindspore_dtype=weight_dtype,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.set_progress_bar_config(disable=True)

    # Train!
    total_batch_size = args.train_batch_size * args.world_size * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
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
            input_model_file = os.path.join(args.output_dir, path, "pytorch_model.ckpt")
            ms.load_param_into_net(unet, ms.load_checkpoint(input_model_file))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    # run inference
    if args.validation_prompt is not None:
        log_validation(pipeline, args, trackers, logging_dir, initial_global_step)

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not is_master(args),
    )

    # keep original embeddings as reference
    orig_embeds_params = text_encoder_1.get_input_embeddings().embedding_table.data.copy()

    train_dataloader_iter = train_dataloader.create_tuple_iterator(num_epochs=args.num_train_epochs - first_epoch)
    for epoch in range(first_epoch, args.num_train_epochs):
        text_encoder_1.set_train(True)
        for step, batch in (
            ((_, None) for _ in range(len(train_dataloader)))  # dummy iterator
            if args.enable_mindspore_data_sink
            else enumerate(train_dataloader_iter)
        ):
            if args.enable_mindspore_data_sink:
                loss, model_pred = sink_process()
            else:
                loss, model_pred = train_step(*batch)

            # Let's make sure we don't update any embedding weights besides the newly added token
            index_no_updates = ops.ones((len(tokenizer_1),), dtype=ms.bool_)
            index_no_updates[min(placeholder_token_ids) : max(placeholder_token_ids) + 1] = False
            text_encoder_1.get_input_embeddings().embedding_table[index_no_updates] = orig_embeds_params[
                index_no_updates
            ]

            # Checks if the accelerator has performed an optimization step behind the scenes
            if train_step.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if global_step % args.save_steps == 0:
                    weight_name = f"learned_embeds-steps-{global_step}.safetensors"
                    save_path = os.path.join(args.output_dir, weight_name)
                    save_progress(
                        text_encoder_1,
                        placeholder_token_ids,
                        args,
                        save_path,
                        safe_serialization=True,
                    )

                if is_master(args):
                    if global_step % args.checkpointing_steps == 0:
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
                        output_model_file = os.path.join(save_path, "pytorch_model.ckpt")
                        ms.save_checkpoint(unet, output_model_file)
                        logger.info(f"Saved state to {save_path}")

                if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                    log_validation(
                        pipeline,
                        args,
                        trackers,
                        logging_dir,
                        global_step,
                    )

            logs = {"loss": loss.numpy().item(), "lr": optimizer.get_lr().numpy().item()}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    if args.validation_prompt:
        log_validation(
            pipeline,
            args,
            trackers,
            logging_dir,
            global_step,
            is_final_validation=True,
        )

    # Create the pipeline using the trained modules and save it.
    if is_master(args):
        if args.save_as_full_pipeline:
            pipeline.save_pretrained(args.output_dir)
        # Save the newly trained embeddings
        weight_name = "learned_embeds.safetensors"
        save_path = os.path.join(args.output_dir, weight_name)
        save_progress(
            text_encoder_1,
            placeholder_token_ids,
            args,
            save_path,
            safe_serialization=True,
        )

    for tracker_name, tracker in trackers.items():
        if tracker_name == "tensorboard":
            tracker.close()


class TrainStepForTI(TrainStep):
    def __init__(
        self,
        vae: nn.Cell,
        text_encoder_1: nn.Cell,
        text_encoder_2: nn.Cell,
        unet: nn.Cell,
        optimizer: nn.Optimizer,
        noise_scheduler,
        weight_dtype,
        length_of_dataloader,
        args,
    ):
        super().__init__(
            text_encoder_1,
            optimizer,
            StaticLossScaler(65536),
            None,
            args.gradient_accumulation_steps,
            gradient_accumulation_kwargs=dict(length_of_dataloader=length_of_dataloader),
        )
        self.unet = unet
        self.vae = vae
        self.vae_scaling_factor = self.vae.config.scaling_factor
        self.text_encoder_1 = text_encoder_1
        self.text_encoder_2 = text_encoder_2
        self.noise_scheduler = noise_scheduler
        self.noise_scheduler_num_train_timesteps = noise_scheduler.config.num_train_timesteps
        self.noise_scheduler_prediction_type = noise_scheduler.config.prediction_type
        self.weight_dtype = weight_dtype
        self.args = AttrJitWrapper(**vars(args))

    def forward(self, pixel_values, input_ids_1, input_ids_2, add_time_ids):
        # Convert images to latent space
        latents = self.vae.diag_gauss_dist.sample(self.vae.encode(pixel_values.to(self.weight_dtype))[0])
        latents = latents * self.vae_scaling_factor

        # Sample noise that we'll add to the latents
        noise = ops.randn_like(latents, dtype=latents.dtype)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = ops.randint(0, self.noise_scheduler_num_train_timesteps, (bsz,))
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states_1 = self.text_encoder_1(input_ids_1, output_hidden_states=True)[-1][-2].to(
            dtype=self.weight_dtype
        )
        encoder_output_2 = self.text_encoder_2(input_ids_2.reshape(input_ids_1.shape[0], -1), output_hidden_states=True)
        encoder_hidden_states_2 = encoder_output_2[-1][-2].to(dtype=self.weight_dtype)
        added_cond_kwargs = {"text_embeds": encoder_output_2[0], "time_ids": add_time_ids}
        encoder_hidden_states = ops.cat([encoder_hidden_states_1, encoder_hidden_states_2], axis=-1)

        # Predict the noise residual
        model_pred = self.unet(
            noisy_latents, timesteps, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs, return_dict=False
        )[0]

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler_prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler_prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler_prediction_type}")

        loss = ops.mse_loss(model_pred.float(), target.float(), reduction="mean")

        loss = self.scale_loss(loss)
        return loss, model_pred


if __name__ == "__main__":
    main()
