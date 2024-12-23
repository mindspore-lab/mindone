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
import gc
import itertools
import logging
import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
import yaml
from huggingface_hub.utils import insecure_hashlib
from PIL import Image
from PIL.ImageOps import exif_transpose
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast

import mindspore as ms
from mindspore import nn, ops
from mindspore.amp import StaticLossScaler
from mindspore.dataset import GeneratorDataset, transforms, vision

from mindone.diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)
from mindone.diffusers.optimization import get_scheduler
from mindone.diffusers.training_utils import AttrJitWrapper, TrainStep, init_distributed_device, is_master, set_seed
from mindone.transformers import CLIPTextModelWithProjection, T5EncoderModel

logger = logging.getLogger(__name__)


def load_text_encoders(args, class_one, class_two, class_three):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    text_encoder_three = class_three.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_3", revision=args.revision, variant=args.variant
    )
    return text_encoder_one, text_encoder_two, text_encoder_three


def log_validation(
    pipeline,
    args,
    trackers,
    logging_dir,
    pipeline_args,
    epoch,
    is_final_validation=False,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )

    # run inference
    generator = None if args.seed is None else np.random.Generator(np.random.PCG64(seed=args.seed))
    images = [pipeline(**pipeline_args, generator=generator)[0][0] for _ in range(args.num_validation_images)]

    phase_name = "test" if is_final_validation else "validation"
    if is_master(args):
        validation_logging_dir = os.path.join(logging_dir, phase_name, f"epoch{epoch}")
        os.makedirs(validation_logging_dir, exist_ok=True)
        for idx, img in enumerate(images):
            img.save(os.path.join(validation_logging_dir, f"{idx:04d}.jpg"))
    for tracker_name, tracker_writer in trackers.items():
        if tracker_name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker_writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
        else:
            logger.warning(f"image logging not implemented for {tracker_name}")

    logger.info("Validation done.")
    return images


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModelWithProjection":
        from mindone.transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from mindone.transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
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
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private,"
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
        "--instance_data_dir",
        type=str,
        default=None,
        help=("A folder containing the training data. "),
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing the target image. By "
        "default, the standard Image Dataset maps out 'file_name' "
        "to 'image'.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default=None,
        help="The column of the dataset containing the instance prompt for each image",
    )

    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")

    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=True,
        help="The prompt with identifier specifying the instance, e.g. 'photo of a TOK dog', 'in the style of TOK'",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
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
        "--validation_epochs",
        type=int,
        default=50,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd3-dreambooth",
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
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images.")
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
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
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
        "--text_encoder_lr",
        type=float,
        default=5e-6,
        help="Text encoder learning rate to use.",
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
        "--weighting_scheme", type=str, default="logit_normal", choices=["sigma_sqrt", "logit_normal", "mode"]
    )
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--mode_scale", type=float, default=1.29)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodidy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_weight_decay_text_encoder", type=float, default=1e-03, help="Weight decay to use for text_encoder"
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
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
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument("--distributed", default=False, action="store_true", help="Enable distributed training")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.instance_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--instance_data_dir`")

    if args.dataset_name is not None and args.instance_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--instance_data_dir`")

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        if args.class_data_dir is not None:
            logger.warning("You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompt is not None:
            logger.warning("You need not use --class_prompt without --with_prior_preservation.")

    # Limitations for NOW.
    def error_template(feature, flag):
        return f"{feature} is not yet supported, please do not set --{flag}"

    assert args.allow_tf32 is False, error_template("TF32 Data Type", "allow_tf32")
    assert args.optimizer == "AdamW", error_template("Optimizer besides AdamW", "optimizer")
    assert args.use_8bit_adam is False, error_template("AdamW8bit", "use_8bit_adam")
    if args.push_to_hub is True:
        raise ValueError(
            "You cannot use --push_to_hub due to a security risk of uploading your data to huggingface-hub. "
            "If you know what you are doing, just delete this line and try again."
        )

    return args


class DreamBoothDataset(object):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images.
    """

    def __init__(
        self,
        args,
        instance_data_root,
        instance_prompt,
        tokenizer_one,
        tokenizer_two,
        tokenizer_three,
        class_prompt,
        class_data_root=None,
        class_num=None,
        size=1024,
        repeats=1,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two
        self.tokenizer_three = tokenizer_three

        self.instance_prompt = instance_prompt
        self.custom_instance_prompts = None
        self.class_prompt = class_prompt

        # if --dataset_name is provided or a metadata jsonl file is provided in the local --instance_data directory,
        # we load the training data using load_dataset
        if args.dataset_name is not None:
            try:
                from datasets import load_dataset
            except ImportError:
                raise ImportError(
                    "You are trying to load your data using the datasets library. If you wish to train using custom "
                    "captions please install the datasets library: `pip install datasets`. If you wish to load a "
                    "local folder containing images only, specify --instance_data_dir instead."
                )
            # Downloading and loading a dataset from the hub.
            # See more about loading custom images at
            # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script
            dataset = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                cache_dir=args.cache_dir,
            )
            # Preprocessing the datasets.
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
            instance_images = dataset["train"][image_column]

            if args.caption_column is None:
                logger.info(
                    "No caption column provided, defaulting to instance_prompt for all images. If your dataset "
                    "contains captions/prompts for the images, make sure to specify the "
                    "column as --caption_column"
                )
                self.custom_instance_prompts = None
            else:
                if args.caption_column not in column_names:
                    raise ValueError(
                        f"`--caption_column` value '{args.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                    )
                custom_instance_prompts = dataset["train"][args.caption_column]
                # create final list of captions according to --repeats
                self.custom_instance_prompts = []
                for caption in custom_instance_prompts:
                    self.custom_instance_prompts.extend(itertools.repeat(caption, repeats))
        else:
            self.instance_data_root = Path(instance_data_root)
            if not self.instance_data_root.exists():
                raise ValueError("Instance images root doesn't exists.")

            instance_images = [Image.open(path) for path in list(Path(instance_data_root).iterdir())]
            self.custom_instance_prompts = None

        self.instance_images = []
        for img in instance_images:
            self.instance_images.extend(itertools.repeat(img, repeats))

        self.pixel_values = []
        train_resize = vision.Resize(size, interpolation=vision.Inter.BILINEAR)
        train_crop = vision.CenterCrop(size) if center_crop else vision.RandomCrop(size)
        train_flip = vision.RandomHorizontalFlip(prob=1.0)
        train_transforms = transforms.Compose(
            [
                vision.ToTensor(),
                vision.Normalize([0.5], [0.5], is_hwc=False),
            ]
        )
        for image in self.instance_images:
            image = exif_transpose(image)
            if not image.mode == "RGB":
                image = image.convert("RGB")
            image = train_resize(image)
            if args.random_flip and random.random() < 0.5:
                # flip
                image = train_flip(image)
            if args.center_crop:
                y1 = max(0, int(round((image.height - args.resolution) / 2.0)))
                x1 = max(0, int(round((image.width - args.resolution) / 2.0)))
                image = train_crop(image)
            else:
                h, w = image.height, image.width
                th, tw = args.resolution, args.resolution
                if h < th or w < tw:
                    raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")
                y1 = np.random.randint(0, h - th + 1, size=(1,)).item()
                x1 = np.random.randint(0, w - tw + 1, size=(1,)).item()
                image = image.crop((x1, y1, x1 + tw, y1 + th))
            image = train_transforms(image)[0]
            self.pixel_values.append(image)

        self.num_instance_images = len(self.instance_images)
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                vision.Resize(size, interpolation=vision.Inter.BILINEAR),
                vision.CenterCrop(size) if center_crop else vision.RandomCrop(size),
                vision.ToTensor(),
                vision.Normalize([0.5], [0.5], is_hwc=False),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = self.pixel_values[index % self.num_instance_images]
        example["instance_images"] = instance_image

        if self.custom_instance_prompts:
            caption = self.custom_instance_prompts[index % self.num_instance_images]
            if caption:
                example["instance_prompt"] = caption
            else:
                example["instance_prompt"] = self.instance_prompt

        else:  # custom prompts were provided, but length does not match size of image dataset
            example["instance_prompt"] = self.instance_prompt

        example["instance_tokens_one"] = tokenize_prompt(self.tokenizer_one, example["instance_prompt"])
        example["instance_tokens_two"] = tokenize_prompt(self.tokenizer_two, example["instance_prompt"])
        example["instance_tokens_three"] = tokenize_prompt(self.tokenizer_three, example["instance_prompt"])

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            class_image = exif_transpose(class_image)

            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)[0]
            example["class_prompt"] = self.class_prompt
            example["class_tokens_one"] = tokenize_prompt(self.tokenizer_one, example["class_prompt"])
            example["class_tokens_two"] = tokenize_prompt(self.tokenizer_two, example["class_prompt"])
            example["class_tokens_three"] = tokenize_prompt(self.tokenizer_three, example["class_prompt"])

        return example


def collate_fn(examples, with_prior_preservation=False):
    pixel_values = [example["instance_images"] for example in examples]
    tokens_one = [example["instance_tokens_one"] for example in examples]
    tokens_two = [example["instance_tokens_two"] for example in examples]
    tokens_three = [example["instance_tokens_three"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        pixel_values += [example["class_images"] for example in examples]
        tokens_one += [example["class_tokens_one"] for example in examples]
        tokens_two += [example["class_tokens_two"] for example in examples]
        tokens_three += [example["class_tokens_three"] for example in examples]

    pixel_values = np.stack(pixel_values).astype(np.float32)
    tokens_one = np.concatenate(tokens_one, axis=0)
    tokens_two = np.concatenate(tokens_two, axis=0)
    tokens_three = np.concatenate(tokens_three, axis=0)

    return pixel_values, tokens_one, tokens_two, tokens_three


class PromptDataset(object):
    """A simple dataset to prepare the prompts to generate class images on multiple Devices."""

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="np",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


def main():
    args = parse_args()
    ms.set_context(
        mode=ms.GRAPH_MODE,
        jit_syntax_level=ms.STRICT,
        jit_config={"jit_level": "O2"},
    )

    # read attr distributed, writer attrs rank/local_rank/world_size:
    #   args.local_rank = mindspore.communication.get_local_rank()
    #   args.world_size = mindspore.communication.get_group_size()
    #   args.rank = mindspore.communication.get_rank()
    init_distributed_device(args)

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

    # Generate class images if prior preservation is enabled.
    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            mindspore_dtype = ms.float32
            if args.prior_generation_precision == "fp32":
                mindspore_dtype = ms.float32
            elif args.prior_generation_precision == "fp16":
                mindspore_dtype = ms.float16
            elif args.prior_generation_precision == "bf16":
                mindspore_dtype = ms.bfloat16
            pipeline = StableDiffusion3Pipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                mindspore_dtype=mindspore_dtype,
                revision=args.revision,
                variant=args.variant,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = GeneratorDataset(
                sample_dataset, column_names=["example"], shard_id=args.rank, num_shards=args.world_size
            ).batch(batch_size=args.sample_batch_size)

            sample_dataloader_iter = sample_dataloader.create_tuple_iterator(output_numpy=True)
            for (example,) in tqdm(
                sample_dataloader_iter,
                desc="Generating class images",
                total=len(sample_dataloader),
                disable=not is_master(args),
            ):
                images = pipeline(example["prompt"].tolist())[0]

                for i, image in enumerate(images):
                    hash_image = insecure_hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            del pipeline
            gc.collect()
            ms.ms_memory_recycle()
            logger.warning(
                "After deleting the pipeline, the memory may not be freed correctly by mindspore. "
                "If you encounter an OOM error, please relaunch this script."
            )

    # Handle the repository creation
    if is_master(args):
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(logging_dir, exist_ok=True)

    # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )
    tokenizer_three = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_3",
        revision=args.revision,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )
    text_encoder_cls_three = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_3"
    )

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(
        args, text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    transformer = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
    )

    # We only train the additional adapter LoRA layers
    def set_params_requires_grad(m: nn.Cell, requires_grad: bool):
        for p in m.get_parameters():
            p.requires_grad = requires_grad

    set_params_requires_grad(transformer, True)
    set_params_requires_grad(vae, False)
    if args.train_text_encoder:
        set_params_requires_grad(text_encoder_one, True)
        set_params_requires_grad(text_encoder_two, True)
        set_params_requires_grad(text_encoder_three, True)
    else:
        set_params_requires_grad(text_encoder_one, False)
        set_params_requires_grad(text_encoder_two, False)
        set_params_requires_grad(text_encoder_three, False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = ms.float32
    if args.mixed_precision == "fp16":
        weight_dtype = ms.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = ms.bfloat16

    vae.to(dtype=ms.float32)
    # TODO: We will update the training methods during mixed precision training to ensure the performance and strategies during the training process.
    if args.mixed_precision and args.mixed_precision != "no":
        transformer.to_float(weight_dtype)
        if not args.train_text_encoder:
            text_encoder_one.to(dtype=weight_dtype)
            text_encoder_two.to(dtype=weight_dtype)
            text_encoder_three.to(dtype=weight_dtype)
        else:
            text_encoder_one.to_float(weight_dtype)
            text_encoder_two.to_float(weight_dtype)
            text_encoder_three.to_float(weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder_one.gradient_checkpointing_enable()
            text_encoder_two.gradient_checkpointing_enable()
            text_encoder_three.gradient_checkpointing_enable()

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, output_dir):
        if is_master(args):
            for model in models:
                if isinstance(model, SD3Transformer2DModel):
                    model.save_pretrained(os.path.join(output_dir, "transformer"))
                elif isinstance(model, (CLIPTextModelWithProjection, T5EncoderModel)):
                    if isinstance(model, CLIPTextModelWithProjection):
                        hidden_size = model.config.hidden_size
                        if hidden_size == 768:
                            model.save_pretrained(os.path.join(output_dir, "text_encoder"))
                        elif hidden_size == 1280:
                            model.save_pretrained(os.path.join(output_dir, "text_encoder_2"))
                    else:
                        model.save_pretrained(os.path.join(output_dir, "text_encoder_3"))
                else:
                    raise ValueError(f"Wrong model supplied: {type(model)=}.")

    def load_model_hook(models, input_dir):
        for _ in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()

            # load diffusers style into model
            if isinstance(model, SD3Transformer2DModel):
                load_model = SD3Transformer2DModel.from_pretrained(input_dir, subfolder="transformer")
                model.register_to_config(**load_model.config)

                ms.load_param_into_net(model, load_model.parameters_dict())
            elif isinstance(model, (CLIPTextModelWithProjection, T5EncoderModel)):
                try:
                    load_model = CLIPTextModelWithProjection.from_pretrained(input_dir, subfolder="text_encoder")
                    model(**load_model.config)
                    ms.load_param_into_net(model, load_model.parameters_dict())
                except Exception:
                    try:
                        load_model = CLIPTextModelWithProjection.from_pretrained(input_dir, subfolder="text_encoder_2")
                        model(**load_model.config)
                        ms.load_param_into_net(model, load_model.parameters_dict())
                    except Exception:
                        try:
                            load_model = T5EncoderModel.from_pretrained(input_dir, subfolder="text_encoder_3")
                            model(**load_model.config)
                            ms.load_param_into_net(model, load_model.parameters_dict())
                        except Exception:
                            raise ValueError(f"Couldn't load the model of type: ({type(model)}).")
            else:
                raise ValueError(f"Unsupported model found: {type(model)=}")

            del load_model

    # Define models to load or save for load_model_hook() and save_model_hook()
    models = [transformer]
    if args.train_text_encoder:
        models.extend([text_encoder_one, text_encoder_two, text_encoder_three])

    # Dataset and DataLoaders creation:
    train_dataset = DreamBoothDataset(
        args,
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        tokenizer_one=tokenizer_one,
        tokenizer_two=tokenizer_two,
        tokenizer_three=tokenizer_three,
        class_prompt=args.class_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_num=args.num_class_images,
        size=args.resolution,
        repeats=args.repeats,
        center_crop=args.center_crop,
    )

    train_dataloader = GeneratorDataset(
        train_dataset,
        column_names=["example"],
        shuffle=True,
        shard_id=args.rank,
        num_shards=args.world_size,
        num_parallel_workers=args.dataloader_num_workers,
    ).batch(
        batch_size=args.train_batch_size,
        per_batch_map=lambda examples, batch_info: collate_fn(examples, args.with_prior_preservation),
        input_columns=["example"],
        output_columns=["c1", "c2", "c3", "c4"],  # pixel_values, tokens_one, tokens_two, tokens_three
        num_parallel_workers=args.dataloader_num_workers,
    )

    # If custom instance prompts are NOT provided (i.e. the instance prompt is used for all images),
    # pack the statically computed variables appropriately here. This is so that we don't
    # have to pass them to the dataloader.

    if not train_dataset.custom_instance_prompts:
        tokens_one = ms.Tensor.from_numpy(tokenize_prompt(tokenizer_one, args.instance_prompt))
        tokens_two = ms.Tensor.from_numpy(tokenize_prompt(tokenizer_two, args.instance_prompt))
        tokens_three = ms.Tensor.from_numpy(tokenize_prompt(tokenizer_three, args.instance_prompt))

        if args.with_prior_preservation:
            class_tokens_one = ms.Tensor.from_numpy(tokenize_prompt(tokenizer_one, args.class_prompt))
            class_tokens_two = ms.Tensor.from_numpy(tokenize_prompt(tokenizer_two, args.class_prompt))
            class_tokens_three = ms.Tensor.from_numpy(tokenize_prompt(tokenizer_three, args.class_prompt))

            tokens_one = ops.cat([tokens_one, class_tokens_one], axis=0)
            tokens_two = ops.cat([tokens_two, class_tokens_two], axis=0)
            tokens_three = ops.cat([tokens_three, class_tokens_three], axis=0)
    else:
        tokens_one, tokens_two, tokens_three = None, None, None

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
        power=args.lr_power,
    )

    # Optimization parameters
    if args.train_text_encoder and args.text_encoder_lr:
        # different learning rate for text encoder and unet
        transformer_parameters_with_lr = {"params": transformer.trainable_params(), "lr": lr_scheduler}
        text_encoder_lr_scheduler = get_scheduler(
            args.lr_scheduler,
            args.text_encoder_lr,
            num_warmup_steps=args.lr_warmup_steps,
            num_training_steps=args.max_train_steps,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )

        text_parameters_one_with_lr = {
            "params": text_encoder_one.trainable_params(),
            "weight_decay": args.adam_weight_decay_text_encoder,
            "lr": text_encoder_lr_scheduler,
        }
        text_parameters_two_with_lr = {
            "params": text_encoder_two.trainable_params(),
            "weight_decay": args.adam_weight_decay_text_encoder,
            "lr": text_encoder_lr_scheduler,
        }
        text_parameters_three_with_lr = {
            "params": text_encoder_three.trainable_params(),
            "weight_decay": args.adam_weight_decay_text_encoder,
            "lr": text_encoder_lr_scheduler,
        }
        params_to_optimize = [
            transformer_parameters_with_lr,
            text_parameters_one_with_lr,
            text_parameters_two_with_lr,
            text_parameters_three_with_lr,
        ]
    else:
        params_to_optimize = (
            itertools.chain(
                transformer.trainable_params(),
                text_encoder_one.trainable_params(),
                text_encoder_two.trainable_params(),
                text_encoder_three.trainable_params(),
            )
            if args.train_text_encoder
            else transformer.trainable_params()
        )

    # Optimizer creation
    optimizer = nn.AdamWeightDecay(
        params_to_optimize,
        learning_rate=lr_scheduler,
        beta1=args.adam_beta1,
        beta2=args.adam_beta2,
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
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
        with open(logging_dir / "hparams.yml", "w") as f:
            yaml.dump(vars(args), f, indent=4)
    trackers = dict()
    for tracker_name in args.report_to.split(","):
        if tracker_name == "tensorboard":
            from tensorboardX import SummaryWriter

            trackers[tracker_name] = SummaryWriter(str(logging_dir), write_to_disk=is_master(args))
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
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
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
            load_model_hook(models, os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    # create train_step for training
    train_step = TrainStepForSD3DB(
        vae=vae,
        text_encoder_one=text_encoder_one,
        text_encoder_two=text_encoder_two,
        text_encoder_three=text_encoder_three,
        transformer=transformer,
        optimizer=optimizer,
        noise_scheduler=noise_scheduler_copy,
        weight_dtype=weight_dtype,
        length_of_dataloader=len(train_dataloader),
        args=args,
        tokens_one=tokens_one,
        tokens_two=tokens_two,
        tokens_three=tokens_three,
        custom_instance_prompts=train_dataset.custom_instance_prompts,
    ).set_train()

    if args.enable_mindspore_data_sink:
        sink_process = ms.data_sink(train_step, train_dataloader)
    else:
        sink_process = None

    # create pipeline for validation
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        text_encoder_3=text_encoder_three,
        tokenizer_3=tokenizer_three,
        transformer=transformer,
        revision=args.revision,
        variant=args.variant,
        mindspore_dtype=weight_dtype,
    )

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not is_master(args),
    )

    train_dataloader_iter = train_dataloader.create_tuple_iterator(num_epochs=args.num_train_epochs - first_epoch)
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.set_train(True)
        if args.train_text_encoder:
            text_encoder_one.set_train(True)
            text_encoder_two.set_train(True)
            text_encoder_three.set_train(True)

        for step, batch in (
            ((_, None) for _ in range(len(train_dataloader)))  # dummy iterator
            if args.enable_mindspore_data_sink
            else enumerate(train_dataloader_iter)
        ):
            if args.enable_mindspore_data_sink:
                loss, model_pred = sink_process()
            else:
                loss, model_pred = train_step(*batch)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if train_step.sync_gradients:
                progress_bar.update(1)
                global_step += 1

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
                        save_model_hook(models, save_path)
                        logger.info(f"Saved state to {save_path}")

            log_lr = optimizer.get_lr()
            log_lr = log_lr[0] if isinstance(log_lr, tuple) else log_lr  # grouped lr scenario
            logs = {"loss": loss.numpy().item(), "lr": log_lr.numpy().item()}
            progress_bar.set_postfix(**logs)
            for tracker_name, tracker in trackers.items():
                if tracker_name == "tensorboard":
                    tracker.add_scalars("train", logs, global_step)

            if global_step >= args.max_train_steps:
                break

        if args.validation_prompt is not None and (epoch + 1) % args.validation_epochs == 0:
            pipeline_args = {"prompt": args.validation_prompt}
            log_validation(
                pipeline,
                args,
                trackers,
                logging_dir,
                pipeline_args,
                (epoch + 1),
            )

    # Save the models
    if is_master(args):
        pipeline.save_pretrained(args.output_dir)

    # Final inference
    if args.validation_prompt and args.num_validation_images > 0:
        pipeline_args = {"prompt": args.validation_prompt, "num_inference_steps": 25}
        log_validation(
            pipeline,
            args,
            trackers,
            logging_dir,
            pipeline_args,
            args.num_train_epochs,
            is_final_validation=True,
        )

    # End of training
    for tracker_name, tracker in trackers.items():
        if tracker_name == "tensorboard":
            tracker.close()


def compute_weighting_mse_loss(weighting, pred, target):
    """
    When argument with_prior_preservation is True in DreamBooth training, weighting has different batch_size
    with pred/target which causes errors, therefore we broadcast them to proper shape before mul
    """
    repeats = weighting.shape[0] // pred.shape[0]
    target_ndim = target.ndim
    square_loss = ((pred.float() - target.float()) ** 2).tile((repeats,) + (1,) * (target_ndim - 1))

    weighting_mse_loss = ops.mean(
        (weighting * square_loss).reshape(target.shape[0], -1),
        1,
    )
    weighting_mse_loss = weighting_mse_loss.mean()

    return weighting_mse_loss


class TrainStepForSD3DB(TrainStep):
    def __init__(
        self,
        vae: nn.Cell,
        text_encoder_one: nn.Cell,
        text_encoder_two: nn.Cell,
        text_encoder_three: nn.Cell,
        transformer: nn.Cell,
        optimizer: nn.Optimizer,
        noise_scheduler,
        weight_dtype,
        length_of_dataloader,
        args,
        tokens_one,
        tokens_two,
        tokens_three,
        custom_instance_prompts,
    ):
        super().__init__(
            transformer,
            optimizer,
            StaticLossScaler(4096),
            args.max_grad_norm,
            args.gradient_accumulation_steps,
            gradient_accumulation_kwargs=dict(length_of_dataloader=length_of_dataloader),
        )
        self.transformer = transformer
        self.vae = vae
        self.vae_dtype = vae.dtype
        self.vae_scaling_factor = vae.config.scaling_factor
        self.text_encoder_one = text_encoder_one
        self.text_encoder_two = text_encoder_two
        self.text_encoder_three = text_encoder_three
        self.text_encoder_dtype = text_encoder_one.dtype
        self.noise_scheduler = noise_scheduler
        self.noise_scheduler_num_train_timesteps = noise_scheduler.config.num_train_timesteps
        self.weight_dtype = weight_dtype
        self.args = AttrJitWrapper(**vars(args))
        self.tokens_one = tokens_one
        self.tokens_two = tokens_two
        self.tokens_three = tokens_three
        self.custom_instance_prompts = custom_instance_prompts

        # If no type of tuning is done on the text_encoder and custom instance prompts are NOT
        # provided (i.e. the --instance_prompt is used for all images), we encode the instance prompt once to avoid
        # the redundant encoding.
        if not custom_instance_prompts and not args.train_text_encoder:
            prompt_embeds, pooled_prompt_embeds = self.encode_prompt(tokens_one, tokens_two, tokens_three)
            self.prompt_embeds = prompt_embeds
            self.pooled_prompt_embeds = pooled_prompt_embeds
        else:
            self.prompt_embeds = None
            self.pooled_prompt_embeds = None

    def _encode_prompt_with_t5(self, text_input_ids, num_images_per_prompt=1):
        batch_size = text_input_ids.shape[0]
        prompt_embeds = self.text_encoder_three(text_input_ids)[0]

        dtype = self.text_encoder_dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.tile((1, num_images_per_prompt, 1))
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    def encode_prompt(
        self,
        text_input_ids_one,
        text_input_ids_two,
        text_input_ids_three,
        num_images_per_prompt: int = 1,
    ):
        # text encoder one
        prompt_embeds_one = self.text_encoder_one(text_input_ids_one, output_hidden_states=True)
        pooled_prompt_embeds_one = prompt_embeds_one[0]
        prompt_embeds_one = prompt_embeds_one[-1][-2]
        prompt_embeds_one = prompt_embeds_one.to(dtype=self.text_encoder_dtype)
        prompt_embeds_one = prompt_embeds_one.tile((num_images_per_prompt, 1, 1))

        # text encoder two
        prompt_embeds_two = self.text_encoder_two(text_input_ids_two, output_hidden_states=True)
        pooled_prompt_embeds_two = prompt_embeds_two[0]
        prompt_embeds_two = prompt_embeds_two[-1][-2]
        prompt_embeds_two = prompt_embeds_two.to(dtype=self.text_encoder_dtype)
        prompt_embeds_two = prompt_embeds_two.tile((num_images_per_prompt, 1, 1))

        # CLIPs
        clip_prompt_embeds = ops.cat([prompt_embeds_one, prompt_embeds_two], axis=-1)
        pooled_prompt_embeds = ops.cat([pooled_prompt_embeds_one, pooled_prompt_embeds_two], axis=-1)

        # T5 (text encoder three)
        t5_prompt_embed = self._encode_prompt_with_t5(
            text_input_ids_three,
            num_images_per_prompt=num_images_per_prompt,
        )

        # integreted
        clip_prompt_embeds = ops.Pad(
            paddings=((0, 0), (0, 0), (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1]))
        )(clip_prompt_embeds)
        prompt_embeds = ops.cat([clip_prompt_embeds, t5_prompt_embed], axis=-2)

        return prompt_embeds, pooled_prompt_embeds

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

    def forward(self, pixel_values, tokens_one, tokens_two, tokens_three):
        pixel_values = pixel_values.to(dtype=self.vae_dtype)

        # encode batch prompts when custom prompts are provided for each image
        if not self.custom_instance_prompts:
            # use pre-computed embeddings or tokens.
            if not self.args.train_text_encoder:
                prompt_embeds, pooled_prompt_embeds = self.prompt_embeds, self.pooled_prompt_embeds
            else:
                prompt_embeds, pooled_prompt_embeds = self.encode_prompt(
                    self.tokens_one, self.tokens_two, self.tokens_three
                )
        else:
            # encode batch prompts when custom prompts are provided for each image
            prompt_embeds, pooled_prompt_embeds = self.encode_prompt(tokens_one, tokens_two, tokens_three)

        # Convert images to latent space
        model_input = self.vae.diag_gauss_dist.sample(self.vae.encode(pixel_values)[0])
        model_input = model_input * self.vae_scaling_factor
        model_input = model_input.to(dtype=self.weight_dtype)

        # Sample noise that we'll add to the latents
        noise = ops.randn_like(model_input, dtype=model_input.dtype)
        bsz = model_input.shape[0]

        # Sample a random timestep for each image
        # for weighting schemes where we sample timesteps non-uniformly
        if self.args.weighting_scheme == "logit_normal":
            # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
            u = ops.normal(mean=self.args.logit_mean, stddev=self.args.logit_std, shape=(bsz,))
            u = ops.sigmoid(u)
        elif self.args.weighting_scheme == "mode":
            u = ops.rand(bsz)
            u = 1 - u - self.args.mode_scale * (ops.cos(ms.numpy.pi * u / 2) ** 2 - 1 + u)
        else:
            u = ops.rand(bsz)

        indices = (u * self.noise_scheduler_num_train_timesteps).long()
        timesteps = self.noise_scheduler.timesteps[indices]

        # Add noise according to flow matching.
        sigmas = self.get_sigmas(indices, n_dim=model_input.ndim, dtype=model_input.dtype)
        noisy_model_input = sigmas * noise + (1.0 - sigmas) * model_input

        # Predict the noise residual
        model_pred = self.transformer(
            hidden_states=noisy_model_input,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False,
        )[0]

        # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
        # Preconditioning of the model outputs.
        model_pred = model_pred * (-sigmas) + noisy_model_input

        # TODO (kashif, sayakpaul): weighting sceme needs to be experimented with :)
        # these weighting schemes use a uniform timestep sampling and instead post-weight the loss
        if self.args.weighting_scheme == "sigma_sqrt":
            weighting = (sigmas**-2.0).float()
        elif self.args.weighting_scheme == "cosmap":
            bot = 1 - 2 * sigmas + 2 * sigmas**2
            weighting = 2 / (ms.numpy.pi * bot)
        else:
            weighting = ops.ones_like(sigmas)

        # simplified flow matching aka 0-rectified flow matching loss
        # target = model_input - noise
        target = model_input

        if self.args.with_prior_preservation:
            # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
            model_pred, model_pred_prior = ops.chunk(model_pred, 2, axis=0)
            target, target_prior = ops.chunk(target, 2, axis=0)

            # Compute prior loss
            prior_loss = compute_weighting_mse_loss(weighting, model_pred_prior, target_prior)
        else:
            prior_loss = None

        # Compute regular loss.
        loss = compute_weighting_mse_loss(weighting, model_pred, target)

        if self.args.with_prior_preservation:
            # Add the prior loss to the instance loss.
            loss = loss + self.args.prior_loss_weight * prior_loss

        loss = self.scale_loss(loss)
        return loss, model_pred


if __name__ == "__main__":
    main()
