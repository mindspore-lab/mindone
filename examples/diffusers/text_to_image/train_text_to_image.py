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
# limitations under the License.

import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
import yaml
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import CLIPTokenizer

import mindspore as ms
from mindspore import context, nn, ops
from mindspore.amp import DynamicLossScaler, LossScaler, StaticLossScaler, all_finite
from mindspore.dataset import GeneratorDataset, transforms, vision

from mindone.diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from mindone.diffusers.optimization import get_scheduler
from mindone.diffusers.training_utils import compute_snr, init_distributed_device, is_master, set_seed
from mindone.diffusers.utils import deprecate
from mindone.transformers import CLIPTextModel

logger = logging.getLogger(__name__)

DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}


def log_validation(pipeline, args, trackers, logging_dir, epoch):
    logger.info("Running validation... ")
    pipeline.unet.set_train(False)

    if args.seed is None:
        generator = None
    else:
        generator = np.random.Generator(np.random.PCG64(seed=args.seed))

    images = []
    for i in range(len(args.validation_prompts)):
        image = pipeline(args.validation_prompts[i], num_inference_steps=20, generator=generator)[0][0]
        images.append(image)

    if is_master(args):
        validation_logging_dir = os.path.join(logging_dir, "validation", f"epoch{epoch}")
        os.makedirs(validation_logging_dir, exist_ok=True)
        for idx, img in enumerate(images):
            img.save(os.path.join(validation_logging_dir, f"{idx:04d}.jpg"))

    for tracker_name, tracker_writer in trackers.items():
        if tracker_name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker_writer.add_images("validation", np_images, epoch, dataformats="NHWC")
        else:
            logger.warning(f"image logging not implemented for {tracker_name}")

    logger.info("Validation done.")

    return images


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
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
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
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
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
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
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
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
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
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
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",  # noqa: E501
    )
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
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
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
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )

    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision
    else:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )

    # Limitations for NOW.
    def error_template(feature, flag):
        return f"{feature} is not yet supported, please do not set --{flag}"

    assert args.gradient_accumulation_steps == 1, error_template("Gradient Accumulation", "gradient_accumulation_steps")
    assert args.use_ema is False, error_template("Exponential Moving Average", "use_ema")
    assert args.allow_tf32 is False, error_template("TF32 Data Type", "allow_tf32")
    assert args.use_8bit_adam is False, error_template("AdamW8bit", "use_8bit_adam")
    assert args.enable_xformers_memory_efficient_attention is False, error_template(
        "Memory Efficient Attention from 'xformers'", "enable_xformers_memory_efficient_attention"
    )
    if args.push_to_hub is True:
        raise ValueError(
            "You cannot use --push_to_hub due to a security risk of uploading your data to huggingface-hub. "
            "If you know what you are doing, just delete this line and try again."
        )

    return args


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

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    # Get the target for loss depending on the prediction type
    if args.prediction_type is not None:
        # set prediction_type of scheduler if defined
        noise_scheduler.register_to_config(prediction_type=args.prediction_type)
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
    )
    # set sample_size of unet
    unet.register_to_config(sample_size=args.resolution // (2 ** (len(vae.config.block_out_channels) - 1)))

    # Freeze vae and text_encoder and set unet to trainable
    def freeze_params(m: nn.Cell):
        for p in m.get_parameters():
            p.requires_grad = False

    freeze_params(vae)
    freeze_params(text_encoder)
    unet.set_train(True)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to
    # half-precision as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = ms.float32
    if args.mixed_precision == "fp16":
        weight_dtype = ms.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = ms.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(dtype=weight_dtype)
    vae.to(dtype=weight_dtype)

    # TODO: support EMA, xformers_memory_efficient_attention, TF32, AdamW8bit

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).
    from datasets import disable_caching

    if args.cache_dir is None:
        disable_caching()

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        if args.dataset_name == "webdataset" or args.dataset_name == "imagefolder":
            # Packaged dataset
            dataset = load_dataset(
                args.dataset_name,
                data_dir=args.train_data_dir,
                cache_dir=args.cache_dir,
                # setting streaming=True when using webdataset gives DatasetIter which has different process apis
            )
        else:
            # Downloading and loading a dataset from the hub.
            dataset = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                cache_dir=args.cache_dir,
                data_dir=args.train_data_dir,
            )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
    if args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="np"
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            vision.Resize(args.resolution, interpolation=vision.Inter.BILINEAR),
            vision.CenterCrop(args.resolution) if args.center_crop else vision.RandomCrop(args.resolution),
            vision.RandomHorizontalFlip() if args.random_flip else lambda x: x,
            vision.ToTensor(),
            vision.Normalize([0.5], [0.5], is_hwc=False),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image)[0] for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    if args.max_train_samples is not None:
        dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
    # Set the training transforms
    train_dataset = dataset["train"].with_transform(preprocess_train)

    class UnravelDataset:
        def __init__(self, data):
            self.data = data

        def __getitem__(self, idx):
            idx = idx.item() if isinstance(idx, np.integer) else idx
            example = self.data[idx]
            pixel_values = example["pixel_values"]
            input_ids = example["input_ids"]
            return np.array(pixel_values, dtype=np.float32), np.array(input_ids, dtype=np.int32)

        def __len__(self):
            return len(self.data)

    # DataLoaders creation:
    train_dataloader = GeneratorDataset(
        UnravelDataset(train_dataset),
        column_names=["pixel_values", "input_ids"],
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
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    optimizer = nn.AdamWeightDecay(
        unet.trainable_params(),
        learning_rate=lr_scheduler,
        beta1=args.adam_beta1,
        beta2=args.adam_beta2,
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Prepare everything with our `accelerator`.
    unet.to_float(weight_dtype)

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

    train_step = TrainStep(
        vae=vae,
        text_encoder=text_encoder,
        unet=unet,
        optimizer=optimizer,
        scaler=StaticLossScaler(65536),
        noise_scheduler=noise_scheduler,
        args=args,
    ).set_train()

    if args.enable_mindspore_data_sink:
        sink_process = ms.data_sink(train_step, train_dataloader)
    else:
        sink_process = None

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        mindspore_dtype=weight_dtype,
    )
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
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1].split(".")[0]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            if is_master(args):
                logger.info(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            if is_master(args):
                logger.info(f"Resuming from checkpoint {path}")
            state_dict = ms.load_checkpoint(os.path.join(args.output_dir, path))
            ms.load_param_into_net(unet, state_dict)
            global_step = int(path.split("-")[1].split(".")[0])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    # run inference
    if args.validation_prompts is not None:
        log_validation(pipeline, args, trackers, logging_dir, first_epoch)

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not is_master(args),
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.set_train(True)
        for step, batch in (
            ((_, None) for _ in range(len(train_dataloader)))  # dummy iterator
            if args.enable_mindspore_data_sink
            else enumerate(train_dataloader.create_tuple_iterator())
        ):
            if args.enable_mindspore_data_sink:
                loss = sink_process()
            else:
                loss = train_step(*batch)

            progress_bar.update(1)
            global_step += 1
            for tracker_name, tracker in trackers.items():
                if tracker_name == "tensorboard":
                    tracker.add_scalar("train/loss", loss.numpy().item(), global_step)

            if global_step % args.checkpointing_steps == 0:
                if is_master(args):
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0]))

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
                    ms.save_checkpoint(unet, save_path)
                    logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.numpy().item(), "lr": optimizer.get_lr().numpy().item()}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if args.validation_prompts is not None and (epoch + 1) % args.validation_epochs == 0:
            log_validation(pipeline, args, trackers, logging_dir, epoch + 1)

    # Serialize pipeline.
    if is_master(args):
        pipeline.save_pretrained(args.output_dir)

    # Run a final round of inference.
    if args.validation_prompts is not None:
        log_validation(pipeline, args, trackers, logging_dir, args.num_train_epochs)
    for tracker_name, tracker in trackers.items():
        if tracker_name == "tensorboard":
            tracker.close()


class TrainStep(nn.Cell):
    def __init__(
        self,
        vae: nn.Cell,
        text_encoder: nn.Cell,
        unet: nn.Cell,
        optimizer: nn.Optimizer,
        scaler: LossScaler,
        noise_scheduler,
        args,
    ):
        super().__init__()
        self.unet = unet.set_grad()
        self.optimizer = optimizer
        self.weights = optimizer.parameters
        self.scaler = scaler
        if isinstance(self.scaler, StaticLossScaler):
            self.drop_overflow = False
        elif isinstance(self.scaler, DynamicLossScaler):
            self.drop_overflow = True
        else:
            raise NotImplementedError(f"Unsupported scaler: {type(self.scaler)}")
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode == context.ParallelMode.STAND_ALONE:
            self.grad_reducer = nn.Identity()
        elif self.parallel_mode in (context.ParallelMode.DATA_PARALLEL, context.ParallelMode.HYBRID_PARALLEL):
            self.grad_reducer = nn.DistributedGradReducer(self.weights)
        else:
            raise NotImplementedError(f"When creating reducer, Got Unsupported parallel mode: {self.parallel_mode}")
        if isinstance(unet, nn.Cell) and unet.jit_config_dict:
            self._jit_config_dict = unet.jit_config_dict
        self.clip_grad = args.max_grad_norm is not None
        self.clip_value = args.max_grad_norm

        @ms.jit_class
        class ArgsJitWrapper:
            def __init__(self, **kwargs):
                for name in kwargs:
                    setattr(self, name, kwargs[name])

        self.args = ArgsJitWrapper(**vars(args))
        self.vae = vae
        self.vae_scaling_factor = self.vae.config.scaling_factor
        self.weight_dtype = self.vae.dtype
        self.text_encoder = text_encoder
        self.noise_scheduler = noise_scheduler
        self.noise_scheduler_num_train_timesteps = noise_scheduler.config.num_train_timesteps
        self.noise_scheduler_prediction_type = noise_scheduler.config.prediction_type

        self.forward_and_backward = ops.value_and_grad(self.forward, None, weights=self.weights, has_aux=True)

    def forward(self, pixel_values, input_ids):
        # Convert images to latent space
        latents = self.vae.diag_gauss_dist.sample(self.vae.encode(pixel_values.to(self.weight_dtype))[0])
        latents = latents * self.vae_scaling_factor

        # Sample noise that we'll add to the latents
        noise = ops.randn_like(latents, dtype=latents.dtype)
        if self.args.noise_offset:
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise_offset = self.args.noise_offset * ops.randn((latents.shape[0], latents.shape[1], 1, 1))
            noise += noise_offset.to(noise.dtype)
        if self.args.input_perturbation:
            noise_perturbation = self.args.input_perturbation * ops.randn_like(noise, dtype=noise.dtype)
            new_noise = noise + noise_perturbation.to(noise.dtype)
        else:
            new_noise = None  # graph need this placeholder
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = ops.randint(0, self.noise_scheduler_num_train_timesteps, (bsz,))
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        if self.args.input_perturbation:
            noisy_latents = self.noise_scheduler.add_noise(latents, new_noise, timesteps)
        else:
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        # TODO: method of scheduler should not change the dtype of input.
        #  Remove the casting after cuiyushi confirm that.
        noisy_latents = noisy_latents.to(latents.dtype)

        # Get the text embedding for conditioning
        encoder_hidden_states = self.text_encoder(input_ids)[0]

        if self.noise_scheduler_prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler_prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler_prediction_type}")

        # Predict the noise residual and compute loss
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

        if self.args.snr_gamma is None:
            loss = ops.mse_loss(model_pred.float(), target.float(), reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = compute_snr(self.noise_scheduler, timesteps)
            mse_loss_weights = ops.stack([snr, self.args.snr_gamma * ops.ones_like(timesteps)], axis=1).min(axis=1)[0]
            if self.noise_scheduler_prediction_type == "epsilon":
                mse_loss_weights = mse_loss_weights / snr
            elif self.noise_scheduler_prediction_type == "v_prediction":
                mse_loss_weights = mse_loss_weights / (snr + 1)

            loss = ops.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(axis=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        loss = self.scaler.scale(loss)
        return loss, model_pred

    def update(self, loss, grads):
        if self.clip_grad:
            loss = ops.depend(loss, self.optimizer(ops.clip_by_global_norm(grads, clip_norm=self.clip_value)))
        else:
            loss = ops.depend(loss, self.optimizer(grads))
        return loss

    def construct(self, *inputs):
        (loss, model_pred), grads = self.forward_and_backward(*inputs)
        grads = self.grad_reducer(grads)
        loss = self.scaler.unscale(loss)
        grads = self.scaler.unscale(grads)

        if self.drop_overflow:
            status = all_finite(grads)
            if status:
                loss = self.update(loss, grads)
            loss = ops.depend(loss, self.scaler.adjust(status))
        else:
            loss = self.update(loss, grads)

        return loss


if __name__ == "__main__":
    main()
