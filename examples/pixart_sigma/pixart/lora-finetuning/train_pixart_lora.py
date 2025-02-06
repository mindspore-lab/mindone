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
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import PretrainedConfig, T5Tokenizer

import mindspore as ms
from mindspore import nn, ops
from mindspore.amp import StaticLossScaler
from mindspore.dataset import GeneratorDataset, transforms, vision

from mindone.diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    StableDiffusionPipeline,
    Transformer2DModel,
)
from mindone.diffusers._peft import LoraConfig, get_peft_model
from mindone.diffusers._peft.tuners.tuners_utils import BaseTunerLayer
from mindone.diffusers._peft.utils import get_peft_model_state_dict
from mindone.diffusers.optimization import get_scheduler
from mindone.diffusers.training_utils import (
    AttrJitWrapper,
    TrainStep,
    cast_training_params,
    compute_snr,
    init_distributed_device,
    is_master,
    set_seed,
)
from mindone.diffusers.utils import convert_state_dict_to_diffusers

logger = logging.getLogger(__name__)


def log_validation(pipeline, args, trackers, logging_dir, epoch):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt: "
        f" {args.validation_prompt}."
    )
    pipeline.transformer.set_train(False)

    # run inference
    generator = np.random.Generator(np.random.PCG64(seed=args.seed))
    images = []
    for _ in range(args.num_validation_images):
        images.append(pipeline(args.validation_prompt, num_inference_steps=30, generator=generator)[0][0])

    if is_master(args):
        validation_logging_dir = os.path.join(logging_dir, "validation", f"epoch{epoch}")
        os.makedirs(validation_logging_dir, exist_ok=True)
        for idx, img in enumerate(images):
            img.save(os.path.join(validation_logging_dir, f"{idx: 04d}.jpg"))

    for tracker_name, tracker_writer in trackers.items():
        if tracker_name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker_writer.add_images("validation", np_images, epoch, dataformats="NHWC")
        else:
            logger.warning(f"image logging not implemented for {tracker_name}")

    logger.info("Validation done.")


def unwrap_model(model):
    for name, param in model.parameters_and_names():
        param.name = name
    return model


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from mindone.transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        raise NotImplementedError("RobertaSeriesModelWithTransformation is not yet implemented.")
    elif model_class == "T5EncoderModel":
        from mindone.transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args():
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
        default="en_text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--validation_prompt", type=str, default=None, help="A prompt that is sampled during training for inference."
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
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
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
        "--output_dir",
        type=str,
        default="pixart-model-finetuned-lora",
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
        "--learning_rate",
        type=float,
        default=1e-6,
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
        "--dataloader_num_workers",
        type=int,
        default=1,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
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
    # ----Diffusion Training Arguments----
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help=(
            "The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`."
            "If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen."
        ),
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
    parser.add_argument("--jit_level", default="O1", choices=["O0", "O1"], help="Jit Level")
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
        help="Max number of checkpoints to store.",
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
        "--lorarank",
        type=int,
        default=16,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--micro_conditions",
        default=False,
        action="store_true",
        help="Only set to true for `PixArt-alpha/PixArt-XL-2-1024-MS`",
    )
    parser.add_argument(
        "--max_token_length",
        type=int,
        default=300,
        help="max length for the tokenized text embedding.",
    )

    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args


DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
    "svjack/pokemon-blip-captions-en-zh": ("image", "en_text"),
}


def main():
    args = parse_args()
    ms.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": args.jit_level}, jit_syntax_level=ms.STRICT)
    init_distributed_device(args)

    logging_dir = Path(args.output_dir, args.logging_dir)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if args.seed is not None:
        set_seed(args.seed)

    if is_master(args):
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(logging_dir, exist_ok=True)

    max_length = args.max_token_length

    # For mixed precision training we cast all non-trainable weights
    # (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = ms.float32
    if args.mixed_precision == "fp16":
        weight_dtype = ms.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = ms.bfloat16

    # Load scheduler, tokenizer and models.

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = T5Tokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )

    def freeze_params(m: nn.Cell):
        for p in m.get_parameters():
            p.requires_grad = False

    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    # text_encoder = T5EncoderModel.from_pretrained(
    #     args.pretrained_model_name_or_path,
    #     subfolder="text_encoder",
    #     revision=args.revision
    # )
    freeze_params(text_encoder)
    text_encoder.to(dtype=weight_dtype)

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    freeze_params(vae)
    vae.to(dtype=weight_dtype)
    # vae.to(dtype = ms.float32)

    transformer = Transformer2DModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="transformer")
    freeze_params(transformer)
    transformer.to(dtype=weight_dtype)
    transformer_config = transformer.config
    lora_config = LoraConfig(
        r=args.lorarank,
        init_lora_weights="gaussian",
        lora_alpha=args.lorarank,
        target_modules=[
            "to_k",
            "to_q",
            "to_v",
            "to_out.0",
            "proj_in",
            "proj_out",
            "ff.net.0.proj",
            "ff.net.2",
            "proj",
            "linear",
            "linear_1",
            "linear_2",
        ],
    )

    transformer = get_peft_model(transformer, lora_config)
    if args.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(transformer, dtype=ms.float32)

    transformer.print_trainable_parameters()

    def save_model_hook(models, output_dir):
        if is_master(args):
            transformer.save_pretrained(output_dir)

    def load_model_hook(models, input_dir):
        pass

    if args.enable_xformers_memory_efficient_attention:
        transformer.enable_xformers_memory_efficient_attention()

    lora_layers = list(filter(lambda p: p.requires_grad, transformer.get_parameters()))

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

    def tokenize_captions(examples, is_train=True, proportion_empty_prompts=0.0, max_length=120):
        captions = []
        for caption in examples[caption_column]:
            if random.random() < proportion_empty_prompts:
                captions.append("")
            elif isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column '{caption_column}' should contain either strings or lists of strings."
                )
        inputs = tokenizer(captions, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
        return inputs.input_ids, inputs.attention_mask

    train_transforms = transforms.Compose(
        [
            vision.Resize(args.resolution, interpolation=vision.Inter.BILINEAR),
            vision.CenterCrop(args.resolution) if args.center_crop else vision.RandomCrop(args.resolution),
            vision.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            vision.ToTensor(),
            vision.Normalize([0.5], [0.5], is_hwc=False),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image)[0] for image in images]
        examples["input_ids"], examples["prompt_attention_mask"] = tokenize_captions(
            examples, proportion_empty_prompts=args.proportion_empty_prompts, max_length=max_length
        )
        return examples

    # if is_master(args):
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
            prompt_attention_mask = example["prompt_attention_mask"]
            return (
                np.array(pixel_values, dtype=np.float32),
                np.array(input_ids, dtype=np.int32),
                np.array(prompt_attention_mask, dtype=np.float32),
            )

        def __len__(self):
            return len(self.data)

    train_dataloader = GeneratorDataset(
        UnravelDataset(train_dataset),
        column_names=["pixel_values", "input_ids", "prompt_attention_mask"],
        shuffle=True,
        shard_id=args.rank,
        num_shards=args.world_size,
        num_parallel_workers=args.dataloader_num_workers,
    ).batch(
        batch_size=args.train_batch_size,
        drop_remainder=True,
        num_parallel_workers=args.dataloader_num_workers,
    )

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
        # num_cycles=args.lr_num_cycles,
        # power=args.lr_power,
    )

    optimizer = nn.AdamWeightDecay(
        lora_layers,
        learning_rate=lr_scheduler,
        beta1=args.adam_beta1,
        beta2=args.adam_beta2,
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Replace accelerator.prepare
    # for peft_model in models:
    for _, module in transformer.cells_and_names():
        if isinstance(module, BaseTunerLayer):
            for layer_name in module.adapter_layer_names:
                module_dict = getattr(module, layer_name)
                for key, layer in module_dict.items():
                    if key in module.active_adapters and isinstance(layer, nn.Cell):
                        layer.to_float(weight_dtype)

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

    train_step = TrainStep_PixartSigma(
        vae=vae,
        text_encoder=text_encoder,
        transformer=transformer,
        optimizer=optimizer,
        noise_scheduler=noise_scheduler,
        weight_dtype=weight_dtype,
        length_of_dataloader=len(train_dataloader),
        args=args,
        transformer_config=transformer_config,
    ).set_train()

    if args.enable_mindspore_data_sink:
        sink_process = ms.data_sink(train_step, train_dataloader)
    else:
        sink_process = None

    # # create pipeline
    pipeline_args = {}
    if vae is not None:
        pipeline_args["vae"] = vae
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=transformer,
        revision=args.revision,
        variant=args.variant,
        mindspore_dtype=weight_dtype,
        **pipeline_args,
    )

    # # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
    # scheduler_args = {}
    # if "variance_type" in pipeline.scheduler.config:
    #     variance_type = pipeline.scheduler.config.variance_type
    #     if variance_type in ["learned", "learned_range"]:
    #         variance_type = "fixed_small"
    #     scheduler_args["variance_type"] = variance_type
    # pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, **scheduler_args)
    # pipeline.set_progress_bar_config(disable=True)

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
            input_model_file = os.path.join(args.output_dir, path, "pytorch_model.ckpt")
            ms.load_param_into_net(transformer, ms.load_checkpoint(input_model_file))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0
    # if args.validation_prompt is not None:
    #     log_validation(pipeline, args, trackers, logging_dir, first_epoch)
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
        train_loss = 0.0
        for step, batch in (
            ((_, None) for _ in range(len(train_dataloader)))  # dummy iterator
            if args.enable_mindspore_data_sink
            else enumerate(train_dataloader_iter)
        ):
            if args.enable_mindspore_data_sink:
                loss, model_pred = sink_process()
            else:
                loss, model_pred = train_step(*batch)
            train_loss += loss.numpy().item()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if train_step.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                for tracker_name, tracker in trackers.items():
                    if tracker_name == "tensorboard":
                        tracker.add_scalar("train/loss", train_loss, global_step)
                train_loss = 0.0

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
                        save_model_hook(transformer, save_path)
                        output_model_file = os.path.join(save_path, "pytorch_model.ckpt")
                        ms.save_checkpoint(transformer, output_model_file)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.numpy().item(), "lr": optimizer.get_lr().numpy().item()}
            progress_bar.set_postfix(**logs)
            for tracker_name, tracker in trackers.items():
                if tracker_name == "tensorboard":
                    tracker.add_scalars("train", logs, global_step)

            if global_step >= args.max_train_steps:
                break
        if args.validation_prompt is not None and (epoch + 1) % args.validation_epochs == 0:
            log_validation(pipeline, args, trackers, logging_dir, epoch + 1)
    if is_master(args):
        unwrapped_transformer = unwrap_model(transformer)
        transformer_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_transformer))
        StableDiffusionPipeline.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=transformer_lora_state_dict,
            safe_serialization=True,
        )

    # Final inference
    if args.validation_prompt is not None:
        log_validation(pipeline, args, trackers, logging_dir, args.num_train_epochs)
    for tracker_name, tracker in trackers.items():
        if tracker_name == "tensorboard":
            tracker.close()


class TrainStep_PixartSigma(TrainStep):
    def __init__(
        self,
        vae: nn.Cell,
        text_encoder: nn.Cell,
        transformer: nn.Cell,
        optimizer: nn.Optimizer,
        noise_scheduler,
        weight_dtype,
        length_of_dataloader,
        args,
        transformer_config,
    ):
        super().__init__(
            transformer,
            optimizer,
            StaticLossScaler(65536),
            args.max_grad_norm,
            args.gradient_accumulation_steps,
            gradient_accumulation_kwargs=dict(length_of_dataloader=length_of_dataloader),
        )

        self.transformer = transformer
        self.vae = vae
        if self.vae is not None:
            self.vae_scaling_factor = vae.config.scaling_factor
        self.text_encoder = text_encoder
        self.noise_scheduler = noise_scheduler
        self.noise_scheduler_num_train_timesteps = noise_scheduler.config.num_train_timesteps
        self.noise_scheduler_prediction_type = noise_scheduler.config.prediction_type
        self.weight_dtype = weight_dtype
        self.args = AttrJitWrapper(**vars(args))
        self.transformer_config = transformer_config

    def forward(self, pixel_values, input_ids, attention_mask=None):
        pixel_values = pixel_values.to(dtype=self.weight_dtype)
        latents = self.vae.diag_gauss_dist.sample(self.vae.encode(pixel_values)[0])
        latents = latents * self.vae_scaling_factor

        # Sample noise that we'll add to the latents
        noise = ops.randn_like(latents, dtype=latents.dtype)
        if self.args.noise_offset:
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += self.args.noise_offset * ops.randn((latents.shape[0], latents.shape[1], 1, 1))

        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = ops.randint(0, self.noise_scheduler_num_train_timesteps, (bsz,))
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        # noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        prompt_embeds = self.text_encoder(input_ids, attention_mask=attention_mask)[0]
        prompt_attention_mask = attention_mask

        # Get the target for loss depending on the prediction type
        if self.args.prediction_type is not None:
            # set prediction_type of scheduler if defined
            self.noise_scheduler.register_to_config(prediction_type=self.args.prediction_type)

        if self.noise_scheduler_prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler_prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler_prediction_type}")

        # Prepare micro-conditions.
        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        if self.transformer_config.sample_size == 128 and self.args.micro_conditions:
            resolution = ops.tensor([self.args.resolution, self.args.resolution]).repeat(bsz, 1)
            aspect_ratio = ops.tensor([1.0 * self.args.resolution / self.args.resolution]).repeat(bsz, 1)
            resolution = resolution.to(dtype=self.weight_dtype)
            aspect_ratio = aspect_ratio.to(dtype=self.weight_dtype)
            added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}

        output = self.transformer(
            noisy_latents,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_attention_mask,
            timestep=timesteps,
            added_cond_kwargs=added_cond_kwargs,
        )

        model_pred = output[0].chunk(2, 1)[0]

        if self.args.snr_gamma is None:
            loss = ops.mse_loss(model_pred.float(), target.float(), reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = compute_snr(self.noise_scheduler, timesteps)
            if self.noise_scheduler_prediction_type == "v_prediction":
                # Velocity objective requires that we add one to SNR values before we divide by them.
                snr = snr + 1
            mse_loss_weights = (
                ops.stack([snr, self.args.snr_gamma * ops.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
            )

            loss = ops.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        loss = self.scale_loss(loss)
        return loss, model_pred


if __name__ == "__main__":
    main()
