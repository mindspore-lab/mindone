#!/usr/bin/env python
import argparse
import logging
import os
import sys

import yaml

import mindspore as ms
import mindspore.nn as nn
from mindspore import ops
from mindspore.dataset import GeneratorDataset

# TODO: remove in future when mindone is ready for install
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)
import math

from omnigen import OmniGen, OmniGenProcessor
from omnigen.processor import CenterCropTransform
from omnigen.train_helper import DatasetFromJson, TrainDataCollator, sample_timestep, sample_x0
from omnigen.utils import load_ckpt_params
from tensorboardX import SummaryWriter
from tqdm.auto import tqdm
from transformers.models.phi3.configuration_phi3 import Phi3Config

from mindspore.amp import StaticLossScaler
from mindspore.dataset import transforms, vision
from mindspore.nn.utils import no_init_parameters

# from mindone.transformers.models.phi3.modeling_phi3 import Phi3RMSNorm, Phi3MLP, Phi3Attention, Phi3LongRoPEScaledRotaryEmbedding
from mindone.diffusers import AutoencoderKL
from mindone.diffusers._peft import LoraConfig, get_peft_model
from mindone.diffusers.training_utils import AttrJitWrapper, TrainStep, cast_training_params
from mindone.trainers.lr_schedule import create_scheduler
from mindone.trainers.optim import create_optimizer
from mindone.utils.amp import auto_mixed_precision
from mindone.utils.logger import set_logger

logger = logging.getLogger(__name__)


def freeze_params(m: nn.Cell):
    for p in m.get_parameters():
        p.requires_grad = False


def main(args):
    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir)

    # 1. Init env
    ms.set_context(mode=args.mode, device_target=args.device_target, jit_config=dict(jit_level=args.jit_level))
    set_logger(output_dir=os.path.join(args.results_dir, "logs"), rank=0)

    # 2. Model initialize and weight loading

    # 2.1 OmniGen model
    logger.info("OmniGen init")
    config = Phi3Config.from_pretrained(args.model_path)
    with no_init_parameters():
        model = OmniGen(config)
    model.llm.config.use_cache = False
    # model.llm.gradient_checkpointing_enable()
    custom_fp32_cells = []
    if args.dtype == "fp16":
        model_dtype = ms.float16
        model = auto_mixed_precision(model, amp_level="O2", dtype=model_dtype, custom_fp32_cells=custom_fp32_cells)
    elif args.dtype == "bf16":
        model_dtype = ms.bfloat16
        model = auto_mixed_precision(model, amp_level="O2", dtype=model_dtype, custom_fp32_cells=custom_fp32_cells)
    else:
        model_dtype = ms.float32
    if args.model_weight:
        model = load_ckpt_params(model, args.model_weight)
    else:
        logger.info("Initialize network randomly.")

    # 2.2 VAE
    # Keep VAE freeze and compute in float32
    logger.info("vae init")
    vae = AutoencoderKL.from_pretrained(os.path.join(args.model_path, "vae"), mindspore_dtype=ms.float32)
    freeze_params(vae)

    # 2.3 Processor
    processor = OmniGenProcessor.from_pretrained(args.model_path)

    # 3. LoRA config
    if args.use_lora:
        freeze_params(model)
        transformer_lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            init_lora_weights="gaussian",
            target_modules=["qkv_proj", "o_proj"],
        )
        model = get_peft_model(model, transformer_lora_config)
        if args.dtype == "fp16" or args.dtype == "bf16":
            cast_training_params(model, dtype=ms.float32)

        model.print_trainable_parameters()
        lora_parameters = list(filter(lambda p: p.requires_grad, model.get_parameters()))

    # 4. build dataset
    crop_func = CenterCropTransform
    image_transform = transforms.Compose(
        [
            crop_func(args.max_image_size),
            vision.ToTensor(),
            vision.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], is_hwc=False),
        ]
    )

    train_dataset = DatasetFromJson(
        json_file=args.json_file,
        image_path=args.image_path,
        processer=processor,
        image_transform=image_transform,
        max_input_length_limit=args.max_input_length_limit,
        condition_dropout_prob=args.condition_dropout_prob,
    )

    train_dataloader = GeneratorDataset(
        train_dataset,
        column_names=["col1", "col2"],
        shuffle=True,
        num_parallel_workers=args.num_parallel_workers,
        num_shards=1,
        shard_id=0,
    )

    collate_fn = TrainDataCollator(pad_token_id=processor.text_tokenizer.eos_token_id, hidden_size=3072)

    train_dataloader = train_dataloader.batch(
        args.batch_size_per_device,
        drop_remainder=True,
        per_batch_map=lambda col1, col2, batch_info: collate_fn(col1, col2),
        num_parallel_workers=args.num_parallel_workers,
        input_columns=["col1", "col2"],
        output_columns=["input_ids", "attention_mask", "position_ids", "output_images"],
    )

    # 5. build training utils: lr, optim

    if args.scale_lr:
        args.lr = args.lr * args.gradient_accumulation_steps * args.train_batch_size * args.world_size
    if args.lr_scheduler != "constant":
        assert (
            args.optim != "adamw_exp"
        ), "For dynamic LR, mindspore.experimental.optim.AdamW needs to work with LRScheduler"
        lr = create_scheduler(
            name=args.lr_scheduler,
            steps_per_epoch=train_dataloader.get_dataset_size(),
            lr=args.lr,
            end_lr=args.end_learning_rate,
            warmup_steps=args.warmup_steps,
            decay_steps=args.decay_steps,
            num_epochs=args.epochs,
        )

    optimizer = create_optimizer(
        lora_parameters,
        name=args.optim,
        betas=args.betas,
        weight_decay=args.weight_decay,
        lr=lr,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.epochs * num_update_steps_per_epoch

    key_info = "Key Settings:\n" + "=" * 50 + "\n"
    key_info += "\n".join(
        [
            f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.mode}",
            f"JIT level: {args.jit_level}",
            f"Data path: {args.json_file}",
            f"Model type: {args.dtype}",
            f"Batch size: {args.batch_size_per_device}",
            f"Weight decay: {args.weight_decay}",
            f"Grad accumulation steps: {args.gradient_accumulation_steps}",
            f"Num examples: {len(train_dataset)}",
            f"Num batches each epoch: {len(train_dataloader)}",
            f"Num epochs: {args.epochs}",
            f"Total optimization steps: {max_train_steps}",
            f"Max grad norm: {args.max_grad_norm}",
        ]
    )
    key_info += "\n" + "=" * 50
    print(key_info)

    with open(os.path.join(args.results_dir, "args.yaml"), "w") as f:
        yaml.safe_dump(vars(args), stream=f, default_flow_style=False, sort_keys=False)

    global_step = 0
    first_epoch = 0
    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=0,
        desc="Steps",
    )
    train_step = TrainStep_OmniGen(
        vae=vae,
        model=model,
        optimizer=optimizer,
        model_dtype=model_dtype,
        length_of_dataloader=len(train_dataloader),
        args=args,
    ).set_train(True)
    train_dataloader_iter = train_dataloader.create_tuple_iterator(num_epochs=args.epochs)
    for epoch in range(first_epoch, args.epochs):
        model.set_train(True)
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader_iter):
            loss, _ = train_step(*batch)
            train_loss += loss.numpy().item()
            # Checks if the accelerator has performed an optimization step behind the scenes
            if train_step.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % (args.ckpt_every * args.gradient_accumulation_steps) == 0 and global_step > 0:
                    save_path = os.path.join(args.results_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)
                    model.save_pretrained(save_path)

                    logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.numpy().item()}  # , "lr": optimizer.get_lr().numpy().item()}
            progress_bar.set_postfix(**logs)

            trackers = {"tensorboard": SummaryWriter(log_dir=os.path.join(args.results_dir, "logs"))}
            for tracker_name, tracker in trackers.items():
                if tracker_name == "tensorboard":
                    tracker.add_scalar("train/loss", logs["loss"], global_step)
                    # tracker.add_scalar("train/lr", logs["lr"], global_step)
            if global_step >= max_train_steps:
                break


class TrainStep_OmniGen(TrainStep):
    def __init__(
        self,
        vae: nn.Cell,
        model: nn.Cell,
        optimizer: nn.Optimizer,
        model_dtype,
        length_of_dataloader,
        args,
    ):
        super().__init__(
            model,
            optimizer,
            StaticLossScaler(65536),
            args.max_grad_norm,
            args.gradient_accumulation_steps,
            gradient_accumulation_kwargs=dict(length_of_dataloader=length_of_dataloader),
        )

        self.model = model
        self.vae = vae
        if self.vae is not None:
            self.vae_scaling_factor = vae.config.scaling_factor

        self.model_dtype = model_dtype

        self.args = AttrJitWrapper(**vars(args))

    def forward(self, input_ids, attention_mask, position_ids, output_images):
        output_images = self.vae_encode(output_images)
        _bs = len(output_images)  # bs
        x0 = sample_x0(output_images)
        t = sample_timestep(output_images)
        t_bs = t.view(_bs, 1, 1, 1)
        xt = t_bs * output_images + (1 - t_bs) * x0
        ut = output_images - x0
        model_output = self.model(xt, t, input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
        loss = ops.mse_loss(ut, model_output, reduction="mean")
        loss = ops.mean(loss)
        loss = self.scale_loss(loss)
        return loss, model_output

    def vae_encode(self, x):
        if x is not None:
            if self.vae.config.shift_factor is not None:
                _x = ops.stop_gradient(self.vae.encode(x)[0])
                x = self.vae.diag_gauss_dist.sample(_x)
                x = (x - self.vae.config.shift_factor) * self.vae.config.scaling_factor
            else:
                _x = ops.stop_gradient(self.vae.encode(x)[0])
                x = self.vae.diag_gauss_dist.sample(_x)
                x = x * self.vae.config.scaling_factor
            x = x.to(self.model_dtype)
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OmniGen Training script", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Environment setting
    parser.add_argument("--device_target", default="Ascend", choices=["CPU", "GPU", "Ascend"], help="Device target.")
    parser.add_argument("--mode", default=1, type=int, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1).")
    parser.add_argument("--jit_level", default="O1", choices=["O0", "O1"], help="Jit Level")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "fp16", "bf16"],
        help=("Choose model dtype"),
    )

    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--model_path", type=str, default="pretrained_model")
    parser.add_argument("--model_weight", type=str)
    parser.add_argument("--json_file", type=str)
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--batch_size_per_device", type=int, default=1)
    parser.add_argument("--num_parallel_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=20000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_input_length_limit", type=int, default=1024)
    parser.add_argument("--condition_dropout_prob", type=float, default=0.1)

    parser.add_argument("--max_image_size", type=int, default=1024)

    parser.add_argument(
        "--use_lora",
        action="store_true",
    )
    parser.add_argument("--lora_rank", type=int, default=8)

    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine_decay",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--optim", default="adamw_re", type=str, help="optimizer")
    parser.add_argument(
        "--betas",
        type=float,
        nargs="+",
        default=[0.9, 0.999],
        help="Specify the [beta1, beta2] parameter for the AdamW optimizer.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay.")
    parser.add_argument("--warmup_steps", default=10, type=int, help="warmup steps")
    parser.add_argument("--end_learning_rate", default=1e-7, type=float, help="The end learning rate for Adam.")
    parser.add_argument("--decay_steps", default=0, type=int, help="lr decay steps.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    args = parser.parse_args()
    assert args.max_image_size % 16 == 0, "Image size must be divisible by 16."
    main(args)
