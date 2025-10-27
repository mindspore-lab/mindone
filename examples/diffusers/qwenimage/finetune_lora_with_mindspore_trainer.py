"""
Qwen-Image model fine-tuning script using LoRA.

This script with default values fine-tunes a pretrained Thinker model from Qwen-Image,
on the `lambdalabs/pokemon-blip-captions` dataset for pokemon image generation.

Usage:
```
DEVICE_ID=0 python finetune_lora_with_mindspore_trainer.py \
    --model_path Qwen/Qwen-Image \
    --lora_rank 8 \
    --lora_alpha 16 \
    --dataset_path lambdalabs/pokemon-blip-captions \
    --output_dir ./outputs/lora \
    --num_train_epochs 1 \
    --eval_strategy no \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --save_strategy steps \
    --save_steps 500 \
    --logging_steps 1 \
    --save_total_limit 1 \
    --download_num_workers 4
```

with multi-cards:
note that bf16 requires mindspore>=2.7.0:
```
export ASCEND_RT_VISIBLE_DEVICES=0,1
NPUS=2
MASTER_PORT=9000
LOG_DIR=outputs/lora
msrun --bind_core=True --worker_num=${NPUS} --local_worker_num=${NPUS} --master_port=${MASTER_PORT} --log_dir=${LOG_DIR}/parallel_logs \
python finetune_lora_with_mindspore_trainer.py \
    --output_dir ${LOG_DIR} \
    --num_train_epochs 1 \
    --learning_rate 1e-5 \
    --save_strategy no \
    --bf16
```
"""

import inspect
import io
import logging
import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Union

import evaluate
import numpy as np
from datasets import load_dataset
from PIL import Image
from transformers import HfArgumentParser

import mindspore as ms
import mindspore.mint.distributed as dist
from mindspore import nn, ops

from mindone.diffusers import QwenImagePipeline
from mindone.diffusers.training_utils import cast_training_params
from mindone.peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from mindone.trainers import create_optimizer
from mindone.transformers.mindspore_adapter import MindSporeArguments, init_environment
from mindone.transformers.optimization import get_scheduler
from mindone.transformers.trainer import Trainer
from mindone.transformers.training_args import TrainingArguments

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


@dataclass
class MyArguments(MindSporeArguments, TrainingArguments):
    amp_opt_level: str = field(default="O0")
    dataset_path: str = field(default="lambdalabs/pokemon-blip-captions")
    deepspeed: str = field(default="zero3.json")
    device_target: str = field(default="Ascend")
    do_eval: bool = field(default=False)
    enable_flash_attention: bool = field(default=True)
    # gradient_checkpointing: bool = field(default=False)  # LoRA does not support
    is_distribute: bool = field(default=False)
    lora_rank: int = field(default=8, metadata={"help": "The dimension of the LoRA update matrices."})
    lora_alpha: int = field(default=16, metadata={"help": "The scaling factor alpha of the LoRA."})
    mode: int = field(default=ms.PYNATIVE_MODE, metadata={"help": "Graph(not supported)/Pynative"})
    model_path: str = field(default="Qwen/Qwen-Image")
    output_dir: str = field(default="./outputs")
    per_device_train_batch_size: int = field(
        default=1, metadata={"help": "Batch size per device for training."}
    )  # no use
    resume: Union[bool, str] = field(default=False, metadata={"help": "Resume training from a checkpoint."})
    save_strategy: str = field(default="no", metadata={"help": "Save strategy, no, steps or epoch"})


@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    max_length: int = field(default=4096, metadata={"help": "Fixed token length for training."})
    height: int = field(default=512)
    width: int = field(default=512)
    num_inference_steps: int = field(default=8, metadata={"help": "Inference steps when denoising in training."})


def freeze_params(m: nn.Cell):
    for p in m.get_parameters():
        p.requires_grad = False


def main():
    parser = HfArgumentParser((MyArguments, DataArguments))
    args, data_args = parser.parse_args_into_dataclasses()

    init_environment(args)

    dist.init_process_group()
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL)
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    args.rank_size = world_size
    args.rank = local_rank
    args.zero_stage = 3

    # 1. Load materials
    # 1.1 Load pretrained model from pipe
    ms_dtype = ms.bfloat16 if args.bf16 else (ms.float16 if args.fp16 else ms.float32)
    parent_model = QwenImagePipeline.from_pretrained(
        args.model_path,
        mindspore_dtype=ms_dtype,
    )
    data_args.vae_config = parent_model.vae.config
    data_args.ms_dtype = ms_dtype

    # 1.2 the dataset
    dataset = load_dataset("parquet", data_dir=args.dataset_path, split="train")
    dataset = dataset.shuffle(seed=42)
    train_indices = list(range(666))
    eval_indices = list(range(666, 833))

    def process_function(examples):
        image = Image.open(io.BytesIO(examples["image"]["bytes"])).convert("RGB").resize((512, 512))
        txt = examples["text"]

        # prepare the inputs
        encoder_hidden_states, encoder_hidden_states_mask = parent_model.encode_prompt(txt)
        height = data_args.height
        width = data_args.width
        batch_size = encoder_hidden_states.shape[0]
        hidden_states = parent_model.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=parent_model.transformer.config.in_channels // 4,
            height=height,
            width=width,
            dtype=encoder_hidden_states.dtype,
            generator=np.random.Generator(np.random.PCG64(seed=42)),
            latents=None,
        )

        # prepare the labels: convert the image to latent space
        pixel_values = ms.Tensor(np.array(image, dtype=np.float32)) / 255.0  # (H, W, C) = (512, 512, 3)
        pixel_values = pixel_values.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)

        # write to dataset
        examples["encoder_hidden_states"] = encoder_hidden_states[0].astype(np.float32)
        examples["encoder_hidden_states_mask"] = encoder_hidden_states_mask[0].asnumpy()
        examples["hidden_states"] = hidden_states[0].astype(np.float32)
        examples["txt_seq_lens"] = encoder_hidden_states_mask.shape[-1]
        examples["labels"] = pixel_values

        if not args.do_eval:
            examples.pop("text")  # remove text from examples
            examples.pop("image")  # remove image from examples

        return examples

    tokenized_datasets = dataset.map(process_function, batched=False)
    train_dataset = tokenized_datasets.select(train_indices)
    eval_dataset = tokenized_datasets.select(eval_indices)

    dataset_len = len(train_dataset)
    num_update_steps_per_epoch = max(1, dataset_len // args.gradient_accumulation_steps)
    num_training_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)

    # 2. Prepare for LoRA
    # 2.1. Determine the target model
    model = parent_model.transformer
    model.config.use_cache = False
    model.gradient_checkpointing = True
    model.training = True
    freeze_params(model)
    freeze_params(parent_model.vae)
    freeze_params(parent_model.text_encoder)

    # 2.2. Prepare the LoRA config
    # all attn linear layers
    text_enc_modules = []
    vae_enc_modules = []
    transformer_attn_modules = []
    for i in range(model.config.num_layers):
        transformer_attn_modules.append(f"transformer_blocks.{i}.attn.to_q")
        transformer_attn_modules.append(f"transformer_blocks.{i}.attn.to_k")
        transformer_attn_modules.append(f"transformer_blocks.{i}.attn.to_v")
        transformer_attn_modules.append(f"transformer_blocks.{i}.attn.add_q_proj")
        transformer_attn_modules.append(f"transformer_blocks.{i}.attn.add_k_proj")
        transformer_attn_modules.append(f"transformer_blocks.{i}.attn.add_v_proj")
        transformer_attn_modules.append(f"transformer_blocks.{i}.attn.to_out.0")
        transformer_attn_modules.append(f"transformer_blocks.{i}.attn.to_add_out")

    target_modules = text_enc_modules + vae_enc_modules + transformer_attn_modules
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )

    model = get_peft_model(model, lora_config)
    if args.fp16 or args.bf16:
        cast_training_params(model, dtype=ms.float32)
    model.print_trainable_parameters()

    # 3. [optional] Prepare the evalutaion metric
    if args.do_eval:  # TODO: do not support yet
        metric = evaluate.load("mse")

        def compute_metrics(eval_pred):
            preds, labels = eval_pred
            return metric.compute(predictions=preds, references=labels)

    else:
        compute_metrics = None

    # 4. Training setups: lr scheduler, optimizer, trainer, etc.
    # lr scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler_type,
        base_lr=args.learning_rate,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
        scheduler_specific_kwargs=args.lr_scheduler_kwargs,
    )
    # [required] optimizer
    # FIXME: since only train lora layer,
    # auto-creating optimizer in transformers Trainer may occur empty params list since there is not trainable layernorm layers.
    optimizer_kwargs = {
        "name": "adamw",
        "betas": (args.adam_beta1, args.adam_beta2),
        "eps": args.adam_epsilon,
        "lr": lr_scheduler,
    }
    optimizer = create_optimizer(model.get_base_model().trainable_params(), **optimizer_kwargs)

    # trainer
    trainer = Trainer(
        # model=model.get_base_model(),  # use base model for parsing construct() arguments
        model=TrainStepForQwenImage(
            model.get_base_model(),
            parent_model.vae.decode,
            parent_model.scheduler,
            parent_model.image_processor.postprocess,
            data_args,
        ),
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, lr_scheduler),  # for LoRA
    )  # do not support compute_loss yet

    # trainer.train(resume_from_checkpoint=args.resume) # FIXME: do not support resume training yet
    # FIXME: now use the code below temorarily
    if isinstance(args.resume, str) or (isinstance(args.resume, bool) and args.resume):
        from transformers.trainer_callback import TrainerState
        from transformers.trainer_utils import get_last_checkpoint

        TRAINER_STATE_NAME = "trainer_state.json"
        resume_from_checkpoint = None
        # load potential checkpoint
        resume_path = args.resume if isinstance(args.resume, str) else args.output_dir
        resume_from_checkpoint = get_last_checkpoint(resume_path)
        if resume_from_checkpoint is None:
            raise ValueError(f"No valid checkpoint found in {resume_path}.")
        trainer._load_from_checkpoint(resume_from_checkpoint)
        trainer.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
        trainer.args.num_train_epochs -= trainer.state.epoch

    # train the model and save the LoRA weights
    def save_lora_model(model, output_dir):
        if args.zero_stage == 3:
            all_gather_op = ops.AllGather()

            transformer_lora_layers_to_save_new = {}
            transformer_lora_layers_to_save = get_peft_model_state_dict(model)

            for name, param in transformer_lora_layers_to_save.items():
                if name.startswith("base_model.model."):
                    name = name.replace("base_model.model.", "")
                data = ms.Tensor(all_gather_op(param).asnumpy())
                transformer_lora_layers_to_save_new[name] = data

            if args.rank == 0:
                QwenImagePipeline.save_lora_weights(
                    output_dir,
                    transformer_lora_layers=transformer_lora_layers_to_save_new,
                    weight_name="adapter_model.safetensors",
                )

        else:
            model.save_pretrained(output_dir)

        print(f"Lora model has been saved in {output_dir}.")

    if trainer.args.num_train_epochs > 0:
        trainer.train()
        save_lora_model(model, os.path.join(args.output_dir, "lora"))

    # 5. Inference and evaluation
    if args.do_eval:  # FIXME: bf16 not supported yet
        print("Fuse lora weights into pipe and do eval.")

        # loading function
        def load_lora_model(model, parent_model, input_dir):
            if args.zero_stage == 3:
                import gc

                del model
                del parent_model
                gc.collect()
                ms.hal.empty_cache()

                parent_model = QwenImagePipeline.from_pretrained(
                    args.model_path,
                    mindspore_dtype=ms_dtype,
                )
                parent_model.load_lora_weights(
                    input_dir, weight_name="adapter_model.safetensors", adapter_name="qwenimage-lora"
                )
                parent_model.fuse_lora()
            else:
                model.merge_and_unload()  # merge LoRA weights into the base model
                parent_model.transformer = model.get_base_model()  # replace thinker with LoRA-enhanced model
                parent_model.set_train(False)

        load_lora_model(model, parent_model, os.path.join(args.output_dir, "lora"))

        # inference function
        def inference(txt):
            image = parent_model(
                prompt=txt,
                width=data_args.width,
                height=data_args.height,
                num_inference_steps=8,
                true_cfg_scale=1.0,
                generator=np.random.Generator(np.random.PCG64(seed=42)),
            )[0][0]
            return image

        def calculate_pixel_error(img1, img2):
            arr1 = np.array(img1, dtype=np.float32)
            arr2 = np.array(img2, dtype=np.float32)
            return np.mean(np.abs(arr1 - arr2))

        for idx, example in enumerate(eval_dataset):
            infer_image = inference(example["text"])
            infer_image_save_path = os.path.join(args.output_dir, f"infer_{idx}.png")
            infer_image.save(infer_image_save_path)

            ref_image = Image.open(io.BytesIO(example["image"]["bytes"])).convert("RGB").resize((512, 512))
            ref_image_save_path = os.path.join(args.output_dir, f"ref_{idx}.png")
            ref_image.save(ref_image_save_path)
            error = calculate_pixel_error(infer_image, ref_image)

            log_entry = f"Generation: #{idx} in {infer_image_save_path} with pixel errors {error:.2}\n"
            with open(os.path.join(args.output_dir, "results.txt"), "a") as f:
                print(log_entry.strip(), file=f)


class TrainStepForQwenImage(nn.Cell):
    def __init__(self, base_model, vae_decode, scheduler, image_postprocess, data_args):
        super().__init__()
        self.base_model = base_model
        self.vae_decode = vae_decode
        self.scheduler = scheduler
        self.image_postprocess = image_postprocess
        self.args = data_args

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, _, channels = latents.shape

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), 1, height, width)

        return latents

    def retrieve_timesteps(
        self,
        scheduler,
        num_inference_steps: Optional[int] = None,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        **kwargs,
    ):
        if timesteps is not None and sigmas is not None:
            raise ValueError(
                "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
            )
        if timesteps is not None:
            accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
            if not accepts_timesteps:
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" timestep schedules. Please check whether you are using the correct scheduler."
                )
            scheduler.set_timesteps(timesteps=timesteps, **kwargs)
            timesteps = scheduler.timesteps
            num_inference_steps = len(timesteps)
        elif sigmas is not None:
            accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
            if not accept_sigmas:
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" sigmas schedules. Please check whether you are using the correct scheduler."
                )
            scheduler.set_timesteps(sigmas=sigmas, **kwargs)
            timesteps = scheduler.timesteps
            num_inference_steps = len(timesteps)
        else:
            scheduler.set_timesteps(num_inference_steps, **kwargs)
            timesteps = scheduler.timesteps
        return timesteps, num_inference_steps

    def calculate_shift(
        self,
        image_seq_len,
        base_seq_len: int = 256,
        max_seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
    ):
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu

    def construct(
        self,
        hidden_states,
        encoder_hidden_states,
        encoder_hidden_states_mask,
        txt_seq_lens,
        labels,
        *args,
    ):
        # Prapre timesteps
        latents = hidden_states
        sigmas = np.linspace(1.0, 1 / self.args.num_inference_steps, self.args.num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = self.calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, _ = self.retrieve_timesteps(
            self.scheduler,
            self.args.num_inference_steps,
            sigmas=sigmas,
            mu=mu,
        )

        # Denoising loop
        self.scheduler.set_begin_index(0)
        for i, t in enumerate(timesteps):
            timestep = t.expand((latents.shape[0],)).to(latents.dtype)
            noise_pred = self.base_model(
                hidden_states=latents,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                timestep=timestep / 1000,
                img_shapes=[(1, 32, 32)],
                txt_seq_lens=txt_seq_lens,
                return_dict=False,
            )[0]

            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        latents = self._unpack_latents(latents, self.args.height, self.args.width, 8)  # vae_scale_facotr=8
        latents = latents.to(self.args.ms_dtype)
        latents_mean = (
            ms.tensor(self.args.vae_config.latents_mean).view(1, self.args.vae_config.z_dim, 1, 1, 1).to(latents.dtype)
        )
        latents_std = 1.0 / ms.tensor(self.args.vae_config.latents_std).view(1, self.args.vae_config.z_dim, 1, 1, 1).to(
            latents.dtype
        )
        latents = latents / latents_std + latents_mean
        preds = self.vae_decode(latents, return_dict=False)[0][:, :, 0]
        preds = self.image_postprocess(preds, output_type="ms")

        loss = ms.mint.mean(
            ((preds - labels) ** 2).reshape(preds.shape[0], -1),
            dim=1,
        )

        return loss


if __name__ == "__main__":
    main()
