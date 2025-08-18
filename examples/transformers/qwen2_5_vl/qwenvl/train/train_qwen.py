# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import logging
import os
import pathlib
import sys
from pathlib import Path

import transformers

import mindspore as ms
import mindspore.mint as mint
import mindspore.mint.distributed as dist
import mindspore.nn as nn

project_root = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.append(str(project_root))

from qwenvl.data.data_qwen import make_supervised_data_module
from qwenvl.data.data_qwen_packed import make_supervised_data_module_packed
from qwenvl.train.argument import DataArguments, ModelArguments, TrainingArguments
from qwenvl.train.trainer import replace_qwen2_vl_attention_class

import mindone.transformers
from mindone.diffusers._peft import LoraConfig
from mindone.transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Trainer

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: mindone.transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        ms.runtime.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        trainer._save(output_dir, state_dict=state_dict)


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        model.visual.set_grad(True)
        for _, p in model.visual.parameters_and_names():
            p.requires_grad = True
    else:
        model.visual.set_grad(False)
        for _, p in model.visual.parameters_and_names():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        model.visual.merger.set_grad(True)
        for _, p in model.visual.merger.parameters_and_names():
            p.requires_grad = True
    else:
        model.visual.merger.set_grad(False)
        for _, p in model.visual.merger.parameters_and_names():
            p.requires_grad = False

    if model_args.tune_mm_llm and not model_args.tune_mm_lora:
        model.model.set_grad(True)
        model.lm_head.set_grad(True)
        for _, p in model.model.parameters_and_names():
            p.requires_grad = True
        for _, p in model.lm_head.parameters_and_names():
            p.requires_grad = True
    elif model_args.tune_mm_lora:
        model.model.set_grad(True)
        model.lm_head.set_grad(True)
        for _, p in model.model.parameters_and_names():
            p.requires_grad = False
        for _, p in model.lm_head.parameters_and_names():
            p.requires_grad = False
    else:
        model.model.set_grad(False)
        model.lm_head.set_grad(False)
        for _, p in model.model.parameters_and_names():
            p.requires_grad = False
        for _, p in model.lm_head.parameters_and_names():
            p.requires_grad = False


def add_lora(model_args, model):
    target_modules = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
    ]
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    model.add_adapter(lora_config)


def train(attn_implementation="flash_attention_2"):
    global local_rank

    try:
        dist.init_process_group()
        ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL)
        local_rank = dist.get_rank()
        world_size = dist.get_world_size()
    except RuntimeError as e:
        print(f"Cannot start distribution training. {repr(e)}")
        local_rank = 0
        world_size = 1

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.rank_size = world_size
    training_args.rank = local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    if "qwen2.5" in model_args.model_name_or_path.lower():
        with nn.no_init_parameters():
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                mindspore_dtype=(ms.bfloat16 if training_args.bf16 else None),
            )
        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
        ).image_processor
        data_args.model_type = "qwen2.5vl"
    else:
        raise NotImplementedError()

    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        pass
        # if hasattr(model, "enable_input_require_grads"):
        #     model.enable_input_require_grads()
        # else:

        #     def make_inputs_require_grad(module, input, output):
        #         output.requires_grad_(True)

        #     model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    set_model(model_args, model)

    if model_args.tune_mm_lora:
        add_lora(model_args, model)
        if training_args.bf16 and training_args.optim == "bf16_adamw":
            model.to(ms.bfloat16)

    if mint.distributed.get_rank() == 0:
        model.visual.print_trainable_parameters()
        model.model.print_trainable_parameters()

    if data_args.data_packing:
        data_module = make_supervised_data_module_packed(tokenizer=tokenizer, data_args=data_args)
    else:
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
