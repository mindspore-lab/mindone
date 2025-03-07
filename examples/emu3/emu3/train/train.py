# -*- coding: utf-8 -*-

"""
This script would run OOM.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import transformers as tf
from emu3.mllm import Emu3Config, Emu3ForCausalLM, Emu3Tokenizer
from emu3.train.datasets import Emu3FeatureDataset

import mindspore as ms

# from mindone.utils.amp import auto_mixed_precision
from mindspore.amp import auto_mixed_precision

# from mindone.trainers import get_scheduler
from mindone.transformers.mindspore_adapter import MindSporeArguments, init_environment
from mindone.transformers.trainer import Trainer
from mindone.transformers.training_args import TrainingArguments as tf_TrainingArguments

# from mindspore import nn


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="BAAI/Emu3-Stage1")  # Emu3-Gen/Chat/Stage1


@dataclass
class DataArguments:
    train_data_path: Optional[str] = field(default=None)
    eval_data_path: Optional[str] = field(default=None)
    null_prompt_prob: float = field(default=0.05)
    apply_loss_on_only_vision: bool = field(default=True)
    apply_loss_on_only_text: bool = field(default=False)
    ignore_index: int = field(default=-100)
    visual_token_pattern: str = field(default="<|visual token {token_id:0>6d}|>")
    codebook_size: Optional[int] = field(default=32768)


@dataclass
class TrainingArguments(MindSporeArguments, tf_TrainingArguments):
    report_to: List[str] = field(default_factory=list)
    remove_unused_columns: bool = field(default=False)
    min_learning_rate: Optional[float] = field(default=None)
    image_area: Optional[int] = field(default=None)  # image max resolution, e.g. 720*720, 512*512
    max_position_embeddings: Optional[int] = field(default=None)
    output_dir: str = field(default="./outputs")  # output directory for checkpoints
    enable_flash_attention: bool = field(default=True)  # enable flash_attention_2
    gradient_checkpointing: bool = field(default=True)  # activate gradient checkpointing
    is_distribute: bool = field(default=False)  # use data parallel
    precicion: str = field(default="bf16")  # model precision


def update_configs(model_config, args, fields):
    cross_update = lambda a, b, field_name: (
        setattr(b, field_name, getattr(a, field_name))
        if getattr(b, field_name, None) is None
        else setattr(a, field_name, getattr(b, field_name))
    )

    for f in fields:
        cross_update(model_config, args, f)


def train():
    parser = tf.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # init training env
    init_environment(training_args)

    # load pre-trained model and tokenizer (usually Emu3-Stage1)
    model_config = Emu3Config.from_pretrained(model_args.model_name_or_path)
    update_configs(model_config, training_args, ["image_area", "max_position_embeddings"])
    if training_args.min_learning_rate is not None:
        training_args.lr_scheduler_kwargs["min_lr"] = training_args.min_learning_rate

    dtype = ms.bfloat16
    if training_args.dtype == "fp16":
        dtype = ms.float16
    model = Emu3ForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=model_config,
        attn_implementation="flash_attention_2" if training_args.enable_flash_attention else None,
        mindspore_dtype=dtype,
        use_safetensors=True,
    )
    model = auto_mixed_precision(model, amp_level="auto", dtype=dtype)
    print("loaded Emu3 model")

    tokenizer = Emu3Tokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.max_position_embeddings,
        padding_side="right",
        use_fast=False,
    )

    # load data
    train_dataset = Emu3FeatureDataset(data_args, tokenizer=tokenizer, split="train")
    eval_dataset = None
    if (data_args.eval_data_path is not None) and training_args.do_eval:
        eval_dataset = Emu3FeatureDataset(data_args, tokenizer=tokenizer, split="test")

    # training configs
    if training_args.do_eval:
        import evaluate

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)

    else:
        compute_metrics = None

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    # by default, use AdamW, linear lr scheduler

    trainer.train()


if __name__ == "__main__":
    train()
