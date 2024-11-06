""" Llama 3 model fine-tuning script.
    This script with default values fine-tunes a pretrained Meta Llama3 on the `Yelp/yelp_review_full` dataset,

    Run with multiple cards, example as 8 cards:
        msrun --bind_core=True --worker_num=8 --local_worker_num=8 --master_port=9000 --log_dir=outputs/parallel_logs \
        python finetune_with_mindspore_trainer.py \
          --model_name meta-llama/Meta-Llama-3-8B \
          --dataset_path Yelp/yelp_review_full \
          --output_dir ./outputs \
          --per_device_train_batch_size 8 \
          \
          --is_distribute True \
          --zero_stage 2 \
          --fp16 \

    Run with single card:
        python finetune_with_mindspore_trainer.py \
          --model_name meta-llama/Meta-Llama-3-8B \
          --dataset_path Yelp/yelp_review_full \
          --output_dir ./outputs \
          --per_device_train_batch_size 8
"""


import evaluate
import numpy as np
import mindspore as ms

from datasets import load_dataset
from transformers import AutoTokenizer, HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional

from mindone.transformers.models.llama import LlamaForSequenceClassification
from mindone.transformers.trainer import Trainer
from mindone.transformers.training_args import TrainingArguments
from mindone.transformers.mindspore_adapter import MindSporeArguments, init_environment


@dataclass
class MyArguments(MindSporeArguments, TrainingArguments):
    model_path: str = field(default="meta-llama/Meta-Llama-3-8B/")
    dataset_path: str = field(default="Yelp/yelp_review_full")
    output_dir: str = field(default="./outputs")
    enable_flash_attention: bool = field(default=True)
    gradient_checkpointing: bool = field(default=True)
    is_distribute: bool = field(default=False)


def main():

    parser = HfArgumentParser(
        MyArguments
    )
    args = parser.parse_args_into_dataclasses()[0]

    init_environment(args)

    dataset = load_dataset(args.dataset_path)
    dataset["train"] = dataset["train"].shuffle(seed=42).select(range(1000))
    dataset["test"] = dataset["test"].shuffle(seed=42).select(range(1000))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512,        # Note: pad is need for training batch size is gather than 1.
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    small_train_dataset = tokenized_datasets["train"]
    small_eval_dataset = tokenized_datasets["test"]

    model = LlamaForSequenceClassification.from_pretrained(args.model_path, num_labels=5, use_flash_attention_2=args.enable_flash_attention)

    if args.do_eval:
        metric = evaluate.load("accuracy")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)
    else:
        compute_metrics = None

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == '__main__':
    main()
