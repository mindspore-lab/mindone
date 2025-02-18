"""
Llama 3 model fine-tuning script.
This script with default values fine-tunes a pretrained Meta Llama3 on the `Yelp/yelp_review_full` dataset,
"""


from dataclasses import dataclass, field

import evaluate
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, HfArgumentParser

import mindspore as ms

from mindone.transformers.mindspore_adapter import MindSporeArguments, init_environment
from mindone.transformers.models.llama import LlamaForSequenceClassification
from mindone.transformers.trainer import Trainer
from mindone.transformers.training_args import TrainingArguments


@dataclass
class MyArguments(MindSporeArguments, TrainingArguments):
    model_path: str = field(default="meta-llama/Meta-Llama-3-8B")
    dataset_path: str = field(default="Yelp/yelp_review_full")
    output_dir: str = field(default="./outputs")
    enable_flash_attention: bool = field(default=True)
    gradient_checkpointing: bool = field(default=True)
    is_distribute: bool = field(default=False)


def main():
    parser = HfArgumentParser(MyArguments)
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
            max_length=512,  # Note: pad is need for training batch size is gather than 1.
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    small_train_dataset = tokenized_datasets["train"]
    small_eval_dataset = tokenized_datasets["test"]

    model = LlamaForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=5,
        use_flash_attention_2=args.enable_flash_attention,
        mindspore_dtype=ms.bfloat16 if args.bf16 else (ms.float16 if args.fp16 else None),
    )

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


if __name__ == "__main__":
    main()
