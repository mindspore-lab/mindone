from dataclasses import dataclass, field

import evaluate
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, HfArgumentParser

from mindone.transformers.mindspore_adapter import MindSporeArguments, init_environment
from mindone.transformers.models.bert import BertForSequenceClassification
from mindone.transformers.trainer import Trainer
from mindone.transformers.training_args import TrainingArguments


@dataclass
class MyArguments(MindSporeArguments, TrainingArguments):
    model_path: str = field(default="google-bert/bert-base-cased")
    dataset_path: str = field(default="Yelp/yelp_review_full")


def main():
    parser = HfArgumentParser(MyArguments)
    args = parser.parse_args_into_dataclasses()[0]

    init_environment(args)

    dataset = load_dataset(args.dataset_path)
    dataset["train"] = dataset["train"].shuffle(seed=42).select(range(1000))
    dataset["test"] = dataset["test"].shuffle(seed=42).select(range(1000))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

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

    model = BertForSequenceClassification.from_pretrained(args.model_path, num_labels=5)

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
