"""
Llama 3 model fine-tuning script.
This script with default values fine-tunes a pretrained Meta Llama3 on the `Yelp/yelp_review_full` dataset,
"""


import argparse
import ast
from typing import Dict

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

import mindspore as ms
from mindspore import nn

from mindone.transformers.mindspore_adapter import HF2MSDataset, TrainOneStepWrapper, auto_mixed_precision
from mindone.transformers.models.llama import LlamaForSequenceClassification


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="meta-llama/Meta-Llama-3-8B", help="pretrained model name")
    parser.add_argument("--dataset_path", type=str, default="Yelp/yelp_review_full", help="dataset path.")
    parser.add_argument(
        "--zero_stage", type=int, default=0, choices=[0, 1, 2], help="stage of ZeRO optimizer parallelism"
    )
    parser.add_argument(
        "--fp16", action="store_true", default=False, help="whether or not to enable mix precision with float16"
    )
    parser.add_argument(
        "--bf16", action="store_true", default=False, help="whether or not to enable mix precision with bfloat16"
    )
    parser.add_argument(
        "--is_distribute", type=ast.literal_eval, default=False, help="whether or not to run distribute"
    )
    parser.add_argument("--rank", type=int, default=0, help="id of card")
    parser.add_argument("--rank_size", type=int, default=1, help="num of cards")
    args = parser.parse_args()
    print(args)

    # 0. set mindspore context
    ms.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"})
    if args.is_distribute:
        from mindspore.communication import get_group_size, get_rank, init

        init()
        args.rank = get_rank()
        args.rank_size = get_group_size()
        ms.reset_auto_parallel_context()
        ms.set_auto_parallel_context(
            parallel_mode=ms.ParallelMode.DATA_PARALLEL,
            gradients_mean=True,
            device_num=get_group_size(),
        )

    # 1. create dataset
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

    def ms_data_collator(features, batch_info):
        first = features[0]
        assert isinstance(first, Dict)
        batch = {}
        batch["labels"] = np.array([f["label"] for f in features], dtype=np.int32)
        for k, v in first.items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, np.ndarray):
                    batch[k] = np.stack([f[k] for f in features])
                else:
                    batch[k] = np.array([f[k] for f in features])
        return batch

    batch_size, num_epochs = 1, 3
    train_dataloader = ms.dataset.GeneratorDataset(
        HF2MSDataset(small_train_dataset), column_names="item", shard_id=args.rank, num_shards=args.rank_size
    )
    train_dataloader = train_dataloader.batch(batch_size=batch_size, per_batch_map=ms_data_collator)
    train_dataloader = train_dataloader.repeat(1)
    train_dataloader = train_dataloader.create_dict_iterator(num_epochs=num_epochs, output_numpy=True)

    # 2. create train network and mix precision
    model = LlamaForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=5,
        use_flash_attention_2=True,
        mindspore_dtype=ms.bfloat16 if args.bf16 else (ms.float16 if args.fp16 else None),
    )
    model.gradient_checkpointing_enable()

    assert not (args.fp16 and args.bf16)
    if args.fp16:
        model = auto_mixed_precision(model, "O2", ms.float16)
    if args.bf16:
        model = auto_mixed_precision(model, "O2", ms.bfloat16)

    if args.zero_stage == 0:
        optimizer = nn.AdamWeightDecay(model.trainable_params(), learning_rate=5e-6)
    elif args.zero_stage == 1:
        from mindone.transformers.mindspore_adapter import AdamWeightDecayZeRO1

        optimizer = AdamWeightDecayZeRO1(model.trainable_params(), learning_rate=5e-6)
    elif args.zero_stage == 2:
        from mindone.transformers.mindspore_adapter import AdamWeightDecayZeRO2

        optimizer = AdamWeightDecayZeRO2(model.trainable_params(), learning_rate=5e-6)
    else:
        raise ValueError

    class ReturnLoss(nn.Cell):
        def __init__(self, model):
            super(ReturnLoss, self).__init__(auto_prefix=False)
            self.model = model

        def construct(self, *args, **kwargs):
            outputs = self.model(*args, **kwargs)
            loss = outputs[0]
            return loss

    train_model = TrainOneStepWrapper(ReturnLoss(model), optimizer)

    # 3. training
    train_model.set_train()
    for step, batch in enumerate(train_dataloader):
        batch = batch["item"]

        # inputs dict to tuple
        tuple_inputs = (
            ms.Tensor(batch["input_ids"], ms.int32),
            ms.Tensor(batch["attention_mask"], ms.bool_),
            None,
            None,
            None,
            ms.tensor(batch["labels"], ms.int32),
        )

        loss, _, overflow = train_model(*tuple_inputs)

        print(f"step: {step}, loss: {loss}")


if __name__ == "__main__":
    main()
