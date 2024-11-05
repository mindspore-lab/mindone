import os
import argparse
import evaluate
import numpy as np
import mindspore as ms
from mindspore import nn
from typing import Dict
from datasets import load_dataset
from transformers import AutoTokenizer, HfArgumentParser
from dataclasses import dataclass, field

from mindone.transformers.models.llama import LlamaForSequenceClassification
from mindone.transformers.trainer import Trainer
from mindone.transformers.training_args import TrainingArguments
from mindone.transformers.mindspore_adapter import HF2MSDataset, TrainOneStepWrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="meta-llama/Meta-Llama-3-8B", help="pretrained model name")
    parser.add_argument("--dataset_path", type=str, default="Yelp/yelp_review_full", help="dataset path.")
    parser.add_argument("--train_batch_size", type=int, default=1, help="train batch size")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="dataset path.")
    parser.add_argument("--rank_size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 1. create dataset
    dataset = load_dataset(args.dataset_path)
    dataset["train"] = dataset["train"].shuffle(seed=42).select(range(1000))

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

    batch_size, num_epochs = 1, 3
    train_dataloader = ms.dataset.GeneratorDataset(HF2MSDataset(small_train_dataset), column_names="item")
    train_dataloader = train_dataloader.batch(batch_size=batch_size, per_batch_map=ms_data_collator)
    train_dataloader = train_dataloader.repeat(1)
    train_dataloader = train_dataloader.create_dict_iterator(num_epochs=num_epochs, output_numpy=True)

    # 2. create train network
    model = LlamaForSequenceClassification.from_pretrained(args.model_path, num_labels=5, use_flash_attention_2=True)
    optimizer = nn.AdamWeightDecay(model, learning_rate=5e-6)
    train_model = TrainOneStepWrapper(model, optimizer)

    # 3. training
    train_model.set_train()
    for batch in train_dataloader:
        tuple_inputs = (
            ms.Tensor(batch["input_ids"], ms.int32),
            ms.Tensor(batch["attention_mask"], ms.bool_),
            None,
            None,
            None,
            ms.tensor(batch["labels"], ms.int32)
        )

    if args.do_train:
        nb_tr_steps, tr_loss, exp_average_loss = 0, 0, None
        model.set_train(True)
        for epoch in range(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_steps = 0
            train_iterator = train_dataloader.create_tuple_iterator(num_epochs=1, output_numpy=True)
            for step, batch in enumerate(train_iterator):
                input_ids, mc_token_ids, lm_labels, mc_labels = batch

                # to tensor
                input_ids, mc_token_ids, lm_labels, mc_labels = \
                    ms.Tensor(input_ids), ms.Tensor(mc_token_ids), ms.Tensor(lm_labels), ms.Tensor(mc_labels)

                loss = train_step_fn(input_ids, mc_token_ids=mc_token_ids, lm_labels=lm_labels, mc_labels=mc_labels)
                tr_loss += loss.asnumpy().item()
                exp_average_loss = (
                    loss.item() if exp_average_loss is None else 0.7 * exp_average_loss + 0.3 * loss.item()
                )
                nb_tr_steps += 1
                logger.info("Epoch: {}, Step: {}, Training loss: {:.2e}".format(epoch, step, exp_average_loss))

    # Save a trained model
    if args.do_train:
        # Save a trained model, configuration and tokenizer
        model_to_save = model  # Only save the model itself

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

        ms.save_checkpoint(model_to_save, output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(args.output_dir)


if __name__ == '__main__':
    main()
