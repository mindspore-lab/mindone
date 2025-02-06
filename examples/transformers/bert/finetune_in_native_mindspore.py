import argparse

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

import mindspore as ms
from mindspore import nn

from mindone.transformers.mindspore_adapter import HF2MSDataset, TrainOneStepWrapper
from mindone.transformers.models.bert import BertForSequenceClassification


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="google-bert/bert-base-cased", help="pretrained model name")
    parser.add_argument("--dataset_path", type=str, default="Yelp/yelp_review_full", help="dataset path.")
    args = parser.parse_args()
    print(args)

    ms.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"})

    # 1. create dataset
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
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    small_train_dataset = tokenized_datasets["train"]

    def ms_data_collator(features, batch_info):
        batch = {}
        for k, v in features[0].items():
            batch[k] = (
                np.stack([f[k] for f in features]) if isinstance(v, np.ndarray) else np.array([f[k] for f in features])
            )
        return batch

    batch_size, num_epochs = 8, 3
    train_dataloader = ms.dataset.GeneratorDataset(HF2MSDataset(small_train_dataset), column_names="item")
    train_dataloader = train_dataloader.batch(batch_size=batch_size, per_batch_map=ms_data_collator)
    train_dataloader = train_dataloader.repeat(1)
    train_dataloader = train_dataloader.create_dict_iterator(num_epochs=num_epochs, output_numpy=True)

    # 2. create train network
    model = BertForSequenceClassification.from_pretrained(args.model_path, num_labels=5)
    optimizer = nn.AdamWeightDecay(model.trainable_params(), learning_rate=5e-6)

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
            None,
            ms.tensor(batch["labels"], ms.int32),
        )

        loss, _, overflow = train_model(*tuple_inputs)

        print(f"step: {step}, loss: {loss}")


if __name__ == "__main__":
    main()
