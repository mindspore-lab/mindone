import csv
import json
import logging
import random
from pathlib import Path

import numpy as np

import mindspore as ms

logger = logging.getLogger(__name__)


class TextDataset:
    def __init__(
        self,
        data_file_path,
        tokenizer=None,
        file_column="video",
        caption_column="caption",
        random_drop_text=False,
        random_drop_text_ratio=0.1,
        output_columns=["file_path", "caption"],
    ):
        logger.info(f"loading annotations from {data_file_path} ...")
        self.parse_data_file(data_file_path)
        self.tokenizer = tokenizer
        self.caption_column = caption_column
        self.file_column = file_column
        self.random_drop_text = random_drop_text
        self.random_drop_text_ratio = random_drop_text_ratio
        self.output_columns = output_columns
        if "caption" not in output_columns:
            raise ValueError("caption column is not in output_colum")
        self.read_captions(self.dataset)
        logger.info(f"Number of text prompts: {self.num_captions}")

    def parse_data_file(self, data_file_path):
        if data_file_path.endswith(".csv"):
            with open(data_file_path, "r") as csvfile:
                self.dataset = list(csv.DictReader(csvfile))
        elif data_file_path.endswith(".json"):
            with open(data_file_path, "r") as f:
                self.dataset = json.load(f)
        else:
            raise ValueError("Only support json and csv file now!")

    def __len__(self):
        return self.num_captions

    def read_captions(self, dataset):
        num_captions = 0
        caption_sample_indices = []
        for i, item in enumerate(dataset):
            captions = item[self.caption_column]
            if isinstance(captions, str):
                captions = [captions]
            num_captions += len(captions)
            caption_sample_indices.extend([i] * len(captions))
        self.num_captions = num_captions
        self.caption_sample_indices = caption_sample_indices

    def __getitem__(self, idx_text):
        idx = self.caption_sample_indices[idx_text]
        row = self.dataset[idx]
        captions = row[self.caption_column]
        if isinstance(captions, str):
            captions = [captions]
        file_path = row[self.file_column]
        # get the caption id
        first_text_index = self.caption_sample_indices.index(idx)
        index = idx_text - first_text_index
        caption = captions[index]
        # use index as an extra identifier
        identifer = f"-{index}"
        file_path = Path(str(file_path))
        extension = file_path.suffix
        file_path = str(file_path.with_suffix("")) + identifer
        file_path = file_path + extension

        if self.random_drop_text:
            if random.random() <= self.random_drop_text_ratio:
                caption = ""

        if self.tokenizer is not None:
            tokens = self.tokenizer(caption)
            if isinstance(tokens, list):
                tokens = np.array(tokens, dtype=np.int64)
            if len(tokens.shape) == 2:  # in case, the tokenizer output [1, 77]
                tokens = tokens[0]
            text_data = tokens
        else:
            text_data = caption
        if "file_path" in self.output_columns:
            return file_path, text_data
        else:
            return text_data


def create_dataloader(
    ds_config,
    batch_size,
    ds_name="text",
    num_parallel_workers=12,
    max_rowsize=32,
    shuffle=True,
    device_num=1,
    rank_id=0,
    drop_remainder=True,
    return_dataset=False,
):
    if ds_name == "text":
        dataset = TextDataset(**ds_config)
        column_names = getattr(dataset, "output_columns", ["file_path", "caption"])
    else:
        raise NotImplementedError

    dataloader = ms.dataset.GeneratorDataset(
        source=dataset,
        column_names=column_names,
        num_shards=device_num,
        shard_id=rank_id,
        python_multiprocessing=True,
        shuffle=shuffle,
        num_parallel_workers=num_parallel_workers,
        max_rowsize=max_rowsize,
    )

    dl = dataloader.batch(
        batch_size,
        drop_remainder=drop_remainder,
    )
    if return_dataset:
        return dl, dataset
    return dl


if __name__ == "__main__":
    ds_config = dict(data_file_path="../videocomposer/datasets/webvid5/video_caption.csv", tokenizer=None)

    dl = create_dataloader(
        ds_config,
        batch_size=2,
    )

    ds_iter = dl.create_dict_iterator(1, output_numpy=True)
    for step, data in enumerate(ds_iter):
        fp = data["file_path"]
        cap = data["caption"]
        print(fp[0], cap[0])
