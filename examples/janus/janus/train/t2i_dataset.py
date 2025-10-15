import csv
import logging
import os
import random
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import numpy as np
from datasets import load_dataset
from janus.models import VLChatProcessor
from PIL import Image

import mindspore as ms
from mindspore.dataset.transforms import Compose, vision

logger = logging.getLogger(__name__)


class TextImageDataset:
    def __init__(
        self,
        vl_chat_processor: VLChatProcessor,
        csv_path: str = None,
        data_dir: str = None,
        parquet_dir: str = None,
        max_token_length: int = 1024,
        image_size: int = 384,
        null_prompt_prob: float = 0.0,
        num_samples: int = -1,
    ) -> None:
        if parquet_dir is None:  # when not defined, by default take chineseart dataset csv
            self.parquet = False
            logger.info(f"loading annotations from `{csv_path}`.")
            with open(csv_path, "r") as csvfile:
                self.dataset = list(csv.DictReader(csvfile))
            if num_samples > 0:
                logger.info(f"sequential slice dataset samples to {num_samples}")
                self.dataset = self.dataset[:num_samples]
        else:
            self.parquet = True
            logger.info(f"loading annotations from `{parquet_dir}`.")
            self.dataset = load_dataset(parquet_dir, split="test")
            if num_samples > 0:
                logger.info(f"sequential slice dataset samples to {num_samples}")
                self.dataset = self.dataset.select(range(num_samples))

        self.length = len(self.dataset)

        self.data_dir = data_dir
        self.vl_chat_processor = vl_chat_processor
        self.interpolation_mode = vision.Inter.ANTIALIAS  # vision.Inter.BICUBIC
        self.null_prompt_prob = null_prompt_prob

        self.transform = self.create_transform(image_size, self.interpolation_mode)
        self.max_token_length = max_token_length
        if image_size != 384:
            logger.warning(f"JanusPro should be trained using fixed image size of 384, but get {image_size}")

        assert (image_size / 16) ** 2 == self.vl_chat_processor.num_image_tokens, (
            "(image_size / vq_downsample_rate)^2 "
            + " should be equal to number of image tokens set in vl chat processor"
        )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        record = self.dataset[idx]
        if self.parquet:
            image = record["image"].convert("RGB")
        else:
            image_path = os.path.join(self.data_dir, record["image_path"])
            image = Image.open(image_path).convert("RGB")

        if random.random() < self.null_prompt_prob:
            caption = ""
        else:
            if self.parquet:
                caption = record["caption"]
            else:
                caption = record["text_en"]

        # process text
        (
            input_ids,
            labels,
            attention_mask,
            image_seq_mask,
        ) = self.prepare_sft_inputs_and_label(caption)

        # process image
        image = self.transform(image)[0]
        image = image[None, ...]  # add temporal axis

        task_type = np.array(2, dtype=np.int32)

        return task_type, input_ids, labels, attention_mask, image_seq_mask, image

    @staticmethod
    def create_transform(image_size: int, interpolation: vision.Inter) -> Compose:
        return Compose(
            [
                vision.Resize(image_size, interpolation=interpolation),
                vision.CenterCrop(image_size),
                vision.ToTensor(),
                vision.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], is_hwc=False),
            ]
        )

    def prepare_sft_inputs_and_label(self, caption):
        # convert to sft prompt
        conversation = [
            {
                "role": "<|User|>",
                "content": caption,
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.vl_chat_processor.sft_format,
            system_prompt="",
        )

        # TODO: add eos tag?
        vlcp = self.vl_chat_processor
        prompt = sft_format + vlcp.image_start_tag + (vlcp.image_tag * vlcp.num_image_tokens) + vlcp.image_end_tag
        # add image placeholder tokens and padding to max length
        # left padding (default), same as inference. eos will be added
        input_ids = vlcp.tokenizer.encode(
            prompt,
            add_special_tokens=True,
            padding="max_length",
            max_length=self.max_token_length,
            # padding_side='left',
            truncation=True,
        )
        input_ids = np.array(input_ids, np.int32)

        assert (
            input_ids == vlcp.image_id
        ).sum() == vlcp.num_image_tokens, (
            "text + image tokens exceeds max token length, please adjust max_length or num image token"
        )

        attention_mask = np.ones(shape=[len(input_ids)], dtype=np.bool_)
        attention_mask[input_ids == vlcp.pad_id] = 0

        image_seq_mask = np.zeros(shape=[len(input_ids)], dtype=np.bool_)
        image_seq_mask[input_ids == vlcp.image_id] = 1

        # label, only train on vision seq
        ignore_index = -100  # TODO: read from config? but CE Loss didn't accept setting ignore_index
        labels = deepcopy(input_ids)
        labels = np.where(
            (input_ids == vlcp.image_id),
            labels,
            ignore_index,
        )

        return input_ids, labels, attention_mask, image_seq_mask


def _filter_extreme_ratio(dataset: List[Dict[str, Any]], ratio: float = 4.5) -> List[Dict[str, Any]]:
    new_dataset = []
    for record in dataset:
        record_ratio = record.get("ratio", None)
        if record_ratio is None:
            raise ValueError("`ratio` must be provided in dataset column to enable filtering.")
        if abs(record_ratio) > ratio:
            path = record["path"]
            logger.warning(
                f"Skip image `{path}` since the abs. ratio ({record_ratio}) is larger than the threshold ({ratio})."
            )
            continue
        new_dataset.append(record)
    return new_dataset


def create_dataloader_t2i(
    vl_chat_processor: VLChatProcessor,
    csv_path: str = None,
    data_dir: str = None,
    parquet_dir: str = None,
    max_token_length: int = 1024,
    image_size: int = 384,
    null_prompt_prob: float = 0.0,
    num_samples: int = -1,
    batch_size: int = 1,
    shuffle: bool = True,
    num_parallel_workers: int = 8,
    rank: int = 0,
    rank_size: int = 1,
):
    dataset = TextImageDataset(
        csv_path=csv_path,
        data_dir=data_dir,
        parquet_dir=parquet_dir,
        vl_chat_processor=vl_chat_processor,
        max_token_length=max_token_length,
        image_size=image_size,
        null_prompt_prob=null_prompt_prob,
        num_samples=num_samples,
    )

    dataloader = ms.dataset.GeneratorDataset(
        source=dataset,
        column_names=[
            "task_type",
            "input_ids",
            "labels",
            "attention_mask",
            "image_seq_mask",
            "image",
        ],
        shuffle=shuffle,
        num_parallel_workers=num_parallel_workers,
        python_multiprocessing=True,
        num_shards=rank_size,
        shard_id=rank,
    )

    dataloader = dataloader.batch(batch_size, drop_remainder=True)

    return dataloader
