import json
import logging
import os
import random
from typing import Tuple

import numpy as np
from PIL import Image
from pixart.dataset.constant import (
    ASPECT_RATIO_256_BIN,
    ASPECT_RATIO_512_BIN,
    ASPECT_RATIO_1024_BIN,
    ASPECT_RATIO_2048_BIN,
)
from pixart.dataset.utils import classify_height_width_bin
from transformers import AutoTokenizer

from mindspore.dataset.transforms import Compose, vision

logger = logging.getLogger(__name__)


class ImageDataset:
    def __init__(
        self,
        json_path: str,
        image_dir: str,
        image_size: int,
        tokenizer: AutoTokenizer,
        real_prompt_ratio: float = 0.5,
        multi_scale: bool = False,
    ) -> None:
        logger.info(f"loading annotations from `{json_path}`.")
        with open(json_path, "r") as f:
            self.dataset = json.load(f)

        self.length = len(self.dataset)
        logger.info(f"Num data samples: {self.length}")

        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.real_prompt_ratio = real_prompt_ratio
        self.interpolation_mode = vision.Inter.BICUBIC  # TODO: support lanczos
        self.multi_scale = multi_scale

        if not self.multi_scale:
            self.ratio = None
            self.transform = self.create_transform(image_size, self.interpolation_mode)
        else:
            if image_size == 2048:
                self.ratio = ASPECT_RATIO_2048_BIN
            elif image_size == 1024:
                self.ratio = ASPECT_RATIO_1024_BIN
            elif image_size == 512:
                self.ratio = ASPECT_RATIO_512_BIN
            elif image_size == 256:
                self.ratio = ASPECT_RATIO_256_BIN
            else:
                raise ValueError("`image_size` must be 256, 512, 1024, 2048 when `multi_scale=True`")
            self.transform = None

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        record = self.dataset[idx]
        image_path = os.path.join(self.image_dir, record["path"])

        if random.random() < self.real_prompt_ratio:
            text = record["prompt"]
        else:
            text = record["sharegpt4v"]

        # process text
        encoding = self.tokenizer(text, padding="max_length", truncation=True, return_tensors="np")
        text_ids, text_mask = encoding.input_ids, encoding.attention_mask.astype(np.bool_)

        # process image
        image = Image.open(image_path).convert("RGB")

        if not self.multi_scale:
            image = self.transform(image)[0]
        else:
            width, height = image.size
            closest_height, closest_width = classify_height_width_bin(height, width, self.ratio)
            if closest_height / height > closest_width / width:
                resize_size = closest_height, int(width * closest_height / height)
            else:
                resize_size = int(height * closest_width / width), closest_width
            transform = self.create_multi_scale_transform(
                resize_size, (closest_height, closest_width), self.interpolation_mode
            )
            image = transform(image)[0]

        return image, text_ids, text_mask

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

    @staticmethod
    def create_multi_scale_transform(
        image_size: Tuple[int, int], crop_size: Tuple[int, int], interpolation: vision.Inter
    ) -> Compose:
        return Compose(
            [
                vision.Resize(image_size, interpolation=interpolation),
                vision.CenterCrop(crop_size),
                vision.ToTensor(),
                vision.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], is_hwc=False),
            ]
        )
