import json
import logging
import os
import random
from typing import Tuple

import numpy as np
from PIL import Image
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
        text_drop_prob: float = 0.2,
    ) -> None:
        logger.info(f"loading annotations from `{json_path}`.")
        with open(json_path, "r") as f:
            self.dataset = json.load(f)

        self.length = len(self.dataset)

        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.text_drop_prob = text_drop_prob
        self.interpolation_mode = vision.Inter.BILINEAR
        self.transform = self.create_transform(image_size, self.interpolation_mode)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        record = self.dataset[idx]
        image_path = os.path.join(self.image_dir, record["path"])

        if random.random() < self.text_drop_prob:
            text = ""
        else:
            text = record["prompt"]

        # process text
        encoding = self.tokenizer(text, padding="max_length", truncation=True, return_tensors="np")
        text_ids = encoding.input_ids[0]

        # process image
        image = Image.open(image_path).convert("RGB")

        image = self.transform(image)[0]
        image = np.expand_dims(image, axis=0)  # 1, C, H, W
        return image, text_ids

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
