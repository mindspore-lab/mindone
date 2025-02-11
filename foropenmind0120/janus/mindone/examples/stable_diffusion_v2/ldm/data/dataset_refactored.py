import logging
from pathlib import Path
from typing import Callable, List

import numpy as np
import pandas as pd
from PIL import Image

from mindspore.dataset.vision import CenterCrop, RandomCrop, RandomHorizontalFlip, Resize

from mindone.data import BaseDataset

_logger = logging.getLogger(__name__)


class ImageDataset(BaseDataset):
    def __init__(self, data_path: str, image_filter_size: int = 0, drop_text_prob: float = 0.0):
        self._data = self._read_data(Path(data_path))
        self._drop_text_prob = drop_text_prob
        self._filter_images(image_filter_size)

        self.output_columns = ["image", "caption"]

    @staticmethod
    def _read_data(data_path: Path):
        annos = data_path.glob("*.csv")
        combined_df = pd.concat([pd.read_csv(anno) for anno in annos])
        combined_df["dir"] = combined_df.apply(lambda row: data_path / row["dir"], axis=1)
        data = combined_df.rename(columns={"dir": "image", "text": "caption"}).to_dict("records")
        return data

    def _filter_images(self, filter_size: int = 0):
        old_len = len(self._data)
        # Use multithreading to speed up reading image sizes?
        self._data = [
            item for item in self._data if item["image"].exists() and min(Image.open(item["image"]).size) >= filter_size
        ]
        if len(self._data) < old_len:
            _logger.info(
                f"Filtered out {old_len - len(self._data)} images as they are either too small or do not exist"
            )

    def __getitem__(self, idx):
        image = Image.open(self._data[idx]["image"])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image)
        caption = self._data[idx]["caption"] if np.random.rand() >= self._drop_text_prob else ""
        return image, caption

    def __len__(self):
        return len(self._data)

    @staticmethod
    def train_transforms(
        target_size: int, tokenizer: Callable[[str], np.ndarray], random_crop: bool = False
    ) -> List[dict]:
        transforms = [
            {
                "operations": [
                    Resize(target_size),
                    RandomCrop(target_size) if random_crop else CenterCrop(target_size),
                    RandomHorizontalFlip(prob=0.5),
                ],
                "input_columns": ["image"],
            },
            {  # keep python transforms separate from C++ for speed
                "operations": [lambda x: (x / 127.5 - 1.0).astype(np.float32)],
                "input_columns": ["image"],
            },
            {"operations": tokenizer, "input_columns": ["caption"]},
        ]
        return transforms
