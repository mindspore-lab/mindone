import csv
import logging
import os
from typing import Any, Callable, Tuple, Union

from PIL import Image

import mindspore as ms
import mindspore.dataset.vision as vision
from mindspore.dataset.transforms import Compose

logger = logging.getLogger()


def create_transforms(size: Union[int, Tuple[int, int]], interpolation: str = "bicubic"):
    mapping = {"bilinear": vision.Inter.BILINEAR, "bicubic": vision.Inter.BICUBIC}
    pixel_transforms = Compose(
        [
            vision.Resize(size, interpolation=mapping[interpolation]),
            vision.CenterCrop(size),
            vision.HWC2CHW(),
            vision.Normalize([127.5, 127.5, 127.5], [127.5, 127.5, 127.5], is_hwc=False),
        ]
    )
    return pixel_transforms


class TextImageDataset:
    def __init__(
        self,
        csv_path: str,
        image_folder: str,
        tokenizer: Callable[..., Any],
        sample_size: int = 256,
        image_column="image",
        caption_column=None,
    ):
        logger.info(f"loading annotations from {csv_path} ...")
        with open(csv_path, "r") as csvfile:
            self.dataset = list(csv.DictReader(csvfile))
        self.length = len(self.dataset)
        logger.info(f"Num data samples: {self.length}")

        self.image_folder = image_folder

        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)

        # it should match the transformation used in SD/VAE pretraining, especially for normalization
        self.pixel_transforms = create_transforms(sample_size[0], sample_size[1], interpolation="bicubic")
        self.tokenizer = tokenizer
        self.image_column = image_column
        assert self.image_column is not None, "The input csv file must specify the image column"
        self.caption_column = caption_column
        if self.caption_column is not None and self.tokenizer is None:
            logger.warning(
                f"The caption_column is provided {self.caption_column}, but tokenizer is not provided! "
                "The text tokens will be dummy placeholders!"
            )

    def get_batch(self, idx):
        # get image raw pixel
        image_dict = self.dataset[idx]
        image_fn = image_dict[self.image_column]
        if self.caption_column is not None:
            caption = image_dict[self.caption_column]
        else:
            caption = ""  # dummy placeholders
        image_path = os.path.join(self.image_folder, image_fn)
        pixel_values = Image.open(image_path).convert("RGB")
        return pixel_values, caption

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Returns:
            tuple (image, text_data)
                - image: preprocessed images in shape (c, h, w)
                - text_data: if tokenizer provided, tokens shape (context_max_len,), otherwise text string
        """
        pixel_values, caption = self.get_batch(idx)
        pixel_values = self.pixel_transforms(pixel_values)[0]
        token, token_mask = self.tokenizer(caption)

        return pixel_values, token, token_mask


# TODO: parse in config dict
def create_dataloader(
    config,
    tokenizer=None,
    device_num=1,
    rank_id=0,
    image_column=None,
    class_column=None,
    caption_column=None,
):
    dataset = TextImageDataset(
        config["csv_path"],
        config["data_folder"],
        sample_size=config.get("sample_size", 256),
        tokenizer=tokenizer,
        image_column=image_column,
        class_column=class_column,
        caption_column=caption_column,
    )
    data_name = "image"
    print("Total number of samples: ", len(dataset))

    # Larger value leads to more memory consumption. Default: 16
    # prefetch_size = config.get("prefetch_size", 16)
    # ms.dataset.config.set_prefetch_size(prefetch_size)

    dataloader = ms.dataset.GeneratorDataset(
        source=dataset,
        column_names=[
            data_name,
            "caption",
            "label",
        ],
        num_shards=device_num,
        shard_id=rank_id,
        python_multiprocessing=True,
        shuffle=config["shuffle"],
        num_parallel_workers=config["num_parallel_workers"],
        max_rowsize=config["max_rowsize"],
    )

    dl = dataloader.batch(
        config["batch_size"],
        drop_remainder=True,
    )

    return dl
