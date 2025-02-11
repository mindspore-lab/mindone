import csv
import logging
import os

import albumentations
import cv2
import numpy as np
from PIL import Image

import mindspore as ms

logger = logging.getLogger()


def create_transforms(h, w, interpolation="bicubic", backend="al", use_safer_augment=True):
    """
    pipeline: flip -> resize -> crop
    h, w : target resize height, weight
    NOTE: we change interpolation to bicubic for its better precision and used in SD. TODO: check impact on performance
    """
    if backend == "pt":
        from torchvision import transforms
        from torchvision.transforms.functional import InterpolationMode

        mapping = {"bilinear": InterpolationMode.BILINEAR, "bicubic": InterpolationMode.BICUBIC}

        pixel_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.Resize(h, interpolation=mapping[interpolation]),
                transforms.CenterCrop((h, w)),
            ]
        )
    elif backend == "al":
        # expect rgb image in range 0-255, shape (h w c)
        from albumentations import CenterCrop, HorizontalFlip, Resize, SmallestMaxSize

        mapping = {"bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC}
        if use_safer_augment:
            pixel_transforms = albumentations.Compose(
                [
                    SmallestMaxSize(max_size=h, interpolation=mapping[interpolation]),
                    CenterCrop(h, w),
                ],
            )
        else:
            pixel_transforms = albumentations.Compose(
                [
                    HorizontalFlip(p=0.5),
                    Resize(h, h, interpolation=mapping[interpolation]),
                    CenterCrop(h, w),
                ],
            )

    elif backend == "ms":
        # TODO: MindData doesn't support batch transform. can NOT make sure all frames are flipped the same
        from mindspore.dataset.transforms import Compose
        from mindspore.dataset.vision import CenterCrop, Inter, RandomHorizontalFlip, Resize

        mapping = {"bilinear": Inter.BILINEAR, "bicubic": Inter.BICUBIC}
        pixel_transforms = Compose(
            [
                RandomHorizontalFlip(),
                Resize(h, interpolation=mapping[interpolation]),
                CenterCrop((h, w)),
            ]
        )
    else:
        raise NotImplementedError

    return pixel_transforms


class TextImageDataset:
    """
    Dataset to read csv file, which returns image pixel values after transformation (SD VAE),
    token ids (optional), and class label ids (optional).
    Modified from `TextVideoDataset` from AnimateDiff in mindone.
    """

    def __init__(
        self,
        csv_path,
        image_folder,
        sample_size=256,
        transform_backend="al",  # ms, pt, al
        tokenizer=None,
        image_column="image",
        caption_column=None,
        class_column=None,
    ):
        logger.info(f"loading annotations from {csv_path} ...")
        with open(csv_path, "r") as csvfile:
            self.dataset = list(csv.DictReader(csvfile))
        self.length = len(self.dataset)
        logger.info(f"Num data samples: {self.length}")

        self.image_folder = image_folder

        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)

        # it should match the transformation used in SD/VAE pretraining, especially for normalization
        self.pixel_transforms = create_transforms(
            sample_size[0], sample_size[1], interpolation="bicubic", backend=transform_backend
        )
        self.transform_backend = transform_backend
        self.tokenizer = tokenizer
        self.image_column = image_column
        assert self.image_column is not None, "The input csv file must specify the image column"
        self.caption_column = caption_column
        if self.caption_column is not None and self.tokenizer is None:
            logger.warning(
                f"The caption_column is provided {self.caption_column}, but tokenizer is not provided! "
                "The text tokens will be dummy placeholders!"
            )
        self.class_column = class_column

    def get_batch(self, idx):
        # get image raw pixel
        image_dict = self.dataset[idx]
        image_fn = image_dict[self.image_column]
        if self.caption_column is not None:
            caption = image_dict[self.caption_column]
        else:
            caption = ""  # dummy placeholders
        if self.class_column is not None:
            class_label = int(image_dict[self.class_column])
        else:
            class_label = 0  # a dummy class label as a placeholder
        image_path = os.path.join(self.image_folder, image_fn)
        pixel_values = np.array(Image.open(image_path).convert("RGB"))
        return pixel_values, caption, class_label

    def __len__(self):
        return self.length

    def tokenize(self, text):
        # a hack to determine if use transformers.CLIPTokenizer
        # should handle it better
        if type(self.tokenizer).__name__ == "CLIPTokenizer":
            return self._clip_tokenize(text)

        SOT_TEXT = self.tokenizer.sot_text  # "[CLS]"
        EOT_TEXT = self.tokenizer.eot_text  # "[SEP]"
        CONTEXT_LEN = self.tokenizer.context_length

        sot_token = self.tokenizer.encoder[SOT_TEXT]
        eot_token = self.tokenizer.encoder[EOT_TEXT]
        tokens = [sot_token] + self.tokenizer.encode(text) + [eot_token]
        result = np.zeros([CONTEXT_LEN]) + eot_token
        if len(tokens) > CONTEXT_LEN:
            tokens = tokens[: CONTEXT_LEN - 1] + [eot_token]
        result[: len(tokens)] = tokens

        return result.astype(np.int64)

    def _clip_tokenize(self, texts):
        batch_encoding = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.tokenizer.context_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
        )
        tokens = np.array(batch_encoding["input_ids"], dtype=np.int32)
        return tokens

    def __getitem__(self, idx):
        """
        Returns:
            tuple (image, text_data)
                - image: preprocessed images in shape (c, h, w)
                - text_data: if tokenizer provided, tokens shape (context_max_len,), otherwise text string
        """
        pixel_values, caption, class_label = self.get_batch(idx)
        if self.transform_backend == "pt":
            import torch

            pixel_values = torch.from_numpy(pixel_values).permute(2, 0, 1).contiguous()  # (h, w, c) -> (c, h, w)
            pixel_values = self.pixel_transforms(pixel_values)
            pixel_values = pixel_values.numpy()
        elif self.transform_backend == "al":
            output = self.pixel_transforms(image=pixel_values)["image"]
            pixel_values = np.transpose(output, (2, 0, 1))
        else:
            raise NotImplementedError

        pixel_values = (pixel_values / 127.5 - 1.0).astype(np.float32)

        if self.tokenizer is not None:
            tokens = self.tokenize(caption)
            # print("D--: ", type(text_data))
            if isinstance(tokens, list):
                tokens = np.array(tokens, dtype=np.int64)
            if len(tokens.shape) == 2:  # in case, the tokenizer output [1, 77]
                tokens = tokens[0]
            text_data = tokens
        else:
            text_data = np.array([49407], dtype=np.int64)  # dummy token ids as a placeholder. Do not return a string.
        return pixel_values, class_label, text_data


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
