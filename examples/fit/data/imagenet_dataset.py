import logging
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import mindspore as ms
from mindspore.dataset.transforms import Compose, vision

from mindone.models.modules.pos_embed import get_2d_sincos_pos_embed, precompute_freqs_cis_2d

_logger = logging.getLogger()

ALLOWED_FORMAT = {".jpeg", ".jpg", ".bmp", ".png"}


class _ResizeByMaxValue:
    def __init__(self, max_size: int = 256, vae_scale: int = 8, patch_size: int = 2) -> None:
        self.max_size = max_size
        self.scale = vae_scale * patch_size

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        image_area = w * h
        max_area = self.max_size * self.max_size
        if image_area > max_area:
            ratio = max_area / image_area
            new_w = w * np.sqrt(ratio)
            new_h = h * np.sqrt(ratio)
        else:
            new_w = w
            new_h = h

        round_w, round_h = (np.round(np.array([new_w, new_h]) / self.scale) * self.scale).astype(int).tolist()
        if round_w * round_h > max_area:
            round_w, round_h = (np.floor(np.array([new_w, new_h]) / self.scale) * self.scale).astype(int).tolist()

        round_w, round_h = max(round_w, self.scale), max(round_h, self.scale)
        img = img.resize((round_w, round_h), resample=Image.BICUBIC)
        return img


class ImageNetWithPathIterator:
    def __init__(self, config) -> None:
        self.image_paths = self._inspect_images(config["data_folder"])
        self.transform = self._create_transform(
            max_size=config.get("sample_size", 256), patch_size=config.get("patch_size", 2)
        )

    def _inspect_images(self, root: str) -> List[str]:
        images_info = list()

        _logger.info(f"Scanning images under `{root}`.")
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                _, ext = os.path.splitext(f)
                if ext.lower() in ALLOWED_FORMAT:
                    fpath = os.path.join(dirpath, f)
                    images_info.append(fpath)

        if len(images_info) == 0:
            raise RuntimeError(f"Cannot find any image under `{root}`")

        images_info = sorted(images_info)
        return images_info

    def _create_transform(self, max_size: int = 256, patch_size: int = 2):
        operations = Compose(
            [
                _ResizeByMaxValue(max_size=max_size, patch_size=patch_size),
                vision.HWC2CHW(),
                vision.Normalize([127.5, 127.5, 127.5], [127.5, 127.5, 127.5], is_hwc=False),
            ]
        )
        return operations

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]

        with Image.open(path) as f:
            img = f.convert("RGB")

        img = self.transform(img)[0]
        return img, path


class ImageNetLatentIterator:
    def __init__(self, config) -> None:
        self.latent_info = self._inspect_latent(config["data_folder"])
        self.label_mapping = self._create_label_mapping(self.latent_info)
        self.patch_size = config.get("patch_size", 2)
        self.embed_dim = config.get("embed_dim", 1152)
        self.embed_method = config.get("embed_method", "rotate")

    def _inspect_latent(self, root: str) -> List[Dict[str, str]]:
        latent_info = list()

        _logger.info(f"Scanning numpy file under `{root}`.")
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                _, ext = os.path.splitext(f)
                if ext.lower() in ".npy":
                    fpath = os.path.join(dirpath, f)
                    latent_info.append(dict(path=fpath, label=os.path.basename(dirpath)))

        if len(latent_info) == 0:
            raise RuntimeError(f"Cannot find any image under `{root}`")

        latent_info = sorted(latent_info, key=lambda x: x["path"])
        return latent_info

    def _create_label_mapping(self, latent_info: List[Dict[str, str]]):
        labels = set([x["label"] for x in latent_info])
        labels = sorted(list(labels))
        labels = dict(zip(labels, np.arange(len(labels), dtype=np.int32)))
        return labels

    def __len__(self):
        return len(self.latent_info)

    def _random_horiztotal_flip(self, latent: np.ndarray) -> np.ndarray:
        if random.random() < 0.5:
            # perform a random horizontal flip in latent domain
            # mimic the effect of horizontal flip in image (not exactly identical)
            latent = latent[..., ::-1]
        return latent

    def _patchify(self, latent: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        c, h, w = latent.shape
        nh, nw = h // self.patch_size, w // self.patch_size

        latent = np.reshape(latent, (c, nh, self.patch_size, nw, self.patch_size))
        latent = np.transpose(latent, (1, 3, 2, 4, 0))  # nh, nw, patch, patch, c
        latent = np.reshape(latent, (nh * nw, -1))  # nh * nw, patch * patch * c

        if self.embed_method == "rotate":
            pos = precompute_freqs_cis_2d(self.embed_dim, nh, nw).astype(np.float32)
        else:
            pos = get_2d_sincos_pos_embed(self.embed_dim, nh, nw).astype(np.float32)
        return latent, pos

    def __getitem__(self, idx):
        x = self.latent_info[idx]

        latent = np.load(x["path"])

        # N(mean, std)
        mean, std = np.split(latent, 2, axis=0)
        latent = np.random.normal(loc=mean, scale=std, size=mean.shape)

        latent = self._random_horiztotal_flip(latent)
        latent, pos = self._patchify(latent)

        label = self.label_mapping[x["label"]]
        mask = np.ones(latent.shape[0], dtype=np.bool_)
        return latent, label, pos, mask


def create_dataloader_imagenet_preprocessing(
    config,
    device_num: Optional[int] = None,
    rank_id: Optional[int] = None,
):
    dataset = ImageNetWithPathIterator(config)
    dataset = ms.dataset.GeneratorDataset(
        dataset,
        column_names=["image", "path"],
        num_shards=device_num,
        shard_id=rank_id,
        num_parallel_workers=config["num_parallel_workers"],
        shuffle=config["shuffle"],
    )
    dataset = dataset.batch(1, num_parallel_workers=config["num_parallel_workers"])
    return dataset


def create_dataloader_imagenet_latent(
    config,
    device_num: Optional[int] = None,
    rank_id: Optional[int] = None,
):
    dataset = ImageNetLatentIterator(config)
    dataset = ms.dataset.GeneratorDataset(
        dataset,
        column_names=["latent", "label", "pos", "mask"],
        num_shards=device_num,
        shard_id=rank_id,
        num_parallel_workers=config["num_parallel_workers"],
        shuffle=config["shuffle"],
    )

    sample_size = config.get("sample_size", 256)
    patch_size = config.get("patch_size", 2)
    vae_scale = 8
    max_length = sample_size * sample_size // patch_size // patch_size // vae_scale // vae_scale
    embed_dim = config.get("embed_dim", 72)
    C = 4

    pad_info = {
        "latent": ([max_length, patch_size * patch_size * C], 0),
        "label": None,
        "pos": ([max_length, embed_dim], 0),
        "mask": ([max_length], 0),
    }

    dataset = dataset.padded_batch(
        config["batch_size"],
        drop_remainder=True,
        num_parallel_workers=config["num_parallel_workers"],
        pad_info=pad_info,
    )
    return dataset
