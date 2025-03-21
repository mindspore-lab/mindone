import os
from functools import partial

from PIL import Image

import mindspore.dataset as ds
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision


def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x + x - 1


def pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f).convert("RGB")
    return img


def trans_pillow(x):
    x = Image.fromarray(x, "RGB")
    return x


def resize(x, size):
    w, h = x.size
    if h < w:
        scale = size / h
        h = size
        w = int(w * scale)
    else:
        scale = size / w
        w = size
        h = int(h * scale)
    x = x.resize((w, h), Image.LANCZOS)
    return x


def creat_dataset(
    data_path: str,
    final_reso: int,
    batch_size: int,
    is_training: bool = True,
    hflip: bool = False,
    drop_remainder: bool = False,
    mid_reso: float = 1.125,
    num_shards=None,
    shard_id=None,
    num_parallel_workers=None,
):
    # build augmentations
    mid_reso = round(mid_reso * final_reso)  # first resize to mid_reso, then crop to final_reso
    if is_training:
        aug = [
            trans_pillow,
            partial(resize, size=mid_reso),
            # Resize: resize the shorter edge to mid_reso
            vision.RandomCrop((final_reso, final_reso)),
            vision.ToTensor(),
            normalize_01_into_pm1,
        ]
        if hflip:
            aug.insert(0, vision.RandomHorizontalFlip())
    else:
        aug = [
            trans_pillow,
            partial(resize, size=mid_reso),
            # Resize: resize the shorter edge to mid_reso
            vision.CenterCrop((final_reso, final_reso)),
            vision.ToTensor(),
            normalize_01_into_pm1,
        ]

    aug = transforms.Compose(aug)

    root = os.path.join(data_path, "train") if is_training else os.path.join(data_path, "val")

    # build dataset
    dataset = ds.ImageFolderDataset(
        dataset_dir=root,
        shuffle=True if is_training else False,
        decode=True,
        sampler=None,
        num_shards=num_shards,
        shard_id=shard_id,
        num_parallel_workers=num_parallel_workers,
    )

    dataset = dataset.map(operations=aug, input_columns=["image"], python_multiprocessing=True)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=drop_remainder)

    return dataset
