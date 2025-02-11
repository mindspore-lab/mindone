"""Prepare conditional images: transform, normalization, and saving"""
import os
from typing import List, Tuple, Union

import albumentations
import numpy as np
import pandas as pd
from PIL import Image

from mindspore.dataset.vision import Inter

from mindone.data.video_reader import VideoReader

from ..datasets.video_dataset_refactored import create_infer_transforms

VID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


def load_rgb_images(image_paths):
    assert isinstance(image_paths, list) and len(image_paths) > 0, "image paths must be a non-empty list of strings"
    return [Image.open(path).convert("RGB") for path in image_paths]


def read_captions_from_csv(path, caption_column="caption"):
    df = pd.read_csv(path, usecols=[caption_column])
    captions = df[caption_column].values.tolist()
    return captions


def read_captions_from_txt(path):
    captions = []
    with open(path, "r") as fp:
        for line in fp:
            captions.append(line.strip())
    return captions


def transform_conditional_images(image_paths, H, W, random_crop=True, normalize=True, save_dir=None):
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    image_paths = list(image_paths)
    images = load_rgb_images(image_paths)
    if random_crop:
        cropper = albumentations.RandomResizedCrop(H, W, (1.0, 1.0), ratio=(W / H, W / H))
    else:
        cropper = albumentations.CenterCrop(height=H, width=W)

    if normalize:

        def image_norm(image):
            image = image.mean(dim=0, keepdim=True).repeat(3, 1, 1)
            image -= image.min()
            image /= image.max()
            return image

    else:
        image_norm = lambda x: x.astype(np.float32) / 255

    controlnet_images = [
        image_norm(cropper(image=np.array(img).astype(np.uint8))["image"].transpose(2, 0, 1)) for img in images
    ]  # (c, h, w)

    if save_dir is not None:
        assert os.path.exists(save_dir), f"save_dir {save_dir} does not exist!"
        os.makedirs(os.path.join(save_dir, "control_images"), exist_ok=True)
        my_save_dir = os.path.join(save_dir, "control_images")
        existing_files = [f for f in os.listdir(my_save_dir) if os.path.isfile(os.path.join(my_save_dir, f))]
        for i, image in enumerate(controlnet_images, len(existing_files)):
            Image.fromarray((255.0 * (image.transpose(1, 2, 0))).astype(np.uint8)).save(
                f"{save_dir}/control_images/{i}.png"
            )

    controlnet_images = np.expand_dims(np.stack(controlnet_images), axis=0)
    return controlnet_images


def get_references(
    reference_paths: List[Union[str, None]], image_size: Tuple[int, int]
) -> List[Union[List[np.ndarray], None]]:
    # initialize transform operations once only
    get_references.img_transforms = getattr(  # matching `Image.BICUBIC` from the original repo
        get_references, "img_transforms", create_infer_transforms(target_size=image_size, interpolation=Inter.PILCUBIC)
    )
    get_references.vid_transforms = getattr(
        get_references, "vid_transforms", create_infer_transforms(target_size=image_size, interpolation=Inter.BILINEAR)
    )

    references = []
    for paths in reference_paths:  # iterate over samples
        subref = []
        if paths:  # if not None or empty string
            for ref in paths.split(";"):  # iterate over references for each loop
                ext = os.path.splitext(ref)[-1].lower()
                if ext.lower() in VID_EXTENSIONS:
                    with VideoReader(ref) as reader:
                        frames = reader.fetch_frames(num=len(reader))
                    frames = get_references.vid_transforms(frames)
                else:
                    assert ext.lower() in IMG_EXTENSIONS, f"Unsupported file format: {ext}"
                    with open(ref, "rb") as f:
                        frames = Image.open(f).convert("RGB")
                    frames = get_references.img_transforms(np.array(frames))
                subref.append(frames[None, ...].swapaxes(1, 2))  # add batch dimension. TODO: avoid double axes swap
        references.append(subref)
    return references
