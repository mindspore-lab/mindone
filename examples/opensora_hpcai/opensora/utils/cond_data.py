"""Prepare conditional images: transform, normalization, and saving"""
import os
from typing import List, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from PIL import Image

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


def get_references(
    reference_paths: List[Union[str, None]], image_size: Tuple[int, int]
) -> List[Union[List[np.ndarray], None]]:
    # initialize transform operations once only
    get_references.img_transforms = getattr(
        get_references, "img_transforms", create_infer_transforms(target_size=image_size, interpolation=cv2.INTER_AREA)
    )
    get_references.vid_transforms = getattr(
        get_references,
        "vid_transforms",
        create_infer_transforms(target_size=image_size, interpolation=cv2.INTER_AREA),
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
