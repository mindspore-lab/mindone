# Adapted from https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V to work with MindSpore.
import logging
import math
import os
import random
import sys

import numpy as np
from PIL import Image

import mindspore as ms

mindone_lib_path = os.path.abspath("../../")
sys.path.insert(0, mindone_lib_path)

sys.path.append(".")
from functools import partial

import cv2
from albumentations import Compose, Lambda, Resize, ToFloat

logger = logging.getLogger(__name__)


def crop(image, i, j, h, w):
    if len(image.shape) != 3:
        raise ValueError("image should be a 3D tensor")
    return image[i : i + h, j : j + w, ...]


def center_crop_th_tw(image, th, tw, top_crop, **kwargs):
    h, w = image.shape[0], image.shape[1]
    tr = th / tw
    if h / w > tr:
        new_h = int(w * tr)
        new_w = w
    else:
        new_h = h
        new_w = int(h / tr)

    i = 0 if top_crop else int(round((h - new_h) / 2.0))
    j = int(round((w - new_w) / 2.0))
    cropped_image = crop(image, i, j, new_h, new_w)
    return cropped_image


def read_video(video_path: str, num_frames: int, sample_rate: int) -> ms.Tensor:
    from decord import VideoReader, cpu

    decord_vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(decord_vr)
    sample_frames_len = sample_rate * num_frames

    if total_frames > sample_frames_len:
        s = random.randint(0, total_frames - sample_frames_len - 1)
        s = 0
        e = s + sample_frames_len
        num_frames = num_frames
    else:
        s = 0
        e = total_frames
        num_frames = int(total_frames / sample_frames_len * num_frames)
        print(
            f"sample_frames_len {sample_frames_len}, only can sample {num_frames * sample_rate}",
            video_path,
            total_frames,
        )

    frame_id_list = np.linspace(s, e - 1, num_frames, dtype=int)
    video_data = decord_vr.get_batch(frame_id_list).asnumpy()
    return video_data


def create_transform(max_height, max_width, num_frames=None):
    norm_fun = lambda x: 2.0 * x - 1.0

    def norm_func_albumentation(image, **kwargs):
        return norm_fun(image)

    mapping = {"bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC}
    resize = [
        Resize(max_height, max_width, interpolation=mapping["bilinear"]),
    ]

    if num_frames:
        targets = {"image{}".format(i): "image" for i in range(num_frames)}
        resize.append(
            Lambda(
                name="crop_centercrop",
                image=partial(center_crop_th_tw, th=max_height, tw=max_width, top_crop=False),
                p=1.0,
            )
        )
    else:
        targets = {}

    transform = Compose(
        [*resize, ToFloat(255.0), Lambda(name="ae_norm", image=norm_func_albumentation, p=1.0)],
        additional_targets=targets,
    )
    return transform


def preprocess_video(video_data, height: int = 128, width: int = 128):
    num_frames = video_data.shape[0]
    video_transform = create_transform(height, width, num_frames=num_frames)

    inputs = {"image": video_data[0]}
    for i in range(num_frames - 1):
        inputs[f"image{i}"] = video_data[i + 1]

    video_outputs = video_transform(**inputs)
    video_outputs = np.stack(list(video_outputs.values()), axis=0)  # (t h w c)
    video_outputs = np.transpose(video_outputs, (3, 0, 1, 2))  # (c t h w)
    return video_outputs


def preprocess_image(image, height: int = 128, width: int = 128):
    video_transform = create_transform(height, width)

    image = video_transform(image=image)["image"]  # (h w c)
    image = np.transpose(image, (2, 0, 1))[:, None, :, :]  # (c h w) -> (c t h w)
    return image


def transform_to_rgb(x, rescale_to_uint8=True):
    x = np.clip(x, -1, 1)
    x = (x + 1) / 2
    if rescale_to_uint8:
        x = (255 * x).astype(np.uint8)
    return x


def align_to(value, alignment):
    """align hight, width according to alignment

    Args:
        value (int): height or width
        alignment (int): target alignment factor

    Returns:
        int: the aligned value
    """
    return int(math.ceil(value / alignment) * alignment)


def black_image(width, height):
    """generate a black image

    Args:
        width (int): image width
        height (int): image height

    Returns:
        _type_: a black image
    """
    black_image = Image.new("RGB", (width, height), (0, 0, 0))
    return black_image


def get_closest_ratio(height: float, width: float, ratios: list, buckets: list):
    """get the closest ratio in the buckets

    Args:
        height (float): video height
        width (float): video width
        ratios (list): video aspect ratio
        buckets (list): buckets generate by `generate_crop_size_list`

    Returns:
        the closest ratio in the buckets and the corresponding ratio
    """
    aspect_ratio = float(height) / float(width)
    closest_ratio_id = np.abs(ratios - aspect_ratio).argmin()
    closest_ratio = min(ratios, key=lambda ratio: abs(float(ratio) - aspect_ratio))
    return buckets[closest_ratio_id], float(closest_ratio)


def generate_crop_size_list(base_size=256, patch_size=32, max_ratio=4.0):
    """generate crop size list

    Args:
        base_size (int, optional): the base size for generate bucket. Defaults to 256.
        patch_size (int, optional): the stride to generate bucket. Defaults to 32.
        max_ratio (float, optional): th max ratio for h or w based on base_size . Defaults to 4.0.

    Returns:
        list: generate crop size list
    """
    num_patches = round((base_size / patch_size) ** 2)
    assert max_ratio >= 1.0
    crop_size_list = []
    wp, hp = num_patches, 1
    while wp > 0:
        if max(wp, hp) / min(wp, hp) <= max_ratio:
            crop_size_list.append((wp * patch_size, hp * patch_size))
        if (hp + 1) * wp <= num_patches:
            hp += 1
        else:
            wp -= 1
    return crop_size_list


def align_floor_to(value, alignment):
    """align hight, width according to alignment

    Args:
        value (int): height or width
        alignment (int): target alignment factor

    Returns:
        int: the aligned value
    """
    return int(math.floor(value / alignment) * alignment)
