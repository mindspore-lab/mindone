import logging
import math
import os
import random
import sys

import numpy as np
from decord import VideoReader, cpu

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
