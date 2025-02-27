# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import binascii
import logging
import os
import os.path as osp

import imageio
import ml_dtypes
import torch
import torchvision
import tqdm

import mindspore as ms
from mindspore import Parameter, Tensor

__all__ = ["cache_video", "cache_image", "str2bool", "load_pth"]

logger = logging.getLogger(__name__)


def rand_name(length=8, suffix=""):
    name = binascii.b2a_hex(os.urandom(length)).decode("utf-8")
    if suffix:
        if not suffix.startswith("."):
            suffix = "." + suffix
        name += suffix
    return name


def cache_video(tensor, save_file=None, fps=30, suffix=".mp4", nrow=8, normalize=True, value_range=(-1, 1), retry=5):
    # cache file
    cache_file = osp.join("/tmp", rand_name(suffix=suffix)) if save_file is None else save_file

    # save to cache
    error = None
    for _ in range(retry):
        try:
            # preprocess
            tensor = tensor.clamp(min(value_range), max(value_range))
            tensor = torch.stack(
                [
                    torchvision.utils.make_grid(u, nrow=nrow, normalize=normalize, value_range=value_range)
                    for u in tensor.unbind(2)
                ],
                dim=1,
            ).permute(1, 2, 3, 0)
            tensor = (tensor * 255).type(torch.uint8).cpu()

            # write video
            writer = imageio.get_writer(cache_file, fps=fps, codec="libx264", quality=8)
            for frame in tensor.numpy():
                writer.append_data(frame)
            writer.close()
            return cache_file
        except Exception as e:
            logger.warning(e)
            continue
    else:
        print(f"cache_video failed, error: {error}", flush=True)
        return None


def cache_image(tensor, save_file, nrow=8, normalize=True, value_range=(-1, 1), retry=5):
    # cache file
    suffix = osp.splitext(save_file)[1]
    if suffix.lower() not in [".jpg", ".jpeg", ".png", ".tiff", ".gif", ".webp"]:
        suffix = ".png"

    # save to cache
    for _ in range(retry):
        try:
            tensor = tensor.clamp(min(value_range), max(value_range))
            torchvision.utils.save_image(tensor, save_file, nrow=nrow, normalize=normalize, value_range=value_range)
            return save_file
        except Exception as e:
            logger.warning(e)
            continue


def str2bool(v):
    """
    Convert a string to a boolean.

    Supported true values: 'yes', 'true', 't', 'y', '1'
    Supported false values: 'no', 'false', 'f', 'n', '0'

    Args:
        v (str): String to convert.

    Returns:
        bool: Converted boolean value.

    Raises:
        argparse.ArgumentTypeError: If the value cannot be converted to boolean.
    """
    if isinstance(v, bool):
        return v
    v_lower = v.lower()
    if v_lower in ("yes", "true", "t", "y", "1"):
        return True
    elif v_lower in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected (True/False)")


def load_pth(pth_path: str, dtype: ms.Type = ms.bfloat16):
    logger.info(f"Loading Pytorch ckpt from {pth_path}.")
    torch_data = torch.load(pth_path, map_location="cpu")
    mindspore_data = dict()
    for name, value in tqdm.tqdm(torch_data.items(), desc="convert to Mindspore Format"):
        if value.dtype == torch.bfloat16:
            mindspore_data[name] = Parameter(
                Tensor(value.view(dtype=torch.uint16).numpy().view(ml_dtypes.bfloat16), dtype=dtype)
            )
        else:
            mindspore_data[name] = Parameter(Tensor(value.numpy(), dtype=dtype))
    return mindspore_data
