# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import binascii
import logging
import math
import os
import os.path as osp
from typing import List, Optional, Tuple, Union

import imageio
import ml_dtypes
import numpy as np
import torch
import tqdm
from PIL import Image

import mindspore as ms
import mindspore.mint as mint
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
    tensor = tensor.float()

    # cache file
    cache_file = osp.join("/tmp", rand_name(suffix=suffix)) if save_file is None else save_file

    # save to cache
    error = None
    for _ in range(retry):
        try:
            # preprocess
            tensor = tensor.clamp(min(value_range), max(value_range))
            tensor = mint.stack(
                [make_grid_ms(u, nrow=nrow, normalize=normalize, value_range=value_range) for u in tensor.unbind(2)],
                dim=1,
            ).permute(1, 2, 3, 0)
            tensor = (tensor * 255).type(ms.uint8)

            # write video
            writer = imageio.get_writer(cache_file, fps=fps, codec="libx264", quality=8)
            for frame in tensor.asnumpy():
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
            save_image_ms(tensor, save_file, nrow=nrow, normalize=normalize, value_range=value_range)
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
    logger.info(f"Loading PyTorch ckpt from {pth_path}.")
    torch_data = torch.load(pth_path, map_location="cpu")
    mindspore_data = dict()
    for name, value in tqdm.tqdm(torch_data.items(), desc="converting to MindSpore format"):
        if value.dtype == torch.bfloat16:
            mindspore_data[name] = Parameter(
                Tensor(value.view(dtype=torch.uint16).numpy().view(ml_dtypes.bfloat16), dtype=dtype)
            )
        else:
            mindspore_data[name] = Parameter(Tensor(value.numpy(), dtype=dtype))
    return mindspore_data


def pil2tensor(pic: Image.Image) -> ms.Tensor:
    """
    convert PIL image to mindspore.Tensor
    """
    pic = np.array(pic)
    if pic.dtype != np.uint8:
        pic = pic.astype(np.uint8)
    pic = np.transpose(pic, (2, 0, 1))  # hwc -> chw
    tensor = Tensor(pic, dtype=ms.float32)
    tensor = tensor / 255.0

    return tensor


def make_grid_ms(
    tensor: ms.Tensor,
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: float = 0.0,
) -> ms.Tensor:
    """
    Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by ``value_range``. Default: ``False``.
        value_range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Returns:
        grid (Tensor): the tensor containing grid of images.
    """
    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = mint.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.shape[0] == 1:  # if single-channel, convert to 3-channel
            tensor = mint.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.shape[1] == 1:  # single-channel images
        tensor = mint.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if value_range is not None and not isinstance(value_range, tuple):
            raise TypeError("value_range has to be a tuple (min, max) if specified. min and max are numbers")

        def norm_ip(img, low, high):
            img.clamp_(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    if not isinstance(tensor, ms.Tensor):
        raise TypeError("tensor should be of type ms Tensor")
    if tensor.shape[0] == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[2] + padding), int(tensor.shape[3] + padding)
    num_channels = tensor.shape[1]
    grid = mint.full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding).narrow(
                2, x * width + padding, width - padding
            ).copy_(tensor[k])
            k = k + 1
    return grid


def save_image_ms(
    tensor: Union[ms.Tensor, List[ms.Tensor]],
    fp: str,
    format: Optional[str] = None,
    **kwargs,
) -> None:
    """
    Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """

    grid = make_grid_ms(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to(ms.uint8).asnumpy()

    im = Image.fromarray(ndarr)
    im.save(fp, format=format)
