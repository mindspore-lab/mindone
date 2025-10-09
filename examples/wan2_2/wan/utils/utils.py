# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import binascii
import logging
import math
import os
import os.path as osp
import pathlib
import shutil
import subprocess
from typing import BinaryIO, Dict, List, Optional, Tuple, Union

import imageio
import ml_dtypes
import numpy as np
import tqdm
from PIL import Image

import mindspore as ms
import mindspore.dataset.vision.py_transforms_util as py_transforms_util
import mindspore.mint as mint

__all__ = ["save_video", "save_image", "str2bool", "load_pth"]


def _make_grid(
    tensor: Union[ms.Tensor, List[ms.Tensor]],
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
    if not ms.is_tensor(tensor):
        if isinstance(tensor, list):
            for t in tensor:
                if not ms.is_tensor(t):
                    raise TypeError(f"tensor or list of tensors expected, got a list containing {type(t)}")
        else:
            raise TypeError(f"tensor or list of tensors expected, got {type(tensor)}")

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = mint.stack(tensor, dim=0)

    if len(tensor.shape) == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if len(tensor.shape) == 3:  # single image
        if tensor.shape[0] == 1:  # if single-channel, convert to 3-channel
            tensor = mint.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if len(tensor.shape) == 4 and tensor.shape[1] == 1:  # single-channel images
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
        raise TypeError("tensor should be of type mindspore.Tensor")
    if tensor.shape[0] == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[2] + padding), int(tensor.shape[3] + padding)
    num_channels = tensor.shape[1]
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            # Tensor.copy_() is a valid method but seems to be missing from the stubs
            # https://pytorch.org/docs/stable/tensors.html#ms.Tensor.copy_
            grid.narrow(1, y * height + padding, height - padding).narrow(  # type: ignore[attr-defined]
                2, x * width + padding, width - padding
            ).copy_(tensor[k])
            k = k + 1
    return grid


def _save_image(
    tensor: Union[ms.Tensor, List[ms.Tensor]],
    fp: Union[str, pathlib.Path, BinaryIO],
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

    grid = _make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to(ms.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)


def rand_name(length: int = 8, suffix: str = "") -> str:
    name = binascii.b2a_hex(os.urandom(length)).decode("utf-8")
    if suffix:
        if not suffix.startswith("."):
            suffix = "." + suffix
        name += suffix
    return name


def merge_video_audio(video_path: str, audio_path: str) -> None:
    """
    Merge the video and audio into a new video, with the duration set to the shorter of the two,
    and overwrite the original video file.

    Parameters:
    video_path (str): Path to the original video file
    audio_path (str): Path to the audio file
    """
    # set logging
    logging.basicConfig(level=logging.INFO)

    # check
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"video file {video_path} does not exist")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"audio file {audio_path} does not exist")

    base, ext = os.path.splitext(video_path)
    temp_output = f"{base}_temp{ext}"

    try:
        # create ffmpeg command
        command = [
            "ffmpeg",
            "-y",  # overwrite
            "-i",
            video_path,
            "-i",
            audio_path,
            "-c:v",
            "copy",  # copy video stream
            "-c:a",
            "aac",  # use AAC audio encoder
            "-b:a",
            "192k",  # set audio bitrate (optional)
            "-map",
            "0:v:0",  # select the first video stream
            "-map",
            "1:a:0",  # select the first audio stream
            "-shortest",  # choose the shortest duration
            temp_output,
        ]

        # execute the command
        logging.info("Start merging video and audio...")
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # check result
        if result.returncode != 0:
            error_msg = f"FFmpeg execute failed: {result.stderr}"
            logging.error(error_msg)
            raise RuntimeError(error_msg)

        shutil.move(temp_output, video_path)
        logging.info(f"Merge completed, saved to {video_path}")

    except Exception as e:
        if os.path.exists(temp_output):
            os.remove(temp_output)
        logging.error(f"merge_video_audio failed with error: {e}")


def save_video(
    tensor: ms.Tensor,
    save_file: Optional[str] = None,
    fps: int = 30,
    suffix: str = ".mp4",
    nrow: int = 8,
    normalize: bool = True,
    value_range: Tuple[float, float] = (-1, 1),
) -> None:
    # cache file
    cache_file = osp.join("/tmp", rand_name(suffix=suffix)) if save_file is None else save_file

    # save to cache
    try:
        # preprocess
        tensor = tensor.clamp(min(value_range), max(value_range))
        tensor = mint.stack(
            [_make_grid(u, nrow=nrow, normalize=normalize, value_range=value_range) for u in tensor.unbind(2)],
            dim=1,
        ).permute(1, 2, 3, 0)
        tensor = (tensor * 255).type(ms.uint8)

        # write video
        writer = imageio.get_writer(cache_file, fps=fps, codec="libx264", quality=8)
        for frame in tensor.numpy():
            writer.append_data(frame)
        writer.close()
    except Exception as e:
        logging.info(f"save_video failed, error: {e}")


def save_image(
    tensor: ms.Tensor,
    save_file: str,
    nrow: int = 8,
    normalize: bool = True,
    value_range: Tuple[float, float] = (-1, 1),
) -> Optional[str]:
    # cache file
    suffix = osp.splitext(save_file)[1]
    if suffix.lower() not in [".jpg", ".jpeg", ".png", ".tiff", ".gif", ".webp"]:
        suffix = ".png"

    # save to cache
    try:
        tensor = tensor.clamp(min(value_range), max(value_range))
        _save_image(tensor, save_file, nrow=nrow, normalize=normalize, value_range=value_range)
        return save_file
    except Exception as e:
        logging.info(f"save_image failed, error: {e}")


def str2bool(v: Union[str, bool]) -> bool:
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


def masks_like(
    tensor: List[ms.Tensor],
    zero: bool = False,
    generator: Optional[ms.Generator] = None,
    p: float = 0.2,
) -> Tuple[List[ms.Tensor], List[ms.Tensor]]:
    assert isinstance(tensor, list)
    out1 = [mint.ones(u.shape, dtype=u.dtype) for u in tensor]

    out2 = [mint.ones(u.shape, dtype=u.dtype) for u in tensor]

    if zero:
        if generator is not None:
            for u, v in zip(out1, out2):
                random_num = mint.rand(1, generator=generator).item()
                if random_num < p:
                    u[:, 0] = mint.normal(mean=-3.5, std=0.5, size=(1,), generator=generator).expand_as(u[:, 0]).exp()
                    v[:, 0] = mint.zeros_like(v[:, 0])
                else:
                    u[:, 0] = u[:, 0]
                    v[:, 0] = v[:, 0]
        else:
            for u, v in zip(out1, out2):
                u[:, 0] = mint.zeros_like(u[:, 0])
                v[:, 0] = mint.zeros_like(v[:, 0])

    return out1, out2


def best_output_size(w: int, h: int, dw: int, dh: int, expected_area: int) -> Tuple[int, int]:
    # float output size
    ratio = w / h
    ow = (expected_area * ratio) ** 0.5
    oh = expected_area / ow

    # process width first
    ow1 = int(ow // dw * dw)
    oh1 = int(expected_area / ow1 // dh * dh)
    assert ow1 % dw == 0 and oh1 % dh == 0 and ow1 * oh1 <= expected_area
    ratio1 = ow1 / oh1

    # process height first
    oh2 = int(oh // dh * dh)
    ow2 = int(expected_area / oh2 // dw * dw)
    assert oh2 % dh == 0 and ow2 % dw == 0 and ow2 * oh2 <= expected_area
    ratio2 = ow2 / oh2

    # compare ratios
    if max(ratio / ratio1, ratio1 / ratio) < max(ratio / ratio2, ratio2 / ratio):
        return ow1, oh1
    else:
        return ow2, oh2


def load_pth(pth_path: str) -> Dict[str, ms.Parameter]:
    import torch

    torch_data = torch.load(pth_path, map_location="cpu")
    mindspore_data = dict()
    for name, value in tqdm.tqdm(torch_data.items(), desc="converting to MindSpore format"):
        if value.dtype == torch.bfloat16:
            mindspore_data[name] = ms.Parameter(
                ms.Tensor(value.view(dtype=torch.uint16).numpy().view(ml_dtypes.bfloat16))
            )
        else:
            mindspore_data[name] = ms.Parameter(ms.from_numpy(value.numpy()))
    return mindspore_data


def to_tensor(pic: Union[Image.Image, np.ndarray]) -> ms.Tensor:
    tensor = py_transforms_util.to_tensor(pic, np.float32)
    tensor = ms.tensor(tensor)
    return tensor
