import matplotlib
import numpy as np

import mindspore.dataset.vision as vision
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.dataset.vision import Inter


def colorize_depth_maps(depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None):
    """
    Colorize depth maps.
    """
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    if isinstance(depth_map, Tensor):
        depth = depth_map.asnumpy().squeeze()
    elif isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()

    # reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # colorize
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    # deal with valid_mask
    if valid_mask is not None:
        if isinstance(depth_map, Tensor):
            valid_mask = valid_mask.asnumpy()
        valid_mask = valid_mask.squeeze()
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    # change back to Tensor if input is Tensor
    if isinstance(depth_map, Tensor):
        img_colored = Tensor(img_colored_np, mstype.float32)
    elif isinstance(depth_map, np.ndarray):
        img_colored = img_colored_np

    return img_colored


def chw2hwc(chw):
    """
    Convert CHW to HWC format.
    """
    assert 3 == len(chw.shape)
    if isinstance(chw, Tensor):
        hwc = ops.Transpose()(chw, (1, 2, 0))
    elif isinstance(chw, np.ndarray):
        hwc = np.moveaxis(chw, 0, -1)
    return hwc


def resize_max_res(
    img: np.ndarray,
    max_edge_resolution: int,
    resample_method: str = "bilinear",
    stick_size64: bool = False,
) -> np.ndarray:
    """
    Resize image to limit maximum edge length while keeping aspect ratio.
    """
    assert len(img.shape) == 4, f"Invalid input shape {img.shape}"

    original_height, original_width = img.shape[1], img.shape[2]

    downscale_factor = min(max_edge_resolution / original_width, max_edge_resolution / original_height)

    new_width = int(original_width * downscale_factor)
    new_height = int(original_height * downscale_factor)

    if stick_size64:
        new_width = int(round(new_width / 64, 0)) * 64
        new_height = int(round(new_height / 64, 0)) * 64

    resample_method_dict = {
        "bilinear": Inter.BILINEAR,
        "bicubic": Inter.BICUBIC,
        "nearest": Inter.NEAREST,
        "nearest-exact": Inter.NEAREST,
    }
    resample_method = resample_method_dict.get(resample_method, None)

    resized = []
    for i in range(img.shape[0]):
        resized.append(vision.Resize((new_height, new_width), resample_method)(img[i]))
    resized_img = np.stack(resized, axis=0)

    return resized_img


def get_tv_resample_method(method_str: str):
    """
    Get the resample method for MindSpore.
    """
    resample_method_dict = {
        "bilinear": "bilinear",
        "bicubic": "bicubic",
        "nearest": "nearest-exact",
        "nearest-exact": "nearest-exact",
    }
    resample_method = resample_method_dict.get(method_str, None)
    if resample_method is None:
        raise ValueError(f"Unknown resampling method: {method_str}")
    else:
        return resample_method
