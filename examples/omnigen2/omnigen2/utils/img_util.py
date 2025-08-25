"""
Adapted from https://github.com/pytorch/vision/blob/release/0.21/torchvision/transforms/functional.py
and https://github.com/VectorSpaceLab/OmniGen2/blob/main/omnigen2/utils/img_util.py
"""

import sys
from typing import Union

import numpy as np
from PIL import Image

import mindspore as ms
from mindspore import mint


def to_pil_image(pic: Union[ms.Tensor, np.ndarray], mode=None) -> Image.Image:
    """Convert a tensor or an ndarray to PIL Image.

    Args:
        pic (ms.Tensor or numpy.ndarray): Image to be converted to PIL Image.
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).

    .. _PIL.Image mode: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#concept-modes

    Returns:
        PIL Image: Image converted to PIL Image.
    """
    if isinstance(pic, ms.Tensor):
        if pic.ndim == 3:
            pic = mint.permute(pic, (1, 2, 0))
        pic = pic.asnumpy()
    elif not isinstance(pic, np.ndarray):
        raise TypeError(f"pic should be Tensor or ndarray. Got {type(pic)}.")

    if pic.ndim == 2:
        # if 2D image, add channel dimension (HWC)
        pic = np.expand_dims(pic, 2)
    if pic.ndim != 3:
        raise ValueError(f"pic should be 2/3 dimensional. Got {pic.ndim} dimensions.")

    if pic.shape[-1] > 4:
        raise ValueError(f"pic should not have > 4 channels. Got {pic.shape[-1]} channels.")

    npimg = pic

    if np.issubdtype(npimg.dtype, np.floating) and mode != "F":
        npimg = (npimg * 255).astype(np.uint8)

    if npimg.shape[2] == 1:
        expected_mode = None
        npimg = npimg[:, :, 0]
        if npimg.dtype == np.uint8:
            expected_mode = "L"
        elif npimg.dtype == np.int16:
            expected_mode = "I;16" if sys.byteorder == "little" else "I;16B"
        elif npimg.dtype == np.int32:
            expected_mode = "I"
        elif npimg.dtype == np.float32:
            expected_mode = "F"
        if mode is not None and mode != expected_mode:
            raise ValueError(f"Incorrect mode ({mode}) supplied for input type {np.dtype}. Should be {expected_mode}")
        mode = expected_mode

    elif npimg.shape[2] == 2:
        permitted_2_channel_modes = ["LA"]
        if mode is not None and mode not in permitted_2_channel_modes:
            raise ValueError(f"Only modes {permitted_2_channel_modes} are supported for 2D inputs")

        if mode is None and npimg.dtype == np.uint8:
            mode = "LA"

    elif npimg.shape[2] == 4:
        permitted_4_channel_modes = ["RGBA", "CMYK", "RGBX"]
        if mode is not None and mode not in permitted_4_channel_modes:
            raise ValueError(f"Only modes {permitted_4_channel_modes} are supported for 4D inputs")

        if mode is None and npimg.dtype == np.uint8:
            mode = "RGBA"
    else:
        permitted_3_channel_modes = ["RGB", "YCbCr", "HSV"]
        if mode is not None and mode not in permitted_3_channel_modes:
            raise ValueError(f"Only modes {permitted_3_channel_modes} are supported for 3D inputs")
        if mode is None and npimg.dtype == np.uint8:
            mode = "RGB"

    if mode is None:
        raise TypeError(f"Input type {npimg.dtype} is not supported")

    return Image.fromarray(npimg, mode=mode)


def resize_image(image, max_pixels, img_scale_num):
    width, height = image.shape
    cur_pixels = height * width
    ratio = (max_pixels / cur_pixels) ** 0.5
    ratio = min(ratio, 1.0)  # do not upscale input image

    new_height, new_width = (
        int(height * ratio) // img_scale_num * img_scale_num,
        int(width * ratio) // img_scale_num * img_scale_num,
    )

    image = image.resize((new_width, new_height), resample=Image.BICUBIC)
    return image


def create_collage(images: list[np.ndarray]) -> Image.Image:
    """Create a horizontal collage from a list of images."""
    max_height = max(img.shape[0] for img in images)
    total_width = sum(img.shape[1] for img in images)
    canvas = np.zeros((max_height, total_width, 3), dtype=np.float32)

    current_x = 0
    for img in images:
        h, w = img.shape[:2]
        canvas[:h, current_x : current_x + w] = img * 0.5 + 0.5
        current_x += w

    return to_pil_image(canvas)
