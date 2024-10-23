import math
from typing import List, Tuple, Union

import numpy as np

from mindspore import Tensor


def get_anyres_image_grid_shape(
    image_size: Union[Tuple[int, int], List[int], Tensor, np.ndarray],
    grid_pinpoints: List[Tuple[int, int]],
    patch_size: int,
) -> Tuple[int, int]:
    if not isinstance(grid_pinpoints, list):
        raise ValueError("grid_pinpoints should be a list of tuples or lists")

    # ! VERY IMPORTANT if image_size is tensor, must convert to into tuple, otherwise it will cause wrong calculate
    if not isinstance(image_size, (list, tuple)):
        if not isinstance(image_size, (Tensor, np.ndarray)):
            raise ValueError(
                f"image_size invalid type: {type(image_size)} not valid, should be either list, tuple, np.ndarray or tensor"
            )
        image_size = image_size.tolist()

    height, width = select_best_resolution(image_size, grid_pinpoints)
    return height // patch_size, width // patch_size


def image_size_to_num_patches(
    image_size: Union[Tuple[int, int], List[int], Tensor, np.ndarray],
    grid_pinpoints: List[Tuple[int, int]],
    patch_size: int,
) -> int:
    if not isinstance(grid_pinpoints, list):
        raise ValueError("grid_pinpoints should be a list of tuples or lists")

    # ! VERY IMPORTANT if image_size is tensor, must convert to into tuple, otherwise it will cause wrong calculate
    if not isinstance(image_size, (list, tuple)):
        if not isinstance(image_size, (Tensor, np.ndarray)):
            raise ValueError(f"image_size invalid type {type(image_size)} with value {image_size}")
        image_size = image_size.tolist()

    best_resolution = select_best_resolution(image_size, grid_pinpoints)
    height, width = best_resolution
    num_patches = math.ceil(height / patch_size) * math.ceil(width / patch_size) + 1
    return num_patches


def unpad_image(tensor: Tensor, original_size: Union[Tuple[int, int], Tensor]) -> Tensor:
    original_height, original_width = original_size
    current_height, current_width = tensor.shape[1:]

    # HACK: need to be int here, since for ms, ms.int / ms.int -> ms.int
    if isinstance(original_size, Tensor):
        original_height, original_width = original_height.item(), original_width.item()

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


def select_best_resolution(
    original_size: Tuple[int, int], possible_resolutions: List[Tuple[int, int]]
) -> Tuple[int, int]:
    original_height, original_width = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for height, width in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (height, width)

    return best_fit
