import logging
import os
from typing import Callable, List, Optional, Union

import cv2
import numpy as np
from PIL import Image

all = ["image_grid", "resize_and_crop_tensor", "create_save_func"]

logger = logging.getLogger(__name__)


def image_grid(imgs: Union[List[Union[Image.Image, np.ndarray]], np.ndarray], ncols: int = 1) -> Image.Image:
    """
    Inputs: List[NDArray[H, W, C]], NDArray[B, H, W, C]
    """
    if isinstance(imgs, np.ndarray) and len(imgs.shape) != 4:
        raise ValueError("`img` must be a 4-Dims array")

    if (isinstance(imgs, list) and len(imgs) == 1) or (isinstance(imgs, np.ndarray) and imgs.shape[0] == 1):
        img = imgs[0]
        if isinstance(img, np.ndarray):
            assert len(img.shape) == 3
            img = Image.fromarray((img * 255).astype(np.uint8))
        return img

    imgs = [Image.fromarray((x * 255).astype(np.uint8)) if isinstance(x, np.ndarray) else x for x in imgs]

    ncols = min(ncols, len(imgs))
    nrows = len(imgs) // ncols
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(ncols * w, nrows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % ncols * w, i // ncols * h))
    return grid


def resize_and_crop_tensor(samples: np.ndarray, new_width: int, new_height: int) -> np.ndarray:
    orig_height, orig_width = samples.shape[1], samples.shape[2]

    # Check if resizing is needed
    if orig_height != new_height or orig_width != new_width:
        ratio = max(new_height / orig_height, new_width / orig_width)
        resized_width = int(orig_width * ratio)
        resized_height = int(orig_height * ratio)

        # Resize
        samples = np.stack([cv2.resize(x, (resized_width, resized_height)) for x in samples.astype(np.float32)], axis=0)

        # Center Crop
        start_x = (resized_width - new_width) // 2
        end_x = start_x + new_width
        start_y = (resized_height - new_height) // 2
        end_y = start_y + new_height
        samples = samples[:, start_y:end_y, start_x:end_x, :]

    return samples


def create_save_func(
    filename: Optional[str] = None,
    output_dir: str = "./output",
    imagegrid: bool = False,
    grid_cols: int = 1,
) -> Callable[[np.ndarray], None]:
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    cnt = 0

    def save(samples: np.ndarray) -> None:
        nonlocal cnt

        if not imagegrid and samples.shape[0] != 1 and len(samples.shape) == 4:
            # batch visualization
            for i in range(samples.shape[0]):
                _filename = f"{cnt}.png" if filename is None else filename
                img = image_grid(samples[i : i + 1], ncols=1)
                filepath = os.path.join(output_dir, _filename)
                img.save(filepath)
                logger.info(f"save to {filepath}.")
                cnt += 1
        else:
            _filename = f"{cnt}.png" if filename is None else filename
            img = image_grid(samples, ncols=grid_cols)
            filepath = os.path.join(output_dir, _filename)
            img.save(filepath)
            logger.info(f"save to {filepath}.")
            cnt += 1

    return save
