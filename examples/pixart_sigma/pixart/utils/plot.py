import logging
import os
from typing import List, Union

import cv2
import numpy as np
from PIL import Image

all = ["image_grid", "resize_and_crop_tensor", "save_outputs"]

logger = logging.getLogger(__name__)


def image_grid(imgs: Union[List[Union[Image.Image, np.ndarray]], np.ndarray], ncols: int = 1) -> Image.Image:
    if (isinstance(imgs, list) and len(imgs) == 1) or (isinstance(imgs, np.ndarray) and imgs.shape[0] == 1):
        img = imgs[0]
        if isinstance(img, np.ndarray):
            img = Image.fromarray((img * 255).astype(np.uint8))
        return img

    imgs = [Image.fromarray((x * 255).astype(np.uint8)) if isinstance(x, np.ndarray) else x for x in imgs]

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
        samples = np.stack([cv2.resize(x, (resized_width, resized_height)) for x in samples], axis=0)

        # Center Crop
        start_x = (resized_width - new_width) // 2
        end_x = start_x + new_width
        start_y = (resized_height - new_height) // 2
        end_y = start_y + new_height
        samples = samples[:, start_y:end_y, start_x:end_x, :]

    return samples


def save_outputs(
    samples: np.ndarray,
    filename: str = "sample.png",
    output_dir: str = "./output",
    imagegrid: bool = False,
    grid_cols: int = 1,
) -> None:
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    if not imagegrid and samples.shape[0] != 1:
        # batch visualization
        name, ext = os.path.splitext(filename)
        for i in range(samples.shape[0]):
            filepath = os.path.join(output_dir, f"{name}_{i}{ext}")
            img = Image.fromarray((samples[i] * 255).astype(np.uint8))
            img.save(filepath)
            logger.info(f"save to {filepath}.")
    else:
        img = image_grid(samples, ncols=grid_cols)
        filepath = os.path.join(output_dir, filename)
        img.save(filepath)
        logger.info(f"save to {filepath}.")
