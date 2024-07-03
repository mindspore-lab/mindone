from typing import List, Union

import cv2
import numpy as np
from PIL import Image


def image_grid(imgs: List[Union[Image.Image, np.ndarray]], ncols: int = 4) -> Image.Image:
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
