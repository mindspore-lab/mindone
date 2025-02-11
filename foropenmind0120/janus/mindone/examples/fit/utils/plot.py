from typing import List, Union

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
