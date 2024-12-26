from typing import List, Tuple

import cv2
import numpy as np

from mindspore import Tensor


def resize_image(image: np.ndarray, resolution: int) -> np.ndarray:
    h, w = image.shape[:2]
    k = resolution / min(h, w)
    h = int(np.round(h * k / 64.0)) * 64
    w = int(np.round(w * k / 64.0)) * 64
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_LINEAR)


def image2tensor(image: np.ndarray) -> Tensor:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else np.expand_dims(image, 2)
    image = np.expand_dims(image.transpose((2, 0, 1)) / 255.0, axis=0).astype(np.float32)
    return Tensor(image)


def read_images(cond_paths: List[str], size: int, flags=[-1]) -> Tuple[List[Tensor], Tuple[int, int]]:
    conds = []
    image_shape = None
    assert len(cond_paths) == len(flags)
    for i, path in enumerate(cond_paths):
        cond = cv2.imread(path, flags[i])
        cond = resize_image(cond, size)

        if image_shape is None:
            image_shape = cond.shape[:2]
        # FIXME: padding?
        assert image_shape == cond.shape[:2], "All condition images must be resized to the same size."

        conds.append(image2tensor(cond))
    return conds, image_shape
