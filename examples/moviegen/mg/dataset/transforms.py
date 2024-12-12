from typing import Optional, Tuple

import cv2
import numpy as np


class ResizeCrop:
    def __init__(self, size: Optional[Tuple[int, int]] = None, interpolation=cv2.INTER_LINEAR):
        self._size = size
        self._inter = interpolation

    def __call__(self, x: np.ndarray, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        h, w = x.shape[-3:-1]  # support images and videos
        th, tw = size or self._size
        scale = max(th / h, tw / w)
        if scale != 1:  # resize
            if x.ndim == 3:  # if image
                x = cv2.resize(x, None, fx=scale, fy=scale, interpolation=self._inter)
            else:  # if video
                x = np.array([cv2.resize(i, None, fx=scale, fy=scale, interpolation=self._inter) for i in x])
        if x.shape[-3:-1] != (th, tw):  # crop
            i, j = round((x.shape[-3] - th) / 2.0), round((x.shape[-2] - tw) / 2.0)
            x = x[..., i : i + th, j : j + tw, :]
        return x
