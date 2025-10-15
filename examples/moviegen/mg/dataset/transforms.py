import random
from typing import Callable, Tuple, Union

import cv2
import numpy as np


class ResizeCrop:
    """
    Resize and center crop the input image or video to a target size while preserving the aspect ratio.

    Args:
        size (Optional[Tuple[int, int]], optional): The target size. If None, the target size should be passed during the call.
        interpolation (cv2.InterpolationFlags, optional): The interpolation method. Defaults to cv2.INTER_LINEAR.
        preserve_orientation (bool, optional): Whether to preserve the orientation of the image/video. Defaults to True.
    """

    def __init__(
        self,
        size: Union[Tuple[int, int], Callable[[Tuple[int, int]], Tuple[int, int]]],
        interpolation: int = cv2.INTER_LINEAR,
        preserve_orientation: bool = True,
    ):
        self._size = size
        self._inter = interpolation
        self._po = preserve_orientation

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: An array containing either an image (h, w, c) or a video (f, h, w, c).

        Returns:
            A resized and cropped image or video.
        """
        h, w = x.shape[-3:-1]  # support images and videos
        th, tw = self._size if isinstance(self._size, tuple) else self._size(h, w)

        scale = max(th / h, tw / w)
        if self._po and (new_scale := max(th / w, tw / h)) < scale:  # preserve orientation
            scale = new_scale
            th, tw = tw, th

        if scale != 1:  # resize
            if x.ndim == 3:  # if image
                x = cv2.resize(x, None, fx=scale, fy=scale, interpolation=self._inter)
            else:  # if video
                x = np.array([cv2.resize(i, None, fx=scale, fy=scale, interpolation=self._inter) for i in x])

        if x.shape[-3:-1] != (th, tw):  # center crop
            i, j = round((x.shape[-3] - th) / 2.0), round((x.shape[-2] - tw) / 2.0)
            x = x[..., i : i + th, j : j + tw, :]

        return x


class HorizontalFlip:
    """
    Horizontal flip the input image or video.

    Args:
        p: The probability of horizontal flip.
    """

    def __init__(self, p: float = 0.5):
        self._p = p

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: An array containing either an image (h, w, c) or a video (f, h, w, c).
        Returns:
            An image or video flipped with probability p.
        """
        if random.random() < self._p:
            x = np.flip(x, axis=-2)
        return x
