import math
from typing import Dict, List, Union

import cv2
import numpy as np

from .sample_mappers import Rescaler, Transpose


class BatchedResize:
    def __init__(self, image_key: str = "image"):
        self.image_key = image_key

    def __call__(self, samples: List[Dict], target_size: Union[int, List] = 256):
        size = [target_size, target_size] if isinstance(target_size, int) else list(target_size)
        for s in samples:
            img = s[self.image_key]
            img = cv2.resize(img, size)
            s[self.image_key] = img

        return samples


class BatchedRescaler:
    def __init__(self, *args, **kwargs):
        self.rescaler_op = Rescaler(*args, **kwargs)

    def __call__(self, samples: List[Dict], **kwargs) -> List[Dict]:
        for i in range(len(samples)):
            samples[i] = self.rescaler_op(samples[i])
        return samples


class BatchedTranspose:
    def __init__(self, *args, **kwargs):
        self.transpose_op = Transpose(*args, **kwargs)

    def __call__(self, samples: List[Dict], **kwargs) -> List[Dict]:
        for i in range(len(samples)):
            samples[i] = self.transpose_op(samples[i])
        return samples


class BatchedResizedAndRandomCrop:
    def __init__(self, image_key: str = "image"):
        self.image_key = image_key

    def __call__(self, samples: List[Dict], target_size: Union[int, List] = 256):
        target_size = [target_size, target_size] if isinstance(target_size, int) else list(target_size)
        for s in samples:
            img = s[self.image_key]
            if not isinstance(img, np.ndarray) or img.shape[2] != 3:
                raise ValueError(
                    f"{self.__class__.__name__} requires input image to be a numpy.ndarray with channels-first"
                )

            h_o, w_o = img.shape[:2]
            h_t, w_t = target_size[:2]
            origin_hw_ratio = h_o / w_o
            target_hw_ratio = h_t / w_t
            if origin_hw_ratio >= target_hw_ratio:
                size = (math.ceil(w_t * origin_hw_ratio), w_t)
                delta_h = size[0] - h_t
                assert delta_h >= 0
                top, left = np.random.randint(0, delta_h + 1), 0
            else:
                size = (h_t, math.ceil(h_t / origin_hw_ratio))
                delta_w = size[1] - w_t
                assert delta_w >= 0
                top, left = 0, np.random.randint(0, delta_w + 1)

            img = cv2.resize(img, size)
            img = img[top : top + h_t, left : left + w_t, :]

            s[self.image_key] = img
            s["crop_coords_top_left"] = np.array([top, left])
            s["target_size_as_tuple"] = np.array([h_t, w_t])

        return samples
