from typing import Tuple

import numpy as np

from mindspore.dataset.transforms import Compose
from mindspore.dataset.vision import CenterCrop, Inter
from mindspore.dataset.vision import Resize as MSResize

from .bucket import Bucket


class Resize:
    def __init__(self, size: Tuple[int, int], interpolation=Inter.BILINEAR):
        self._h, self._w = size
        self._inter = interpolation

    def __call__(self, x: np.ndarray) -> np.ndarray:
        img_h, img_w = x.shape[-3:-1]  # support images and videos
        scale = max(self._h / img_h, self._w / img_w)
        if scale != 1:
            x = MSResize((round(img_h * scale), round(img_w * scale)), self._inter)(x)
        return x


class BucketResizeCrop:
    def __init__(self, buckets: Bucket):
        self._transforms = {}  # is this reasonable? There are 350+ buckets
        for name, lengths in buckets.ar_criteria.items():
            self._transforms[name] = {}
            for length, ars in lengths.items():
                self._transforms[name][str(length)] = {}
                for ar, hw in ars.items():
                    self._transforms[name][str(length)][ar] = Compose(
                        [MSResize(min(hw), interpolation=Inter.BILINEAR), CenterCrop(hw)]
                    )

    def __call__(self, x, bucket_id):
        return self._transforms[bucket_id[0]][bucket_id[1]][bucket_id[2]](x)
