from typing import Tuple

import cv2
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


class ResizeAndCrop:
    """Resize an RGB image to a target size while preserving the aspect ratio and cropping it.
    Align to resize_crop_to_fill in torch. Ensure no black surrounding produced.
    """

    def __init__(self, target_height, target_width):
        super(ResizeAndCrop, self).__init__()
        self.tar_h = target_height
        self.tar_w = target_width

    def __call__(self, img):
        # Ensure the image is in RGB format
        if img.shape[2] != 3:
            raise ValueError("Input image must be in RGB format with 3 channels.")

        h, w = img.shape[:2]
        th, tw = self.tar_h, self.tar_w  # target
        rh, rw = th / h, tw / w  # ratio

        if rh > rw:
            # target image is thinner than the original image
            new_h, new_w = th, round(w * rh)
            start_y = 0
            start_x = int(round(new_w - tw) / 2.0)
        else:
            # target image is fatter than the original image
            new_h, new_w = round(h * rw), tw
            start_y = int(round(new_h - th) / 2.0)
            start_x = 0

        if rh > rw:
            new_h, new_w = th, round(w * rh)
            start_y = 0
            start_x = int(round(new_w - tw))

        # Resize the image
        # NOTE: for opensora v1.2, HD videos are mainly downsampled according to buckets. The best choice for down-sample interpolation is INTER_AREA.
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Crop the image to the target size
        cropped_img = resized_img[start_y : start_y + self.tar_h, start_x : start_x + self.tar_w]

        return cropped_img


class BucketResizeAndCrop(object):
    """According to bucket config, resize an RGB image to a target size while preserving the aspect ratio and cropping it."""

    def __init__(self, buckets):
        super().__init__()
        self._transforms = {}  # is this reasonable? There are 350+ buckets
        for name, lengths in buckets.ar_criteria.items():
            self._transforms[name] = {}
            for length, ars in lengths.items():
                self._transforms[name][str(length)] = {}
                for ar, hw in ars.items():
                    self._transforms[name][str(length)][ar] = ResizeAndCrop(hw[0], hw[1])

    def __call__(self, image, bucket_id=None):
        resized_img = self._transforms[bucket_id[0]][str(bucket_id[1])][bucket_id[2]](image)
        return resized_img
