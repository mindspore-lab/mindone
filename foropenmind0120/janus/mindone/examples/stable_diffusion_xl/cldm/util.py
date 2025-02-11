import os
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

import mindspore as ms


class CannyDetector:
    def __call__(self, img, low_threshold, high_threshold):
        return cv2.Canny(img, low_threshold, high_threshold)


def get_control(args, num_samples, min_size=1024, save_detected_map=False):
    img = load_control_image(args.control_image_path, min_size=min_size)

    if args.controlnet_mode == "canny":
        detected_map = CannyDetector()(img, args.low_threshold, args.high_threshold)
    elif args.controlnet_mode == "raw":
        # use the image itself as control signal
        detected_map = img
    else:
        raise NotImplementedError(f"mode {args.controlnet_mode} not supported")

    if save_detected_map:
        os.makedirs(args.save_path, exist_ok=True)
        Image.fromarray(detected_map).save(os.path.join(args.save_path, "detected_map.png"))

    control = detected_map.astype(np.float32) / 255.0
    control = np.transpose(control, (2, 0, 1))[None, ...]
    control = np.tile(control, (num_samples, 1, 1, 1))
    control = np.concatenate([control, control], axis=0)
    H, W = control.shape[-2:]
    control = ms.Tensor(control, dtype=ms.float32)
    return control, H, W


def load_control_image(image: str, scale_factor: int = 8, min_size: int = 1024) -> np.ndarray:
    image = Image.open(image)
    image = ImageOps.exif_transpose(image)  # type: Image.Image
    image = image.convert("RGB")
    w, h = image.size
    w, h = _cal_size(w, h, min_size=min_size)
    w, h = (x - x % scale_factor for x in (w, h))
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image, dtype=np.float32)
    return image


def _cal_size(w: int, h: int, min_size: int = 1024) -> Tuple[int, int]:
    if w < h:
        new_h = round(h / w * min_size)
        new_w = min_size
    else:
        new_w = round(w / h * min_size)
        new_h = min_size
    return new_w, new_h
