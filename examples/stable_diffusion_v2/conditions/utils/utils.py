import os

import cv2
import numpy as np

annotator_ckpts_path = os.path.join(os.path.dirname(__file__), "ckpts")


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def resize_image(image: np.ndarray, resolution: int) -> np.ndarray:
    h, w = image.shape[:2]
    k = resolution / min(h, w)
    h = int(np.round(h * k / 64.0)) * 64
    w = int(np.round(w * k / 64.0)) * 64
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
