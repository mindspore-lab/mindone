import os
import sys

sys.path.append("../stable_diffusion_v2")

import cv2
import numpy as np
from conditions.canny.canny_detector import CannyDetector
from conditions.utils import HWC3, resize_image
from PIL import Image

import mindspore as ms


def get_control(args, num_samples, resolution):
    image = cv2.imread(args.control_image_path)
    input_image = np.array(image, dtype=np.uint8)
    img = resize_image(HWC3(input_image), resolution)
    # img = resize_image(HWC3(input_image), min(input_image.shape[:2]))
    if args.controlnet_mode == "canny":
        apply_canny = CannyDetector()
        detected_map = apply_canny(img, args.low_threshold, args.high_threshold)
        detected_map = HWC3(detected_map)
    else:
        raise NotImplementedError(f"mode {args.controlnet_mode} not supported")

    os.makedirs(os.path.join(args.save_path), exist_ok=True)
    Image.fromarray(detected_map).save(os.path.join(args.save_path, "detected_map.png"))

    control = detected_map.copy().astype(np.float32) / 255.0
    control = np.transpose(control, (2, 0, 1))
    control = np.stack([control for _ in range(num_samples)], axis=0).astype(np.float16)
    control = ms.Tensor(control, ms.float16)
    control = ms.ops.concat([control] * 2, axis=0)
    H, W = control.shape[-2:]
    return control, H, W
