"""Prepare conditional images: transform, normalization, and saving"""
import os

import albumentations
import numpy as np
from PIL import Image


def load_rgb_images(image_paths):
    assert isinstance(image_paths, list) and len(image_paths) > 0, "image paths must be a non-empty list of strings"
    return [Image.open(path).convert("RGB") for path in image_paths]


def transform_conditional_images(image_paths, H, W, random_crop=True, normalize=True, save_dir=None):
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    image_paths = list(image_paths)
    images = load_rgb_images(image_paths)
    if random_crop:
        cropper = albumentations.RandomResizedCrop(H, W, 1.0, ratio=(W / H, W / H))
    else:
        cropper = albumentations.CenterCrop(height=H, width=W)

    if normalize:

        def image_norm(image):
            image = image.mean(dim=0, keepdim=True).repeat(3, 1, 1)
            image -= image.min()
            image /= image.max()
            return image

    else:
        image_norm = lambda x: x.astype(np.float32) / 255

    controlnet_images = [
        image_norm(cropper(image=np.array(img).astype(np.uint8))["image"].transpose(2, 0, 1)) for img in images
    ]  # (c, h, w)

    if save_dir is not None:
        assert os.path.exists(save_dir), f"save_dir {save_dir} does not exist!"
        os.makedirs(os.path.join(save_dir, "control_images"), exist_ok=True)
        my_save_dir = os.path.join(save_dir, "control_images")
        existing_files = [f for f in os.listdir(my_save_dir) if os.path.isfile(os.path.join(my_save_dir, f))]
        for i, image in enumerate(controlnet_images, len(existing_files)):
            Image.fromarray((255.0 * (image.transpose(1, 2, 0))).astype(np.uint8)).save(
                f"{save_dir}/control_images/{i}.png"
            )

    controlnet_images = np.expand_dims(np.stack(controlnet_images), axis=0)
    return controlnet_images
