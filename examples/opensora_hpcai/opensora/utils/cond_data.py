"""Prepare conditional images: transform, normalization, and saving"""
import numbers
import os
from typing import List, Tuple, Union

import albumentations
import cv2
import numpy as np
import pandas as pd
from PIL import Image

from mindone.data.video_reader import VideoReader

VID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


def load_rgb_images(image_paths):
    assert isinstance(image_paths, list) and len(image_paths) > 0, "image paths must be a non-empty list of strings"
    return [Image.open(path).convert("RGB") for path in image_paths]


def read_captions_from_csv(path, caption_column="caption"):
    df = pd.read_csv(path, usecols=[caption_column])
    captions = df[caption_column].values.tolist()
    return captions


def read_captions_from_txt(path):
    captions = []
    with open(path, "r") as fp:
        for line in fp:
            captions.append(line.strip())
    return captions


def transform_conditional_images(image_paths, H, W, random_crop=True, normalize=True, save_dir=None):
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    image_paths = list(image_paths)
    images = load_rgb_images(image_paths)
    if random_crop:
        cropper = albumentations.RandomResizedCrop(H, W, (1.0, 1.0), ratio=(W / H, W / H))
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


def _is_tensor_video_clip(clip: np.ndarray) -> bool:
    if not clip.ndim == 4:
        raise ValueError("clip should be 4D. Got %dD" % clip.ndim)

    return True


def crop(clip: np.ndarray, i, j, h, w):
    """
    Args:
        clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
    """
    if clip.ndim != 4:
        raise ValueError("clip should be a 4D tensor")
    return clip[:, i : i + h, j : j + w]


def resize_scale(clip, target_size, interpolation_mode=cv2.INTER_LINEAR):
    if len(target_size) != 2:
        raise ValueError(f"target size should be tuple (height, width), instead got {target_size}")
    H, W = clip.size(-2), clip.size(-1)
    scale_ = target_size[0] / min(H, W)

    new_clip = []
    for frame in clip:
        new_clip.append(cv2.resize(frame, None, fx=scale_, fy=scale_, interpolation=interpolation_mode))

    return np.stack(new_clip)


def resize(clip, target_size, interpolation_mode=cv2.INTER_LINEAR):
    if len(target_size) != 2:
        raise ValueError(f"target size should be tuple (height, width), instead got {target_size}")

    new_clip = []
    for frame in clip:
        new_clip.append(cv2.resize(frame, target_size, interpolation=interpolation_mode))

    return np.stack(new_clip)


def center_crop(clip, crop_size):
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    h, w = clip.size(-2), clip.size(-1)
    th, tw = crop_size
    if h < th or w < tw:
        raise ValueError("height and width must be no smaller than crop_size")

    i = int(round((h - th) / 2.0))
    j = int(round((w - tw) / 2.0))
    return crop(clip, i, j, th, tw)


def resize_crop_to_fill_video(clip, target_size):
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    h, w = clip.shape[1], clip.shape[2]
    th, tw = target_size[0], target_size[1]
    rh, rw = th / h, tw / w
    if rh > rw:
        sh, sw = th, round(w * rh)
        clip = resize(clip, (sw, sh), cv2.INTER_LINEAR)
        i = 0
        j = int(round(sw - tw) / 2.0)
    else:
        sh, sw = round(h * rw), tw
        clip = resize(clip, (sw, sh), cv2.INTER_LINEAR)
        i = int(round(sh - th) / 2.0)
        j = 0
    assert i + th <= clip.shape[1] and j + tw <= clip.shape[2]
    return crop(clip, i, j, th, tw)


def resize_crop_to_fill_image(pil_image, image_size):
    w, h = pil_image.size  # PIL is (W, H)
    th, tw = image_size
    rh, rw = th / h, tw / w
    if rh > rw:
        sh, sw = th, round(w * rh)
        image = pil_image.resize((sw, sh), Image.BICUBIC)
        i = 0
        j = int(round((sw - tw) / 2.0))
    else:
        sh, sw = round(h * rw), tw
        image = pil_image.resize((sw, sh), Image.BICUBIC)
        i = int(round((sh - th) / 2.0))
        j = 0
    arr = np.array(image)
    assert i + th <= arr.shape[0] and j + tw <= arr.shape[1]
    return Image.fromarray(arr[i : i + th, j : j + tw])


class UCFCenterCropVideo:
    """
    First scale to the specified size in equal proportion to the short edge,
    then center cropping
    """

    def __init__(
        self,
        size,
        interpolation_mode=cv2.INTER_LINEAR,
    ):
        if isinstance(size, tuple):
            if len(size) != 2:
                raise ValueError(f"size should be tuple (height, width), instead got {size}")
            self.size = size
        else:
            self.size = (size, size)

        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            torch.tensor: scale resized / center cropped video clip.
                size is (T, C, crop_size, crop_size)
        """
        clip_resize = resize_scale(clip=clip, target_size=self.size, interpolation_mode=self.interpolation_mode)
        clip_center_crop = center_crop(clip_resize, self.size)
        return clip_center_crop

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, interpolation_mode={self.interpolation_mode}"


class ResizeCrop:
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, clip):
        clip = resize_crop_to_fill_video(clip, self.size)
        return clip

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def get_transforms_video(name="center", image_size=(256, 256)):
    def transform_video(video: np.ndarray) -> np.ndarray:
        if name == "center":
            assert image_size[0] == image_size[1], "image_size must be square for center crop"
            video = UCFCenterCropVideo(image_size[0])(video)
        elif name == "resize_crop":
            video = ResizeCrop(image_size)(video)
        else:
            raise NotImplementedError(f"Transform {name} not implemented")

        video = video.astype(np.float32) / 255.0

        # Normalize
        return (video - np.array([0.5, 0.5, 0.5], dtype=np.float32)) / np.array([0.5, 0.5, 0.5], dtype=np.float32)

    return transform_video


def get_transforms_image(name="center", image_size=(256, 256)):
    def transform(pil_image: Image.Image) -> np.ndarray:
        if name == "center":
            assert image_size[0] == image_size[1], "Image size must be square for center crop"
            image = center_crop_arr(pil_image, image_size[0])
        elif name == "resize_crop":
            image = resize_crop_to_fill_image(pil_image, image_size)
        else:
            raise NotImplementedError(f"Transform {name} not implemented")

        image = np.array(image, dtype=np.float32) / 255.0

        # Normalize
        return (image - np.array([0.5, 0.5, 0.5], dtype=np.float32)) / np.array([0.5, 0.5, 0.5], dtype=np.float32)

    return transform


def read_image_from_path(path, transform=None, transform_name="center", num_frames=1, image_size=(256, 256)):
    image = pil_loader(path)
    if transform is None:
        transform = get_transforms_image(image_size=image_size, name=transform_name)
    image = transform(image)
    video = np.expand_dims(image, 0).repeat(num_frames, axis=0)
    return video.transpose(0, 3, 1, 2)  # T C H W


def read_video_from_path(path, transform=None, transform_name="center", image_size=(256, 256)):
    with VideoReader(path) as reader:
        vframes = reader.fetch_frames(num=len(reader))
    if transform is None:
        transform = get_transforms_video(image_size=image_size, name=transform_name)
    video = transform(vframes)
    video = video.transpose(0, 3, 1, 2)  # T C H W
    return video


def read_from_path(path, image_size, transform_name="center") -> np.ndarray:
    ext = os.path.splitext(path)[-1].lower()
    if ext.lower() in VID_EXTENSIONS:
        return read_video_from_path(path, image_size=image_size, transform_name=transform_name)
    else:
        assert ext.lower() in IMG_EXTENSIONS, f"Unsupported file format: {ext}"
        return read_image_from_path(path, image_size=image_size, transform_name=transform_name)


def get_references(
    reference_paths: List[Union[str, None]], image_size: Tuple[int, int]
) -> List[Union[List[np.ndarray], None]]:
    references = []
    for paths in reference_paths:  # iterate over samples
        subref = []
        if paths is not None:
            for ref in paths.split(";"):  # iterate over references for each loop
                subref.append(read_from_path(ref, image_size, transform_name="resize_crop"))
        references.append(subref)
    return references
