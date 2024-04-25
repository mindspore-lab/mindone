import numbers
import os
from typing import List, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from mindone.data.video_reader import VideoReader

VID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


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

        return (video - np.array([0.5, 0.5, 0.5])) / np.array([0.5, 0.5, 0.5])  # Normalize

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

        return (image - np.array([0.5, 0.5, 0.5])) / np.array([0.5, 0.5, 0.5])  # Normalize

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


def process_mask_strategies(
    mask_strategies: List[Union[str, None]]
) -> List[Union[List[List[Union[int, float]]], None]]:
    default_strategy = [1, 0, 0, 0, 1, 0.0]
    processed = []
    for mst in mask_strategies:  # iterate over all samples
        if mst is None:
            processed.append(None)
        else:
            substrategies = []
            for substrategy in mst.split(";"):  # iterate over strategies for each loop
                substrategy = substrategy.split(",")
                assert 1 <= len(substrategy) <= 6, f"Invalid mask strategy: {substrategy}"
                # the first 5 elements are indexes => int, the last one is the edit ratio => float
                substrategy = [int(s) if i < 5 else float(s) for i, s in enumerate(substrategy)]
                substrategies.append(substrategy + default_strategy[len(substrategy) :])
            processed.append(substrategies)
    return processed


def apply_mask_strategy(
    z: np.ndarray,
    references: List[List[np.ndarray]],
    mask_strategies: List[Union[List[Union[int, float]], None]],
    loop_i: int,
) -> Tuple[np.ndarray, np.ndarray]:
    masks = np.ones((z.shape[0], z.shape[2]), dtype=np.float32)
    for batch_id, mask_strategy in enumerate(mask_strategies):
        if mask_strategy is not None:
            for mst in mask_strategy:
                loop_id, ref_id, ref_start, target_start, length, edit_ratio = mst
                if loop_id == loop_i:
                    ref = references[batch_id][ref_id]
                    if ref_start < 0:
                        ref_start = ref.shape[1] + ref_start  # ref: [C, T, H, W]
                    if target_start < 0:
                        target_start = z.shape[2] + target_start  # z: [B, C, T, H, W]
                    z[batch_id, :, target_start : target_start + length] = ref[:, ref_start : ref_start + length]
                    masks[batch_id, target_start : target_start + length] = edit_ratio
    return z, masks


def process_prompts(prompts: List[str], num_loop: int) -> List[List[str]]:
    ret_prompts = []
    for prompt in prompts:
        if prompt.startswith("|0|"):
            prompt_list = prompt.split("|")[1:]
            text_list = []
            for i in range(0, len(prompt_list), 2):
                start_loop = int(prompt_list[i])
                text = prompt_list[i + 1]
                end_loop = int(prompt_list[i + 2]) if i + 2 < len(prompt_list) else num_loop
                text_list.extend([text] * (end_loop - start_loop))
            assert len(text_list) == num_loop, f"Prompt loop mismatch: {len(text_list)} != {num_loop}"
            ret_prompts.append(text_list)
        else:
            ret_prompts.append([prompt] * num_loop)
    return ret_prompts
