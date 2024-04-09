import logging
import os
import random
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

import mindspore as ms
from mindspore.dataset.transforms import Compose, vision

from mindone.models.modules.pos_embed import get_2d_sincos_pos_embed, precompute_freqs_cis_2d

_logger = logging.getLogger()

ALLOWED_FORMAT = {".mp4"}
ALLOWED_IMG_FORMAT = {".jpg", ".bmp", ".png", ".jpeg"}


class _ResizeByMaxValue:
    def __init__(self, max_size: int = 256, vae_scale: int = 8, patch_size: int = 2) -> None:
        self.max_size = max_size
        self.scale = vae_scale * patch_size

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        image_area = w * h
        max_area = self.max_size * self.max_size
        if image_area > max_area:
            ratio = max_area / image_area
            new_w = w * np.sqrt(ratio)
            new_h = h * np.sqrt(ratio)
        else:
            new_w = w
            new_h = h

        round_w, round_h = (np.round(np.array([new_w, new_h]) / self.scale) * self.scale).astype(int).tolist()
        if round_w * round_h > max_area:
            round_w, round_h = (np.floor(np.array([new_w, new_h]) / self.scale) * self.scale).astype(int).tolist()

        round_w, round_h = max(round_w, self.scale), max(round_h, self.scale)
        img = img.resize((round_w, round_h), resample=Image.BICUBIC)
        return img


class VideoWithPathIterator:
    def __init__(self, config) -> None:
        self.video_paths = self.inspect_videos(config["data_folder"])
        self.max_size = config.get("sample_size", 256)
        self.transform = self._create_transform(self.max_size, patch_size=config.get("patch_size", 2))
        self.max_frames = config.get("max_frames", 32)
        self.frame_stride = config.get("frame_stride", 4)
        self.random_crop = config.get("random_crop", False)
        if self.random_crop:
            _logger.warning("`random crop` is turned on.")

    def inspect_videos(self, root: str) -> List[str]:
        videos_info = list()

        _logger.info(f"Scanning video under `{root}`.")
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                _, ext = os.path.splitext(f)
                if ext.lower() in ALLOWED_FORMAT:
                    fpath = os.path.join(dirpath, f)
                    videos_info.append(fpath)

        if len(videos_info) == 0:
            raise RuntimeError(f"Cannot find any image under `{root}`")

        videos_info = sorted(videos_info)
        return videos_info

    def _create_transform(self, max_size: int = 256, patch_size: int = 2):
        operations = Compose(
            [
                _ResizeByMaxValue(max_size=max_size, patch_size=patch_size),
                vision.HWC2CHW(),
                vision.Normalize([127.5, 127.5, 127.5], [127.5, 127.5, 127.5], is_hwc=False),
            ]
        )
        return operations

    def __len__(self):
        return len(self.video_paths)

    def _to_pil(self, image: np.ndarray) -> Image.Image:
        image = image[..., ::-1]  # BGR -> RGB
        image = Image.fromarray(image)
        return image

    def _random_crop(self, video: np.ndarray) -> np.ndarray:
        # this is just for verification of the dynamic shape training.
        # need to be turn off once the verification is ok.
        height_ = np.random.choice(np.arange(128, video.shape[-2] + 16, 16))
        width_ = np.random.choice(np.arange(128, video.shape[-1] + 16, 16))
        video = video[:, :, :height_, :width_]
        return video

    def __getitem__(self, index):
        path = self.video_paths[index]

        vidcap = cv2.VideoCapture(path)
        success, image = vidcap.read()
        try:
            image = self.transform(self._to_pil(image))[0]
        except Exception as e:
            _logger.warning(f"Cannot read video `{path}`. ({str(e)})")
            video = np.zeros((self.frame_stride, 3, self.max_size, self.max_size), dtype=np.float32)
            return video, path
        video = [image]

        i = 0
        while success and len(video) < self.max_frames:
            success, image = vidcap.read()
            if image is None:
                break
            if i % self.frame_stride == 0:
                image = self.transform(self._to_pil(image))[0]
                video.append(image)
            i += 1

        video = np.stack(video, axis=0)

        if self.random_crop:
            video = self._random_crop(video)

        return video, path


class SKYTimeLapse(VideoWithPathIterator):
    def inspect_videos(self, root: str) -> List[List[str]]:
        videos_info = defaultdict(list)

        _logger.info(f"Scanning video under `{root}`.")
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                _, ext = os.path.splitext(f)
                if ext.lower() in ALLOWED_IMG_FORMAT:
                    fpath = os.path.join(dirpath, f)
                    videos_info[os.path.basename(dirpath)].append(fpath)

        videos_info_list = list()
        for v in videos_info.values():
            videos_info_list.append(sorted(v))

        if len(videos_info_list) == 0:
            raise RuntimeError(f"Cannot find any image under `{root}`")

        videos_info_list = sorted(videos_info_list)
        return videos_info_list

    def __getitem__(self, index):
        path = self.video_paths[index]

        read_path = [x for i, x in enumerate(path) if i % self.frame_stride == 0]
        read_path = read_path[: self.max_frames]

        video = [Image.open(x).convert("RGB") for x in read_path]
        video = [self.transform(x)[0] for x in video]
        video = np.stack(video, axis=0)

        if self.random_crop:
            video = self._random_crop(video)

        path = os.path.dirname(path[0])

        return video, path


class VideoLatentIterator:
    def __init__(self, config) -> None:
        self.latent_info = self._inspect_latent(config["data_folder"])
        self.patch_size = config.get("patch_size", 2)
        self.embed_dim = config.get("embed_dim", 1152)
        self.embed_method = config.get("embed_method", "absolute")

    def _inspect_latent(self, root: str) -> List[Dict[str, str]]:
        latent_info = list()

        _logger.info(f"Scanning numpy file under `{root}`.")
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                _, ext = os.path.splitext(f)
                if ext.lower() in ".npy":
                    fpath = os.path.join(dirpath, f)
                    latent_info.append(dict(path=fpath, label=os.path.basename(dirpath)))

        if len(latent_info) == 0:
            raise RuntimeError(f"Cannot find any image under `{root}`")

        latent_info = sorted(latent_info, key=lambda x: x["path"])
        return latent_info

    def __len__(self):
        return len(self.latent_info)

    def _random_horiztotal_flip(self, latent: np.ndarray) -> np.ndarray:
        if random.random() < 0.5:
            # perform a random horizontal flip in latent domain
            # mimic the effect of horizontal flip in image (not exactly identical)
            latent = latent[..., ::-1]
        return latent

    def _patchify(self, latent: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        f, c, h, w = latent.shape
        nh, nw = h // self.patch_size, w // self.patch_size

        latent = np.reshape(latent, (f, c, nh, self.patch_size, nw, self.patch_size))
        latent = np.transpose(latent, (0, 2, 4, 3, 5, 1))  # f, nh, nw, patch, patch, c
        latent = np.reshape(latent, (f, nh * nw, -1))  # f, nh * nw, patch * patch * c

        if self.embed_method == "rotate":
            pos = precompute_freqs_cis_2d(self.embed_dim, nh, nw).astype(np.float32)
        else:
            pos = get_2d_sincos_pos_embed(self.embed_dim, nh, nw).astype(np.float32)

        return latent, pos

    def __getitem__(self, idx):
        x = self.latent_info[idx]

        latent = np.load(x["path"])

        # N(mean, std)
        mean, std = np.split(latent, 2, axis=1)
        latent = mean + std * np.random.randn(*mean.shape).astype(mean.dtype)

        latent = self._random_horiztotal_flip(latent)
        latent, pos = self._patchify(latent)

        mask_t = np.ones(latent.shape[0], dtype=np.bool_)
        mask_s = np.ones(latent.shape[1], dtype=np.bool_)
        return latent, pos, mask_t, mask_s


def create_dataloader_video_preprocessing(
    config,
    device_num: Optional[int] = None,
    rank_id: Optional[int] = None,
):
    if config["dataset_mode"] == "video":
        dataset = VideoWithPathIterator(config)
    elif config["dataset_mode"] == "sky":
        dataset = SKYTimeLapse(config)

    dataset = ms.dataset.GeneratorDataset(
        dataset,
        column_names=["video", "path"],
        num_shards=device_num,
        shard_id=rank_id,
        num_parallel_workers=config["num_parallel_workers"],
        shuffle=config["shuffle"],
    )
    dataset = dataset.batch(1)
    return dataset


def create_dataloader_video_latent(
    config,
    tokenizer: Optional[Callable] = None,
    device_num: Optional[int] = None,
    rank_id: Optional[int] = None,
):
    dataset = VideoLatentIterator(config)
    dataset = ms.dataset.GeneratorDataset(
        dataset,
        column_names=["latent", "pos", "mask_t", "mask_s"],
        num_shards=device_num,
        shard_id=rank_id,
        num_parallel_workers=config["num_parallel_workers"],
        shuffle=config["shuffle"],
    )

    sample_size = config.get("sample_size", 256)
    patch_size = config.get("patch_size", 2)
    vae_scale = 8
    max_length = sample_size * sample_size // patch_size // patch_size // vae_scale // vae_scale
    embed_dim = config.get("embed_dim", 72)
    num_frames = config.get("num_frames", 32)
    C = 4

    pad_info = {
        "latent": ([num_frames, max_length, patch_size * patch_size * C], 0),
        "pos": ([max_length, embed_dim], 0),
        "mask_t": ([num_frames], 0),
        "mask_s": ([max_length], 0),
    }

    dataset = dataset.padded_batch(config["batch_size"], drop_remainder=True, pad_info=pad_info)
    return dataset
