import csv
import glob
import logging
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from mindspore.dataset.transforms import Compose
from mindspore.dataset.vision import CenterCrop, Inter, Normalize

from .bucket import Bucket
from .transforms import BucketResizeCrop, Resize

# FIXME: remove in future when mindone is ready for install
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))
from mindone.data import BaseDataset
from mindone.data.video_reader import VideoReader
from mindone.models.modules.pos_embed import get_2d_sincos_pos_embed

from ..models.layers.rotary_embedding import precompute_freqs_cis

_logger = logging.getLogger(__name__)


def create_infer_transforms(target_size: Tuple[int, int], interpolation=Inter.BILINEAR):
    return Compose(
        [
            Resize(target_size, interpolation=interpolation),
            CenterCrop(target_size),
            lambda x: (x / 255.0).astype(np.float32),  # ms.ToTensor() doesn't support 4D data
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            lambda x: x[None, ...] if x.ndim == 3 else x,  # if image
            lambda x: np.transpose(x, (0, 3, 1, 2)),  # ms.HWC2CHW() doesn't support 4D data
        ]
    )


class VideoDatasetRefactored(BaseDataset):
    def __init__(
        self,
        csv_path: str,
        video_folder: str,
        text_emb_folder: Optional[str] = None,
        vae_latent_folder: Optional[str] = None,
        vae_downsample_rate: float = 8.0,
        vae_scale_factor: float = 0.18215,
        sample_n_frames: int = 16,
        sample_stride: int = 4,
        frames_mask_generator: Optional[Callable[[int], np.ndarray]] = None,
        pre_patchify: bool = False,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        embed_dim: int = 1152,
        num_heads: int = 16,
        max_target_size: int = 512,
        input_sq_size: int = 512,
        in_channels: int = 4,
        buckets: Optional[Bucket] = None,
        filter_data: bool = False,
        *,
        output_columns: List[str],
    ):
        if pre_patchify:
            if not vae_latent_folder:
                raise ValueError("`vae_latent_folder` must be provided when `pre_patchify=True`.")

        self._data = self._read_data(video_folder, csv_path, text_emb_folder, vae_latent_folder, filter_data)
        self._frames = sample_n_frames
        self._stride = sample_stride
        self._min_length = (self._frames - 1) * self._stride + 1
        self._text_emb_folder = text_emb_folder
        self._vae_latent_folder = vae_latent_folder
        self._vae_downsample_rate = vae_downsample_rate
        self._vae_scale_factor = vae_scale_factor
        self._fmask_gen = frames_mask_generator
        self._pre_patchify = pre_patchify
        self._buckets = buckets

        self.output_columns = output_columns
        if self._buckets is not None:
            assert vae_latent_folder is None, "`vae_latent_folder` is not supported with bucketing"
            self.output_columns += ["bucket_id"]  # pass bucket id information to transformations

        if self._pre_patchify:
            self._patch_size = patch_size
            assert self._patch_size[0] == 1
            self._embed_dim = embed_dim
            self._num_heads = num_heads
            self._input_sq_size = input_sq_size

            max_size = int(max_target_size / self._vae_downsample_rate)
            max_length = max_size**2 // np.prod(self._patch_size[1:]).item()
            self.pad_info = {
                "video": ([self._frames, max_length, in_channels * np.prod(self._patch_size).item()], 0),
                "spatial_pos": ([max_length, self._embed_dim], 0),
                "spatial_mask": ([max_length], 0),
                "temporal_pos": ([self._frames, self._embed_dim // self._num_heads], 0),
                "temporal_mask": ([self._frames], 0),
            }

        # check vae latent folder
        if self._vae_latent_folder is not None:
            self.num_latent_resolution = 1
            resolution_indicators = glob.glob(os.path.join(self._vae_latent_folder, "latent_*x*"))
            if len(resolution_indicators) > 1:
                self.num_latent_resolution = len(resolution_indicators)
                self.latent_resolution_prefix = resolution_indicators
                _logger.info("Multi-resolution latents detected: {}".format(self.num_latent_resolution))

        # prepare replacement data in case the loading of a sample fails
        self._prev_ok_sample = self._get_replacement()
        self._require_update_prev = False

    @staticmethod
    def _read_data(
        data_dir: str,
        csv_path: str,
        text_emb_folder: Optional[str] = None,
        vae_latent_folder: Optional[str] = None,
        filter_data: bool = False,
    ) -> List[dict]:
        def _filter_data(sample_):
            if not os.path.isfile(sample_["video"]):
                _logger.warning(f"Video not found: {sample_['video']}")
                return None
            elif "text_emb" in sample_ and not os.path.isfile(sample_["text_emb"]):
                _logger.warning(f"Text embedding not found: {sample_['text_emb']}")
                return None
            elif "vae_latent" in sample_ and not os.path.isfile(sample_["vae_latent"]):
                _logger.warning(f"Text embedding not found: {sample_['vae_latent']}")
                return None
            return sample_

        with open(csv_path, "r") as csv_file:
            try:
                data = []
                for item in csv.DictReader(csv_file):
                    sample = {**item, "video": os.path.join(data_dir, item["video"])}
                    if text_emb_folder:
                        sample["text_emb"] = os.path.join(text_emb_folder, Path(item["video"]).with_suffix(".npz"))
                    if vae_latent_folder:
                        sample["vae_latent"] = os.path.join(vae_latent_folder, Path(item["video"]).with_suffix(".npz"))
                    data.append(sample)
            except KeyError as e:
                _logger.error("CSV file requires `video` (file paths) column.")
                raise e

        if filter_data:
            with ThreadPoolExecutor(max_workers=10) as executor:
                data = [
                    item
                    for item in tqdm(executor.map(_filter_data, data), total=len(data), desc="Filtering data")
                    if item is not None
                ]

        _logger.info(f"Number of data samples: {len(data)}")
        return data

    def _get_replacement(self, max_attempts: int = 100) -> Tuple[Any, ...]:
        attempts = min(max_attempts, len(self))
        error = None
        for idx in range(attempts):
            try:
                return self._get_item(idx)
            except Exception as e:
                error = e
                _logger.debug(f"Failed to load a replacement sample: {e}")

        raise RuntimeError(f"Fail to load a replacement sample in {attempts} attempts. Error: {error}")

    def _get_item(self, idx: int) -> Tuple[Any, ...]:
        data = self._data[idx].copy()
        num_frames = self._frames

        if self._text_emb_folder:
            with np.load(data["text_emb"]) as td:
                data.update({"caption": td["text_emb"], "mask": td["mask"]})

        if self._vae_latent_folder:
            # pick a resolution randomly if there are multi-resolution latents in vae folder
            vae_latent_path = data["vae_latent"]
            if self.num_latent_resolution > 1:
                ridx = random.randint(0, self.num_latent_resolution - 1)
                vae_latent_path = vae_latent_path.replace(self._vae_latent_folder, self.latent_resolution_prefix[ridx])
            # print("D--: vae latent npz: ", vae_latent_path)

            vae_latent_data = np.load(vae_latent_path)

            # get fps from csv, or cached latents, or from original video in order
            if "fps" in data:  # cache FPS for further iterations
                data["fps"] = np.array(data["fps"], dtype=np.float32)
            elif "fps" in vae_latent_data:
                data["fps"] = np.array(vae_latent_data["fps"], dtype=np.float32)
            else:
                with VideoReader(data["video"]) as reader:
                    data["fps"] = self._data[idx]["fps"] = np.array(reader.fps, dtype=np.float32)

            latent_mean, latent_std = vae_latent_data["latent_mean"], vae_latent_data["latent_std"]
            if len(latent_mean) < self._min_length:
                raise ValueError(f"Video is too short: {data['video']}")

            start_pos = random.randint(0, len(latent_mean) - self._min_length)
            batch_index = np.linspace(start_pos, start_pos + self._min_length - 1, num_frames, dtype=int)

            latent_mean, latent_std = latent_mean[batch_index], latent_std[batch_index]
            vae_latent = latent_mean + latent_std * np.random.standard_normal(latent_mean.shape)
            data["video"] = (vae_latent * self._vae_scale_factor).astype(np.float32)

        else:
            with VideoReader(data["video"]) as reader:
                min_length = self._min_length
                if self._buckets:
                    data["bucket_id"] = self._buckets.get_bucket_id(
                        T=len(reader), H=reader.shape[1], W=reader.shape[0], frame_interval=self._stride
                    )
                    if data["bucket_id"] is None:
                        raise ValueError(
                            f"Couldn't assign a bucket to {data['video']}"
                            f" (T={len(reader)}, H={reader.shape[1]}, W={reader.shape[0]})."
                        )

                    num_frames, *_ = self._buckets.get_thw(data["bucket_id"])
                    min_length = (num_frames - 1) * self._stride + 1

                if len(reader) < min_length:
                    raise ValueError(f"Video is too short: {data['video']}")

                start_pos = random.randint(0, len(reader) - min_length)
                data["video"] = reader.fetch_frames(num=num_frames, start_pos=start_pos, step=self._stride)
                data["fps"] = np.array(reader.fps, dtype=np.float32)

        data["num_frames"] = np.array(num_frames, dtype=np.float32)

        if self._fmask_gen is not None:
            data["frames_mask"] = self._fmask_gen(num_frames)

        return tuple(data[c] for c in self.output_columns)

    def __getitem__(self, idx: int) -> Tuple[Any, ...]:
        try:
            sample = self._get_item(idx)
            if self._require_update_prev:
                self._prev_ok_sample = sample
                self._require_update_prev = False
        except Exception as e:
            _logger.warning(f"Failed to fetch sample #{idx}, the video will be replaced. Error: {e}")
            sample = self._prev_ok_sample
            self._require_update_prev = True

        return sample

    def _get_dynamic_size(self, h: int, w: int) -> Tuple[int, int]:
        if h % self._patch_size[1] != 0:
            h += self._patch_size[1] - h % self._patch_size[1]
        if w % self._patch_size[2] != 0:
            w += self._patch_size[2] - w % self._patch_size[2]
        h = h // self._patch_size[1]
        w = w // self._patch_size[2]
        return h, w

    def _patchify(self, latent: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        f, c, h, w = latent.shape

        rs = (h * w * self._vae_downsample_rate**2) ** 0.5
        ph, pw = self._get_dynamic_size(h, w)
        scale = rs / self._input_sq_size
        base_size = round((ph * pw) ** 0.5)

        nh, nw = h // self._patch_size[1], w // self._patch_size[2]

        latent = np.reshape(latent, (f, c, nh, self._patch_size[1], nw, self._patch_size[2]))
        latent = np.transpose(latent, (0, 2, 4, 1, 3, 5))  # f, nh, nw, c, patch, patch
        latent = np.reshape(latent, (f, nh * nw, -1))  # f, nh * nw, c * patch * patch

        spatial_pos = get_2d_sincos_pos_embed(self._embed_dim, nh, nw, scale=scale, base_size=base_size).astype(
            np.float32
        )

        temporal_pos = precompute_freqs_cis(f, self._embed_dim // self._num_heads).astype(np.float32)

        spatial_mask = np.ones(spatial_pos.shape[0], dtype=np.uint8)
        temporal_mask = np.ones(temporal_pos.shape[0], dtype=np.uint8)

        return latent, spatial_pos, spatial_mask, temporal_pos, temporal_mask

    def __len__(self):
        return len(self._data)

    def train_transforms(
        self, target_size: Tuple[int, int], tokenizer: Optional[Callable[[str], np.ndarray]] = None
    ) -> List[dict]:
        transforms = []
        vae_downsample_rate = self._vae_downsample_rate

        if self._buckets is not None:
            vae_downsample_rate = 1
            transforms.extend(
                [
                    {
                        "operations": BucketResizeCrop(self._buckets),
                        "input_columns": ["video", "bucket_id"],
                        "output_columns": ["video"],  # drop `bucket_id` column
                    },
                    {
                        "operations": [
                            lambda x: (x / 255.0).astype(np.float32),  # ms.ToTensor() doesn't support 4D data
                            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                            lambda x: np.transpose(x, (0, 3, 1, 2)),  # ms.HWC2CHW() doesn't support 4D data
                        ],
                        "input_columns": ["video"],
                    },
                ]
            )

        elif not self._vae_latent_folder:
            vae_downsample_rate = 1
            transforms.append(
                {
                    "operations": [
                        Resize(target_size, interpolation=Inter.BILINEAR),
                        CenterCrop(target_size),
                        lambda x: (x / 255.0).astype(np.float32),  # ms.ToTensor() doesn't support 4D data
                        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                        lambda x: np.transpose(x, (0, 3, 1, 2)),  # ms.HWC2CHW() doesn't support 4D data
                    ],
                    "input_columns": ["video"],
                }
            )

        transforms.append(
            {
                "operations": [
                    lambda video: (
                        video,  # need to return the video itself to preserve the column
                        np.array(video.shape[-2] * vae_downsample_rate, dtype=np.float32),
                        np.array(video.shape[-1] * vae_downsample_rate, dtype=np.float32),
                        np.array(video.shape[-2] / video.shape[-1], dtype=np.float32),
                    )
                ],
                "input_columns": ["video"],
                "output_columns": ["video", "height", "width", "ar"],
            }
        )

        if self._pre_patchify:
            transforms.append(
                {
                    "operations": [self._patchify],
                    "input_columns": ["video"],
                    "output_columns": ["video", "spatial_pos", "spatial_mask", "temporal_pos", "temporal_mask"],
                }
            )

        if "caption" in self.output_columns and not self._text_emb_folder:
            if tokenizer is None:
                raise RuntimeError("Please provide a tokenizer for text data in `train_transforms()`.")
            transforms.append({"operations": [tokenizer], "input_columns": ["caption"]})

        return transforms
