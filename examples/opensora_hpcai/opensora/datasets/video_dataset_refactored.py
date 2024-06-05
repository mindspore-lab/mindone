import csv
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from mindspore.dataset.transforms import Compose
from mindspore.dataset.vision import CenterCrop, Inter, Normalize

from .bucket import Bucket
from .transforms import BucketResizeCrop, Resize

# FIXME: remove in future when mindone is ready for install
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))
from mindone.data import BaseDataset
from mindone.data.video_reader import VideoReader

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
        vae_downsample_rate: int = 8,
        vae_scale_factor: float = 0.18215,
        sample_n_frames: int = 16,
        sample_stride: int = 4,
        frames_mask_generator: Optional[Callable[[int], np.ndarray]] = None,
        buckets: Optional[Bucket] = None,
        *,
        output_columns: List[str],
    ):
        self._data = self._read_data(video_folder, csv_path, text_emb_folder, vae_latent_folder)
        self._frames = sample_n_frames
        self._stride = sample_stride
        self._min_length = (self._frames - 1) * self._stride + 1
        self._text_emb_folder = text_emb_folder
        self._vae_latent_folder = vae_latent_folder
        self._vae_downsample_rate = vae_downsample_rate
        self._vae_scale_factor = vae_scale_factor
        self._fmask_gen = frames_mask_generator
        self._buckets = buckets

        self.output_columns = output_columns
        if self._buckets is not None:
            assert vae_latent_folder is None, "`vae_latent_folder` is not supported with bucketing"
            self.output_columns += ["bucket_id"]  # pass bucket id information to transformations

        # prepare replacement data in case the loading of a sample fails
        self._prev_ok_sample = self._get_replacement()
        self._require_update_prev = False

    @staticmethod
    def _read_data(
        data_dir: str, csv_path: str, text_emb_folder: Optional[str] = None, vae_latent_folder: Optional[str] = None
    ) -> List[dict]:
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
            if "fps" not in data:  # cache FPS for further iterations
                with VideoReader(data["video"]) as reader:
                    data["fps"] = self._data[idx]["fps"] = reader.fps
            data["fps"] = np.array(data["fps"], dtype=np.float32)

            with np.load(data["vae_latent"]) as vae_latent_data:
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

        if "caption" in self.output_columns and not self._text_emb_folder:
            if tokenizer is None:
                raise RuntimeError("Please provide a tokenizer for text data in `train_transforms()`.")
            transforms.append({"operations": [tokenizer], "input_columns": ["caption"]})

        return transforms
