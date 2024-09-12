import csv
import glob
import logging
import os
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import cv2
import numpy as np
from decord import VideoReader
from tqdm import tqdm

from mindspore.dataset.transforms import Compose

from mindone.data.video_reader import VideoReader as VideoReader_CV2

from .bucket import Bucket
from .transforms import ResizeCrop

# FIXME: remove in future when mindone is ready for install
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))
from mindone.data import BaseDataset
from mindone.models.modules.pos_embed import get_2d_sincos_pos_embed

from ..models.layers.rotary_embedding import precompute_freqs_cis

_logger = logging.getLogger(__name__)


def create_infer_transforms(target_size: Tuple[int, int], interpolation=cv2.INTER_LINEAR):
    return Compose(
        [
            ResizeCrop(target_size, interpolation=interpolation),
            lambda x: x.astype(np.float32) / 127.5 - 1,
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
        sample_stride: int = 1,
        frames_mask_generator: Optional[Callable[[int], np.ndarray]] = None,
        t_compress_func: Optional[Callable[[int], int]] = None,
        pre_patchify: bool = False,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        embed_dim: int = 1152,
        num_heads: int = 16,
        max_target_size: int = 512,
        input_sq_size: int = 512,
        in_channels: int = 4,
        bucketing: bool = False,
        filter_data: bool = False,
        apply_train_transforms: bool = False,
        target_size: Optional[Tuple[int, int]] = None,
        tokenizer=None,
        video_backend: str = "cv2",
        *,
        output_columns: List[str],
    ):
        assert tokenizer is None, "tokenizer is not supported"
        if pre_patchify and vae_latent_folder is None:
            raise ValueError("`vae_latent_folder` must be provided when `pre_patchify=True`.")
        if text_emb_folder is None:
            raise NotImplementedError(
                "Text embedding during training is not supported, please provide `text_emb_folder`."
            )

        self._data = self._read_data(video_folder, csv_path, text_emb_folder, vae_latent_folder, filter_data)
        self._frames = sample_n_frames
        self._stride = sample_stride
        self._min_length = (self._frames - 1) * self._stride + 1
        self._text_emb_folder = text_emb_folder
        self._vae_latent_folder = vae_latent_folder
        self._vae_downsample_rate = vae_downsample_rate
        self._vae_scale_factor = vae_scale_factor
        self._fmask_gen = frames_mask_generator
        self._t_compress_func = t_compress_func or (lambda x: x)
        self._pre_patchify = pre_patchify
        self._bucketing = bucketing

        self.output_columns = output_columns
        if self._bucketing is not None:
            assert vae_latent_folder is None, "`vae_latent_folder` is not supported with bucketing"
            self.output_columns += ["size"]  # pass bucket id information to transformations

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

        # decord has better performance and may incur memory leak for high-resolution videos
        self.video_backend = video_backend

        self.apply_train_transforms = apply_train_transforms
        if self.apply_train_transforms:
            self.pixel_transforms = ResizeCrop(target_size, interpolation=cv2.INTER_AREA)
            if "size" in self.output_columns:
                self.output_columns.remove("size")
            assert not pre_patchify, "transforms for prepatchify not implemented yet"

        # prepare replacement data in case the loading of a sample fails
        self._prev_ok_sample = self._get_replacement() if self._bucketing is None else None
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
                _logger.error(f"CSV file requires `video` (file paths) column, but got {list(item.keys())}")
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
                _logger.debug(f"Failed to load a replacement sample: {repr(e)}")

        raise RuntimeError(f"Fail to load a replacement sample in {attempts} attempts. Error: {repr(error)}")

    def _get_item(self, idx: int, thw: Optional[Tuple[int, int, int]] = None) -> Tuple[Any, ...]:
        data = {}
        video_path = self._data[idx]["video"]
        text_emb_path = self._data[idx]["text_emb"]

        num_frames = self._frames

        if self._text_emb_folder:
            with np.load(text_emb_path) as td:
                data["caption"] = td["text_emb"]
                data["mask"] = td["mask"].astype(np.uint8)

        if self._vae_latent_folder:
            # pick a resolution randomly if there are multi-resolution latents in vae folder
            vae_latent_path = self._data[idx]["vae_latent"]
            if self.num_latent_resolution > 1:
                ridx = random.randint(0, self.num_latent_resolution - 1)
                vae_latent_path = vae_latent_path.replace(self._vae_latent_folder, self.latent_resolution_prefix[ridx])

            vae_latent_data = np.load(vae_latent_path)
            if "fps" in self._data[idx]:
                data["fps"] = np.array(self._data[idx]["fps"], dtype=np.float32)
            elif "fps" in vae_latent_data:
                data["fps"] = np.array(vae_latent_data["fps"], dtype=np.float32)
            else:
                cap = cv2.VideoCapture(video_path, apiPreference=cv2.CAP_FFMPEG)
                data["fps"] = cap.fps
                cap.release()

            latent_mean, latent_std = vae_latent_data["latent_mean"], vae_latent_data["latent_std"]
            if len(latent_mean) < self._min_length:
                raise ValueError(f"Video is too short: {data['video']}")

            start_pos = random.randint(0, len(latent_mean) - self._min_length)
            batch_index = np.linspace(start_pos, start_pos + self._min_length - 1, num_frames, dtype=int)

            latent_mean, latent_std = latent_mean[batch_index], latent_std[batch_index]
            vae_latent = latent_mean + latent_std * np.random.standard_normal(latent_mean.shape)
            video = (vae_latent * self._vae_scale_factor).astype(np.float32)

        else:
            if self.video_backend == "decord":
                reader = VideoReader(video_path)
                min_length = self._min_length
                video_length = len(reader)
                if thw is not None:
                    num_frames = thw[0]
                    min_length = (num_frames - 1) * self._stride + 1

                if len(reader) < min_length:
                    raise ValueError(f"Video is too short: {data['video']}")

                clip_length = min(video_length, min_length)
                start_pos = random.randint(0, len(reader) - clip_length)

                batch_index = np.linspace(start_pos, start_pos + clip_length - 1, num_frames, dtype=int)
                video = reader.get_batch(batch_index).asnumpy()
                data["fps"] = np.array(
                    reader.get_avg_fps(), dtype=np.float32
                )  # / self._stride  # FIXME: OS v1.1 incorrect
                del reader
            elif self.video_backend == "cv2":
                with VideoReader_CV2(video_path) as reader:
                    min_length = self._min_length
                    if thw is not None:
                        num_frames = thw[0]
                        min_length = (num_frames - 1) * self._stride + 1

                    if len(reader) < min_length:
                        raise ValueError(f"Video is too short: {video_path}")
                    start_pos = random.randint(0, len(reader) - min_length)
                    video = reader.fetch_frames(num=num_frames, start_pos=start_pos, step=self._stride)
                    data["fps"] = np.array(reader.fps, dtype=np.float32)
            else:
                # TODO: add pyav backend and test
                raise NotImplementedError

        data["num_frames"] = np.array(num_frames, dtype=np.float32)

        if self._fmask_gen is not None:
            # return frames mask with respect to the VAE's latent temporal compression
            data["frames_mask"] = self._fmask_gen(self._t_compress_func(num_frames))

        data["video"] = video

        # apply transforms on video frames here
        if self.apply_train_transforms:
            # variable resize and crop, frame-wise
            clip = self.pixel_transforms(video, thw[1:] if thw is not None else None)

            # transpose and norm, clip-wise
            clip = np.transpose(clip, (0, 3, 1, 2))
            clip = clip.astype(np.float32) / 127.5 - 1

            # additional conditions for model
            data["height"] = np.array(clip.shape[-2], dtype=np.float32)
            data["width"] = np.array(clip.shape[-1], dtype=np.float32)
            # NOTE: here ar = h / w, aligned to torch, while the common practice is w / h
            data["ar"] = np.array(clip.shape[-2] / clip.shape[-1], dtype=np.float32)
            data["video"] = clip

        final_outputs = tuple(data.pop(c) for c in self.output_columns)
        del data

        return final_outputs

    def get_bucket(self, thw: Tuple[int, int, int], sample_ids: List[int]) -> Tuple[Any, ...]:
        batch = [self._get_item(sample_id, thw) for sample_id in sample_ids]
        return tuple(np.stack(item) for item in map(list, zip(*batch)))

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

        if self._bucketing is not None:
            vae_downsample_rate = 1
            transforms.extend(
                [
                    {
                        "operations": ResizeCrop(interpolation=cv2.INTER_AREA),
                        "input_columns": ["video", "size"],
                        "output_columns": ["video"],  # drop `size` column
                    },
                    {
                        "operations": [
                            lambda x: x.astype(np.float32) / 127.5 - 1,
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
                        ResizeCrop(target_size, interpolation=cv2.INTER_LINEAR),
                        lambda x: x.astype(np.float32) / 127.5 - 1,
                        lambda x: np.transpose(x, (0, 3, 1, 2)),
                    ],
                    "input_columns": ["video"],
                }
            )
        # the followings are not transformation for video frames, can be excluded
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


class BucketGroupLoader:
    def __init__(
        self,
        dataset: VideoDatasetRefactored,
        buckets: Bucket,
        device_num: int = 1,
        rank_id: int = 0,
        shuffle: bool = False,
        seed: int = 42,
        drop_remainder: bool = True,
        *,
        output_columns: List[str],
    ):
        self._dataset = dataset
        self._buckets = buckets
        self._device_num = device_num
        self._rank = rank_id
        self._shuffle = shuffle
        self._seed = seed
        self._drop_remainder = drop_remainder
        self._epoch = 0
        self._bucket_samples = []
        self.output_columns = output_columns

    def __iter__(self):
        self._i = 0
        self._epoch += 1
        self._bucket_samples = []
        rng = np.random.default_rng(self._seed + self._epoch)

        _logger.debug("Building buckets...")
        if self._shuffle:
            indexes = rng.permutation(len(self._dataset._data)).tolist()
        else:
            indexes = list(range(len(self._dataset._data)))

        bucket_sample_dict = defaultdict(list)
        for i in indexes:
            bucket_id = self._buckets.get_bucket_id(
                int(self._dataset._data[i]["length"]),
                int(self._dataset._data[i]["height"]),
                int(self._dataset._data[i]["width"]),
                frame_interval=self._dataset._stride,
                seed=self._seed + self._epoch + i * self._buckets.num_bucket,  # Following the original implementation
            )
            # group by bucket
            # each data sample is put into a bucket with a similar image/video size
            if bucket_id is not None:
                bucket_sample_dict[bucket_id].append(i)

        # process the samples
        for bucket_id, data_list in bucket_sample_dict.items():
            bs_per_npu = self._buckets.get_batch_size(bucket_id)
            if remainder := len(data_list) % bs_per_npu:
                if not self._drop_remainder:  # when keeping a remainder, pad to make the batch divisible
                    data_list += data_list[: bs_per_npu - remainder]
                else:  # otherwise, drop the remainder
                    data_list = data_list[:-remainder]

            self._bucket_samples.extend(
                [
                    (self._buckets.get_thw(bucket_id), data_list[i : i + bs_per_npu])
                    for i in range(0, len(data_list), bs_per_npu)
                ]
            )

        # randomize the access order
        if self._shuffle:  # double shuffle following the original implementation
            bucket_indexes = rng.permutation(len(self._bucket_samples)).tolist()
            self._bucket_samples = [self._bucket_samples[i] for i in bucket_indexes]

        # make the number of bucket accesses divisible by dp size
        if remainder := len(self._bucket_samples) % self._device_num:
            if self._drop_remainder:
                self._bucket_samples = self._bucket_samples[:-remainder]
            else:
                self._bucket_samples += self._bucket_samples[: self._device_num - remainder]

        # keep only the samples for the current rank
        self._bucket_samples = self._bucket_samples[self._rank :: self._device_num]
        _logger.debug(f"Number of batches per rank: {len(self._bucket_samples)}")
        _logger.debug(f"Rank {self._rank} samples: {self._bucket_samples}")

        return self

    def __next__(self):
        if self._i >= len(self._bucket_samples):
            raise StopIteration

        thw, sample_ids = self._bucket_samples[self._i]
        self._i += 1

        _logger.debug(f"Rank {self._rank}: bucket {thw} | samples {sample_ids} ")
        return self._dataset.get_bucket(thw, sample_ids)

    def __len__(self):
        return len(self._bucket_samples) * self._device_num
