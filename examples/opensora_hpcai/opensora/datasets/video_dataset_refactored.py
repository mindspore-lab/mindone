import csv
import glob
import logging
import os
import random
from concurrent.futures import ThreadPoolExecutor
from math import sqrt
from pathlib import Path
from typing import Any, Callable, Optional, Union

import cv2
import numpy as np
from einops import rearrange, repeat
from tqdm import tqdm

from mindspore.dataset.transforms import Compose

from mindone.data import BaseDataset
from mindone.data.video_reader import VideoReader as VideoReader_CV2
from mindone.models.modules.pos_embed import get_2d_sincos_pos_embed

from ..models.layers.rotary_embedding import precompute_freqs_cis
from ..pipelines.utils_v2 import get_res_lin_function
from .bucket import Bucket
from .transforms import ResizeCrop

_logger = logging.getLogger(__name__)


def create_infer_transforms(target_size: tuple[int, int], interpolation=cv2.INTER_LINEAR):
    return Compose(
        [
            ResizeCrop(target_size, interpolation=interpolation),
            lambda x: x[None, ...] if x.ndim == 3 else x,  # if image
            lambda x: np.transpose(x, (0, 3, 1, 2)),  # ms.HWC2CHW() doesn't support 4D data
            lambda x: x.astype(np.float32) / 127.5 - 1,
        ]
    )


class VideoDatasetRefactored(BaseDataset):
    def __init__(
        self,
        csv_path: str,
        video_folder: str,
        text_emb_folder: Union[str, dict[str, str]],
        empty_text_emb: Optional[Union[str, dict[str, str]]] = None,
        text_drop_prob: float = 0.2,
        vae_latent_folder: Optional[str] = None,
        vae_downsample_rate: float = 8.0,
        vae_scale_factor: float = 0.18215,
        vae_shift_factor: float = 0,
        sample_n_frames: int = 16,
        sample_stride: int = 1,
        frames_mask_generator: Optional[Callable[[int], np.ndarray]] = None,
        t_compress_func: Optional[Callable[[int], int]] = None,
        pre_patchify: bool = False,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        embed_dim: int = 1152,
        num_heads: int = 16,
        max_target_size: int = 512,
        input_sq_size: int = 512,
        in_channels: int = 4,
        buckets: Optional[Bucket] = None,
        filter_data: bool = False,
        apply_transforms_dataset: bool = False,
        target_size: Optional[tuple[int, int]] = None,
        tokenizer=None,
        video_backend: str = "cv2",
        v2_pipeline: bool = False,
        *,
        output_columns: list[str],
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
        self._empty_text_emb = empty_text_emb if text_drop_prob > 0 else None
        if self._empty_text_emb:
            if isinstance(self._empty_text_emb, str):
                assert os.path.exists(self._empty_text_emb), f"Empty text embedding not found: {self._empty_text_emb}"
            else:
                for path in self._empty_text_emb.values():
                    assert os.path.exists(path), f"Empty text embedding not found: {path}"
        self._text_drop_prob = text_drop_prob
        self._vae_latent_folder = vae_latent_folder
        self._vae_downsample_rate = vae_downsample_rate
        self._vae_scale_factor = vae_scale_factor
        self._vae_shift_factor = vae_shift_factor
        self._fmask_gen = frames_mask_generator
        self._t_compress_func = t_compress_func or (lambda x: x)
        self._pre_patchify = pre_patchify
        self._buckets = buckets
        self._v2_pipeline = v2_pipeline

        self.output_columns = output_columns
        if self._buckets:
            assert vae_latent_folder is None, "`vae_latent_folder` is not supported with bucketing"
            self.output_columns += ["size"]  # pass bucket id information to transformations

        self._patch_size = patch_size
        assert self._patch_size[0] == 1
        if self._pre_patchify:
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

        self._transforms = []
        if apply_transforms_dataset:
            self._transforms = self.train_transforms(target_size)

        # prepare replacement data in case the loading of a sample fails
        self._prev_ok_sample = (
            self._get_replacement() if not self._buckets or isinstance(self._buckets, Bucket) else None
        )
        self._require_update_prev = False

    @staticmethod
    def _read_data(
        data_dir: str,
        csv_path: str,
        text_emb_folder: Optional[Union[str, dict[str, str]]] = None,
        vae_latent_folder: Optional[str] = None,
        filter_data: bool = False,
    ) -> list[dict]:
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
                        if isinstance(text_emb_folder, str):
                            sample["text_emb"] = os.path.join(text_emb_folder, Path(item["video"]).with_suffix(".npz"))
                        else:
                            sample["text_emb"] = {
                                name: os.path.join(path, Path(item["video"]).with_suffix(".npy"))
                                for name, path in text_emb_folder.items()
                            }
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

    def _get_replacement(self, max_attempts: int = 100) -> tuple[Any, ...]:
        attempts = min(max_attempts, len(self))
        error = None
        for idx in range(attempts):
            try:
                return self._get_item(idx)
            except Exception as e:
                error = e
                _logger.debug(f"Failed to load a replacement sample: {repr(e)}")

        raise RuntimeError(f"Fail to load a replacement sample in {attempts} attempts. Error: {repr(error)}")

    def _get_item(self, idx: int) -> tuple[Any, ...]:
        data = self._data[idx].copy()

        num_frames = self._frames

        if self._text_emb_folder:
            if self._empty_text_emb and random.random() <= self._text_drop_prob:
                data["text_emb"] = self._empty_text_emb

            if isinstance(data["text_emb"], str):
                with np.load(data["text_emb"]) as td:
                    data.update({"caption": td["text_emb"], "mask": td["mask"].astype(np.uint8)})
            else:
                for enc_name, path in data["text_emb"].items():
                    data.update({enc_name + "_caption": np.load(path)})  # No masks in V2.0

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
                with VideoReader_CV2(data["video"]) as reader:
                    self._data[idx]["fps"] = data["fps"] = reader.fps

            latent_mean, latent_std = vae_latent_data["latent_mean"], vae_latent_data["latent_std"]
            if len(latent_mean) < self._min_length:
                raise ValueError(f"Video is too short: {data['video']}")

            start_pos = random.randint(0, len(latent_mean) - self._min_length)
            batch_index = np.linspace(start_pos, start_pos + self._min_length - 1, num_frames, dtype=int)

            latent_mean, latent_std = latent_mean[batch_index], latent_std[batch_index]
            vae_latent = latent_mean + latent_std * np.random.standard_normal(latent_mean.shape)
            video = (self._vae_scale_factor * (vae_latent - self._vae_shift_factor)).astype(np.float32)

            data["height"] = np.array(video.shape[-2] * self._vae_downsample_rate, dtype=np.float32)
            data["width"] = np.array(video.shape[-1] * self._vae_downsample_rate, dtype=np.float32)
            # NOTE: here ar = h / w, aligned to torch, while the common practice is w / h
            data["ar"] = np.array(video.shape[-2] / video.shape[-1], dtype=np.float32)

        else:
            if self.video_backend == "decord":
                from decord import VideoReader

                reader = VideoReader(data["video"])
                min_length = self._min_length
                video_length = len(reader)
                if self._buckets:
                    cap = cv2.VideoCapture(data["video"], apiPreference=cv2.CAP_FFMPEG)
                    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()

                    bucket_id = self._buckets.get_bucket_id(
                        T=video_length, H=frame_h, W=frame_w, frame_interval=self._stride
                    )
                    if bucket_id is None:
                        raise ValueError(
                            f"Couldn't assign a bucket to {data['video']}"
                            f" (T={video_length}, H={frame_h}, W={frame_w})."
                        )
                    num_frames, *data["size"] = self._buckets.get_thw(bucket_id)
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
                with VideoReader_CV2(data["video"]) as reader:
                    min_length = self._min_length
                    if self._buckets:
                        bucket_id = self._buckets.get_bucket_id(
                            T=len(reader), H=reader.shape[1], W=reader.shape[0], frame_interval=self._stride
                        )
                        if bucket_id is None:
                            raise ValueError(
                                f"Couldn't assign a bucket to {data['video']}"
                                f" (T={len(reader)}, H={reader.shape[1]}, W={reader.shape[0]})."
                            )
                        num_frames, *data["size"] = self._buckets.get_thw(bucket_id)
                        min_length = (num_frames - 1) * self._stride + 1

                    if len(reader) < min_length:
                        raise ValueError(f"Video is too short: {data['video']}")
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
        if self._transforms:
            data = self._apply_transforms(data, self._transforms)

        return tuple(data[c] for c in self.output_columns)

    def __getitem__(self, idx: int) -> tuple[Any, ...]:
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

    def _get_dynamic_size(self, h: int, w: int) -> tuple[int, int]:
        if h % self._patch_size[1] != 0:
            h += self._patch_size[1] - h % self._patch_size[1]
        if w % self._patch_size[2] != 0:
            w += self._patch_size[2] - w % self._patch_size[2]
        h = h // self._patch_size[1]
        w = w // self._patch_size[2]
        return h, w

    def _patchify(self, latent: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    # TODO: merge with _pre_patchify
    def _prepare_inputs(
        self,
        img: np.ndarray,
        t5_emb: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare the input for the model.

        Args:
            t5 (HFEmbedder): The T5 model.
            clip (HFEmbedder): The CLIP model.
            img (Tensor): The image tensor.
            prompt (str | list[str]): The prompt(s).

        Returns:
            dict[str, Tensor]: The input dictionary.

            img_ids: used for positional embedding in T,H,W dimensions later
            text_ids: for positional embedding, but set to 0 for now since our text encoder already encodes positional information
        """
        t, h, w = img.shape[-3:]

        # follow SD3 time shift, shift_alpha = 1 for 256px and shift_alpha = 3 for 1024 px
        shift_alpha = get_res_lin_function()((h * w) // 4)
        # add temporal influence
        shift_alpha *= sqrt(t)  # for image, T=1 so no effect

        img = rearrange(img, "c t (h ph) (w pw) -> (t h w) (c ph pw)", ph=self._patch_size[1], pw=self._patch_size[2])
        img_ids = np.zeros((t, h // self._patch_size[1], w // self._patch_size[2], 3), dtype=np.int32)
        img_ids[..., 0] = img_ids[..., 0] + np.arange(t)[:, None, None]
        img_ids[..., 1] = img_ids[..., 1] + np.arange(h // self._patch_size[1])[None, :, None]
        img_ids[..., 2] = img_ids[..., 2] + np.arange(w // self._patch_size[2])[None, None, :]
        img_ids = repeat(img_ids, "t h w c -> (t h w) c")

        # Encode the tokenized prompts
        txt_ids = np.zeros((t5_emb.shape[0], 3), dtype=np.int32)
        return img, img_ids, t5_emb, txt_ids, np.array(shift_alpha, dtype=np.float32)

    def train_transforms(
        self, target_size: tuple[int, int], tokenizer: Optional[Callable[[str], np.ndarray]] = None
    ) -> list[dict]:
        transforms = []
        vae_downsample_rate = self._vae_downsample_rate

        if not self._vae_latent_folder:
            vae_downsample_rate = 1
            transforms.append(
                {
                    "operations": [
                        ResizeCrop(target_size, interpolation=cv2.INTER_AREA),
                        lambda x: np.transpose(x, (0, 3, 1, 2)),  # ms.HWC2CHW() doesn't support 4D data
                        lambda x: x.astype(np.float32) / 127.5 - 1,
                    ],
                    "input_columns": ["video", "size"] if self._buckets else ["video"],
                    "output_columns": ["video"],  # drop `size` column
                }
            )

        if not self._v2_pipeline:
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

        if self._v2_pipeline:
            transforms.append(
                {  # TODO: merge with _pre_patchify
                    "operations": [self._prepare_inputs],
                    "input_columns": ["video", "t5_caption"],
                    "output_columns": ["video", "video_ids", "t5_caption", "txt_ids", "shift_alpha"],
                }
            )

        if "caption" in self.output_columns and not self._text_emb_folder:
            if tokenizer is None:
                raise RuntimeError("Please provide a tokenizer for text data in `train_transforms()`.")
            transforms.append({"operations": [tokenizer], "input_columns": ["caption"]})

        return transforms
