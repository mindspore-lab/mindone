import csv
import logging
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from tqdm import tqdm

from mindone.data import BaseDataset
from mindone.data.video_reader import VideoReader

from .buckets import get_target_size
from .transforms import ResizeCrop

_logger = logging.getLogger(__name__)


IMAGE_EXT = (".jpg", ".jpeg", ".png", ".gif", ".webp")


class ImageVideoDataset(BaseDataset):
    """
    A dataset for loading image and video data from a CSV file.

    Args:
        csv_path: The path to the CSV file containing the data annotations.
        video_folder: The path to the folder containing the videos.
        text_emb_folder: The folder or dictionary ({emb_name: folder}) containing the text embeddings.
        empty_text_emb: The path to the empty text embedding file or dictionary ({emb_name: file}). Default: None.
        text_drop_prob: The probability of dropping a text embedding during training. Default: 0.2.
        tae_latent_folder: The folder containing the TAE latent files. Default: None.
        tae_scale_factor: The scale factor for TAE latent files. Default: 1.5305.
        tae_shift_factor: The shift factor for TAE latent files. Default: 0.0609.
        target_size: The target size for resizing the frames. Default: 256px.
        sample_n_frames: The number of frames to sample from a video. Default: 16.
        sample_stride: The stride for sampling frames. Default: 1.
        deterministic_sample: Whether to sample frames starting from the beginning
                              (useful for an overfitting experiment on small datasets). Default: False.
        frames_mask_generator: The mask generator for frames. Default: None.
        t_compress_func: The function that returns the number of frames in the compressed (latent) space. Default: None.
        filter_data: Whether to filter out samples that are not found. Default: False.
        apply_transforms_dataset: Whether to apply the transformations to the dataset immediately during loading.
                                  Default: False.
        output_columns: The list of column names to output.
    """

    def __init__(
        self,
        csv_path: str,
        video_folder: str,
        text_emb_folder: Union[str, Dict[str, str]],
        empty_text_emb: Optional[Union[str, Dict[str, str]]] = None,
        text_drop_prob: float = 0.2,
        tae_latent_folder: Optional[str] = None,
        tae_scale_factor: float = 1.5305,
        tae_shift_factor: float = 0.0609,
        target_size: Union[str, Tuple[int, int]] = "256px",
        sample_n_frames: Union[int, List[int]] = 16,
        sample_stride: int = 1,
        deterministic_sample: bool = False,
        frames_mask_generator: Optional[Callable[[int], np.ndarray]] = None,
        t_compress_func: Optional[Callable[[int], int]] = None,
        filter_data: bool = False,
        apply_transforms_dataset: bool = False,
        *,
        output_columns: List[str],
    ):
        self._data = self._read_data(video_folder, csv_path, text_emb_folder, tae_latent_folder, filter_data)
        self._frames = sample_n_frames if isinstance(sample_n_frames, int) else sorted(sample_n_frames)
        if isinstance(self._frames, list) and self._frames[0] == 1:
            self._frames.pop(0)  # drop image length
        self._stride = sample_stride
        self._deterministic = deterministic_sample

        self._text_emb_folder = text_emb_folder
        self._empty_text_emb = empty_text_emb if text_drop_prob > 0 else None
        if self._empty_text_emb:
            if isinstance(self._empty_text_emb, str):
                assert os.path.exists(self._empty_text_emb), f"Empty text embedding not found: {self._empty_text_emb}"
            else:
                for path in self._empty_text_emb.values():
                    assert os.path.exists(path), f"Empty text embedding not found: {path}"
        self._text_drop_prob = text_drop_prob

        self._tae_latent_folder = tae_latent_folder
        if tae_latent_folder and (sample_stride > 1):
            _logger.info("TAE latent folder is specified, strides for sampling will be ignored.")
        self._tae_scale_factor = tae_scale_factor
        self._tae_shift_factor = tae_shift_factor

        self._fmask_gen = frames_mask_generator
        self._t_compress_func = t_compress_func or (lambda x: x)

        self.output_columns = output_columns

        self._transforms = None
        if apply_transforms_dataset:
            self._transforms = self.train_transforms(
                lambda h, w: get_target_size(target_size, h, w) if isinstance(target_size, str) else target_size,
                interpolation=cv2.INTER_AREA,
            )

        # prepare replacement data in case the loading of a sample fails
        self._prev_ok_sample = self._get_replacement()
        self._require_update_prev = False

    @staticmethod
    def _read_data(
        data_dir: str,
        csv_path: str,
        text_emb_folder: Optional[Union[str, Dict[str, str]]] = None,
        tae_latent_folder: Optional[str] = None,
        filter_data: bool = False,
    ) -> List[dict]:
        def _filter_data(sample_):
            if not os.path.isfile(sample_["video"]):
                _logger.warning(f"Video not found: {sample_['video']}")
                return None
            if "text_emb" in sample_:
                if isinstance(sample_["text_emb"], str) and not os.path.isfile(sample_["text_emb"]):
                    _logger.warning(f"Text embedding not found: {sample_['text_emb']}")
                    return None
                else:
                    for name, path in sample_["text_emb"].items():
                        if not os.path.isfile(sample_["text_emb"][name]):
                            _logger.warning(f"Text embedding not found: {sample_['text_emb'][name]}")
                            return None
            if "tae_latent" in sample_ and not os.path.isfile(sample_["tae_latent"]):
                _logger.warning(f"Text embedding not found: {sample_['tae_latent']}")
                return None
            return sample_

        with open(csv_path, "r", encoding="utf-8") as csv_file:
            try:
                data = []
                for item in csv.DictReader(csv_file):
                    sample = {**item, "video": os.path.join(data_dir, item["video"])}
                    if text_emb_folder:
                        if isinstance(text_emb_folder, str):
                            sample["text_emb"] = os.path.join(text_emb_folder, Path(item["video"]).with_suffix(".npz"))
                        else:
                            sample["text_emb"] = {
                                name: os.path.join(path, Path(item["video"]).with_suffix(".npz"))
                                for name, path in text_emb_folder.items()
                            }
                    if tae_latent_folder:
                        sample["tae_latent"] = os.path.join(tae_latent_folder, Path(item["video"]).with_suffix(".npz"))
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

    def _get_replacement(self, max_attempts: int = 100) -> Tuple[np.ndarray, ...]:
        attempts, error = min(max_attempts, len(self)), None
        for idx in range(attempts):
            try:
                return self._get_item(idx)
            except Exception as e:
                error = e
                _logger.debug(f"Failed to load a replacement sample: {repr(e)}")

        raise RuntimeError(f"Fail to load a replacement sample in {attempts} attempts. Error: {repr(error)}")

    def _get_item(self, idx: int) -> Tuple[np.ndarray, ...]:
        data = self._data[idx].copy()

        if self._text_emb_folder:
            if self._empty_text_emb and random.random() <= self._text_drop_prob:
                data["text_emb"] = self._empty_text_emb

            if isinstance(data["text_emb"], str):
                with np.load(data["text_emb"]) as td:
                    data.update({"caption": td["text_emb"], "mask": td["mask"]})
            else:
                for enc_name, path in data["text_emb"].items():
                    with np.load(path) as td:
                        data.update({enc_name + "_caption": td["text_emb"], enc_name + "_mask": td["mask"]})

        if self._tae_latent_folder:
            tae_latent_data = np.load(data["tae_latent"])
            latent_mean, latent_std = tae_latent_data["latent_mean"], tae_latent_data["latent_std"]  # T C H W

            # find the video target length
            min_length = 1
            if len(latent_mean) > 1:  # if a video
                if isinstance(self._frames, int):
                    min_length = self._frames
                else:
                    min_length = float("inf")
                    for i in range(len(self._frames) - 1, -1, -1):
                        if len(latent_mean) >= self._frames[i]:
                            min_length = self._frames[i]
                            break
                if len(latent_mean) < min_length:
                    raise ValueError(f"Video is too short: {data['video']}")

            start_pos = 0 if self._deterministic else random.randint(0, len(latent_mean) - min_length)
            latent_mean = latent_mean[start_pos : start_pos + min_length]
            latent_std = latent_std[start_pos : start_pos + min_length]
            tae_latent = np.random.normal(latent_mean, latent_std).astype(np.float32)
            data["video"] = (tae_latent - self._tae_shift_factor) * self._tae_scale_factor

        else:
            if data["video"].lower().endswith(IMAGE_EXT):
                num_frames = 1
                data["fps"] = np.array(120, dtype=np.float32)  # FIXME: extract as IMG_FPS
                data["video"] = cv2.cvtColor(cv2.imread(data["video"]), cv2.COLOR_BGR2RGB)
            else:
                with VideoReader(data["video"]) as reader:
                    if isinstance(self._frames, int):
                        num_frames = self._frames
                        min_length = (self._frames - 1) * self._stride + 1
                    else:
                        min_length = float("inf")
                        for i in range(len(self._frames) - 1, -1, -1):
                            if len(reader) >= (self._frames[i] - 1) * self._stride + 1:
                                num_frames = self._frames[i]
                                min_length = (self._frames[i] - 1) * self._stride + 1
                                break
                    if len(reader) < min_length:
                        raise ValueError(f"Video is too short: {data['video']}")

                    start_pos = 0 if self._deterministic else random.randint(0, len(reader) - min_length)
                    data["video"] = reader.fetch_frames(
                        num=num_frames, start_pos=start_pos, step=self._stride
                    )  # T H W C
                    data["fps"] = np.array(reader.fps / self._stride, dtype=np.float32)

            data["num_frames"] = np.array(num_frames, dtype=np.float32)

        if self._fmask_gen is not None:
            # return frames mask with respect to the TAE's latent temporal compression
            data["frames_mask"] = self._fmask_gen(self._t_compress_func(num_frames))

        if self._transforms:
            data = self._apply_transforms(data, self._transforms)

        return tuple(data[c] for c in self.output_columns)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, ...]:
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
        self,
        target_size: Union[Tuple[int, int], Callable[[Tuple[int, int]], Tuple[int, int]]],
        interpolation: int = cv2.INTER_LINEAR,
        tokenizer: Optional[Callable[[str], np.ndarray]] = None,
    ) -> List[dict]:
        transforms = []
        if not self._tae_latent_folder:
            transforms.append(
                {
                    "operations": [
                        ResizeCrop(target_size, interpolation=interpolation),
                        lambda x: x.astype(np.float32) / 127.5 - 1,
                        lambda x: x[None, ...] if x.ndim == 3 else x,  # if image
                        lambda x: np.transpose(x, (0, 3, 1, 2)),  # T H W C -> T C H W
                    ],
                    "input_columns": ["video"],
                }
            )

        if "caption" in self.output_columns and not self._text_emb_folder:
            if tokenizer is None:
                raise RuntimeError("Please provide a tokenizer for text data in `train_transforms()`.")
            transforms.append({"operations": [tokenizer], "input_columns": ["caption"]})

        return transforms
