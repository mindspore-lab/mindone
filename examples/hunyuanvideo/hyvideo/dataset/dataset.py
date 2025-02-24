import csv
import logging
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from hyvideo.constants import VAE_PATH
from hyvideo.modules.posemb_layers import get_nd_rotary_pos_embed
from hyvideo.utils.dataset_utils import DecordDecoder, create_video_transforms
from tqdm import tqdm

_logger = logging.getLogger(__name__)


IMAGE_EXT = (".jpg", ".jpeg", ".png", ".gif", ".webp")


class ImageVideoDataset:
    """
    A dataset for loading image and video data from a CSV file.

    Args:
        csv_path: The path to the CSV file containing the data annotations.
        video_folder: The path to the folder containing the videos.
        text_emb_folder: The folder or dictionary ({emb_name: folder}) containing the text embeddings.
        empty_text_emb: The path to the empty text embedding file or dictionary ({emb_name: file}). Default: None.
        text_drop_prob: The probability of dropping a text embedding during training. Default: 0.2.
        vae_latent_folder: The folder containing the vae latent files. Default: None.
        vae_scale_factor: The scale factor for vae latent files. Default: 0.476986.
        vae_shift_factor: The shift factor for vae latent files. Default: None.
        target_size: The target size for resizing the frames. Default: None.
        sample_n_frames: The number of frames to sample from a video. Default: 16.
        sample_stride: The stride for sampling frames. Default: 1.
        deterministic_sample: Whether to sample frames starting from the beginning
                              (useful for an overfitting experiment on small datasets). Default: False.
        frames_mask_generator: The mask generator for frames. Default: None.
        t_compress_func: The function that returns the number of frames in the compressed (latent) space. Default: None.
        filter_data: Whether to filter out samples that are not found. Default: False.
        output_columns: The list of column names to output.
    """

    def __init__(
        self,
        csv_path: str,
        video_folder: str,
        text_emb_folder: Union[str, Dict[str, str]],
        empty_text_emb: Optional[Union[str, Dict[str, str]]] = None,
        text_drop_prob: float = 0.2,
        vae_latent_folder: Optional[str] = None,
        vae_scale_factor: float = 0.476986,
        vae_shift_factor: float = None,
        target_size: Optional[Tuple[int, int]] = None,
        sample_n_frames: int = 16,
        sample_stride: int = 1,
        deterministic_sample: bool = False,
        frames_mask_generator: Optional[Callable[[int], np.ndarray]] = None,
        t_compress_func: Optional[Callable[[int], int]] = None,
        filter_data: bool = False,
        tokenizer: Optional[Callable[[str], np.ndarray]] = None,
        *,
        output_columns: List[str],
        # VAE
        vae_type: str = "884-16c-hy",
        # Hyvideo
        model_patch_size: list = [1, 2, 2],
        model_hidden_size: int = 3072,
        model_heads_num: int = 24,
        model_rope_dim_list: List[int] = [16, 56, 56],
        rope_theta: int = 256,
    ):
        self._data = self._read_data(video_folder, csv_path, text_emb_folder, vae_latent_folder, filter_data)
        self._frames = sample_n_frames
        self._stride = sample_stride
        self._min_length = (self._frames - 1) * self._stride + 1
        self._deterministic = deterministic_sample

        self._text_emb_folder = text_emb_folder
        self._empty_text_emb = empty_text_emb if text_drop_prob > 0 else None
        if self._empty_text_emb:
            if isinstance(self._empty_text_emb, str):
                assert os.path.exists(self._empty_text_emb), f"Empty text embedding not found: {self._empty_text_emb}"

        self._text_drop_prob = text_drop_prob

        self._vae_latent_folder = vae_latent_folder
        self._vae_scale_factor = vae_scale_factor
        assert self._vae_scale_factor is not None, "vae_scale_factor must be specified"
        self._vae_shift_factor = vae_shift_factor
        self._fmask_gen = frames_mask_generator
        self._t_compress_func = t_compress_func or (lambda x: x)

        self.output_columns = output_columns
        self.tokenizer = tokenizer

        self.pixel_transforms = create_video_transforms(
            size=target_size,
            crop_size=target_size,
            random_crop=False,
            disable_flip=False,
            num_frames=sample_n_frames,
        )

        self.vae_type = vae_type
        self.model_patch_size = model_patch_size
        self.model_hidden_size = model_hidden_size
        self.model_heads_num = model_heads_num
        self.model_rope_dim_list = model_rope_dim_list
        assert self.vae_type in VAE_PATH, f"Expected vae_type to be one of {VAE_PATH.keys()}"
        self.rope_theta = rope_theta
        self.freqs_cos, self.freqs_sin = self.get_rotary_pos_embed(sample_n_frames, target_size[0], target_size[1])

        # prepare replacement data in case the loading of a sample fails
        self._prev_ok_sample = self._get_replacement()
        self._require_update_prev = False

    def get_rotary_pos_embed(self, video_length, height, width):
        target_ndim = 3
        ndim = 5 - 2
        # 884
        if "884" in self.vae_type:
            latents_size = [(video_length - 1) // 4 + 1, height // 8, width // 8]
        elif "888" in self.vae_type:
            latents_size = [(video_length - 1) // 8 + 1, height // 8, width // 8]
        else:
            latents_size = [video_length, height // 8, width // 8]

        if isinstance(self.model_patch_size, int):
            assert all(s % self.model_patch_size == 0 for s in latents_size), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({self.model_patch_size}), "
                f"but got {latents_size}."
            )
            rope_sizes = [s // self.model_patch_size for s in latents_size]
        elif isinstance(self.model_patch_size, list):
            assert all(s % self.model_patch_size[idx] == 0 for idx, s in enumerate(latents_size)), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({self.model_patch_size}), "
                f"but got {latents_size}."
            )
            rope_sizes = [s // self.model_patch_size[idx] for idx, s in enumerate(latents_size)]

        if len(rope_sizes) != target_ndim:
            rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes  # time axis
        head_dim = self.model_hidden_size // self.model_heads_num
        rope_dim_list = self.model_rope_dim_list
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert sum(rope_dim_list) == head_dim, "sum(rope_dim_list) should equal to head_dim of attention layer"

        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            rope_dim_list,
            rope_sizes,
            theta=self.rope_theta,
            use_real=True,
            theta_rescale_factor=1,
        )
        return freqs_cos.asnumpy(), freqs_sin.asnumpy()

    @staticmethod
    def _read_data(
        data_dir: str,
        csv_path: str,
        text_emb_folder: Optional[Union[str, Dict[str, str]]] = None,
        vae_latent_folder: Optional[str] = None,
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

            if "vae_latent" in sample_ and not os.path.isfile(sample_["vae_latent"]):
                _logger.warning(f"Text embedding not found: {sample_['vae_latent']}")
                return None
            return sample_

        with open(csv_path, "r", encoding="utf-8") as csv_file:
            try:
                data = []
                for item in csv.DictReader(csv_file):
                    sample = {**item, "video": os.path.join(data_dir, item["video"])}
                    if text_emb_folder:
                        if isinstance(text_emb_folder, str):
                            sample["text_emb"] = os.path.join(
                                text_emb_folder, Path(item["video"] + "-*").with_suffix(".npz")
                            )
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

    def _get_replacement(self, max_attempts: int = 100) -> Tuple[np.ndarray, ...]:
        attempts, error = min(max_attempts, len(self)), None
        for idx in range(attempts):
            try:
                return self._get_item(idx)
            except Exception as e:
                error = e
                _logger.debug(f"Failed to load a replacement sample: {repr(e)}")

        raise RuntimeError(f"Fail to load a replacement sample in {attempts} attempts. Error: {repr(error)}")

    def _get_item(self, idx: int, thw: Optional[Tuple[int, int, int]] = None) -> Tuple[np.ndarray, ...]:
        data = self._data[idx].copy()
        num_frames = self._frames

        if self._text_emb_folder:
            if self._empty_text_emb and random.random() <= self._text_drop_prob:
                data["text_emb"] = self._empty_text_emb

            if isinstance(data["text_emb"], str):
                with np.load(data["text_emb"]) as td:
                    data.update({"prompt_embeds": td["prompt_embeds"], "prompt_mask": td["prompt_mask"]})
                    if "prompt_embeds_2" in td:
                        data.update({"prompt_embeds_2": td["prompt_embeds_2"]})

        if self._vae_latent_folder:
            vae_latent_data = np.load(data["vae_latent"])
            latent_mean, latent_std = vae_latent_data["latent_mean"], vae_latent_data["latent_std"]  # C T H W
            if 1 < len(latent_mean) < self._min_length:  # TODO: add support for buckets
                raise ValueError(f"Video is too short: {data['video']}")

            start_pos = 0 if self._deterministic else random.randint(0, len(latent_mean) - self._min_length)
            batch_index = np.linspace(start_pos, start_pos + self._min_length - 1, num_frames, dtype=int)

            latent_mean, latent_std = latent_mean[batch_index], latent_std[batch_index]
            vae_latent = np.random.normal(latent_mean, latent_std).astype(np.float32)

            vae_latent = vae_latent * self._vae_scale_factor
            data["video"] = vae_latent
        else:
            if data["video"].lower().endswith(IMAGE_EXT):
                num_frames = 1
                data["fps"] = np.array(120, dtype=np.float32)  # FIXME: extract as IMG_FPS
                data["video"] = np.array(cv2.cvtColor(cv2.imread(data["video"]), cv2.COLOR_BGR2RGB))[
                    None, ...
                ]  # (1, H, W, 3)
            else:
                decord_vr = DecordDecoder(data["video"])
                min_length = self._min_length
                if thw is not None:
                    num_frames, *data["size"] = thw
                    min_length = (num_frames - 1) * self._stride + 1
                if decord_vr.get_num_frames() < min_length:
                    raise ValueError(f"Video is too short: {data['video']}")
                start_pos = 0 if self._deterministic else random.randint(0, decord_vr.get_num_frames() - min_length)
                frame_indices = np.arange(start_pos, decord_vr.get_num_frames())[:: self._stride]
                if len(frame_indices) < num_frames:
                    print(f"The number of frames of video {data['video']} is less than the number of frames required.")
                frame_indices = frame_indices[:num_frames]
                data["video"] = decord_vr.get_batch(frame_indices)  # T H W C
                data["fps"] = np.array(decord_vr.get_avg_fps() / self._stride, dtype=np.float32)

        data["num_frames"] = np.array(num_frames, dtype=np.float32)

        if self._fmask_gen is not None:
            # return frames mask with respect to the vae's latent temporal compression
            data["frames_mask"] = self._fmask_gen(self._t_compress_func(num_frames))
        # video/image transforms: resize, crop, normalize, reshape
        pixel_values = data["video"]
        inputs = {"image": pixel_values[0]}
        for i in range(num_frames - 1):
            inputs[f"image{i}"] = pixel_values[i + 1]

        output = self.pixel_transforms(**inputs)
        pixel_values = np.stack(list(output.values()), axis=0)
        # (t h w c) -> (c t h w)
        pixel_values = np.transpose(pixel_values, (3, 0, 1, 2))
        data["video"] = pixel_values / 127.5 - 1.0

        # rope frequencies
        data["freqs_cos"] = self.freqs_cos
        data["freqs_sin"] = self.freqs_sin

        if "caption" in self.output_columns and not self._text_emb_folder:
            if self.tokenizer is None:
                raise RuntimeError("Please provide a tokenizer for text data.")
            data["caption"] = self.tokenizer(data["caption"])

        return tuple(data[c] for c in self.output_columns)

    def get_bucket(self, thw: Tuple[int, int, int], sample_ids: List[int]) -> Tuple[np.ndarray, ...]:
        batch = [self._get_item(sample_id, thw) for sample_id in sample_ids]
        return tuple(np.stack(item) for item in map(list, zip(*batch)))

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
