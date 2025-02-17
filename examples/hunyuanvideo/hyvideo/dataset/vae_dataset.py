import copy
import csv
import json
import logging
import os
import random
from typing import List, Literal, Tuple, Union

import imageio
import numpy as np
from decord import VideoReader
from hyvideo.utils.dataset_utils import create_video_transforms
from PIL import Image, ImageSequence

logger = logging.getLogger(__name__)


def read_gif(gif_path, mode="RGB"):
    with Image.open(gif_path) as fp:
        frames = np.array([np.array(frame.convert(mode)) for frame in ImageSequence.Iterator(fp)])
    return frames


def get_video_path_list(folder):
    video_paths = []
    fmts = ["avi", "mp4", "gif"]

    for root, dirs, files in os.walk(folder):
        for file in files:
            for fmt in fmts:
                if file.endswith(f".{fmt}"):
                    video_paths.append(os.path.join(root, file))

    return sorted(video_paths)


class VideoDataset:
    def __init__(
        self,
        data_file_path: str = None,
        data_folder: str = None,
        size: Union[Tuple[int], List[int], int] = 384,
        crop_size: Union[Tuple[int], List[int], int] = 256,
        random_crop: bool = False,
        sample_stride: int = 4,
        sample_n_frames: int = 16,
        return_image: bool = False,
        transform_backend: str = "al",
        video_column: str = "video",
        disable_flip: bool = True,
        dynamic_sample: bool = False,  # random sample rate
        dynamic_start_index: bool = True,  # random start index
        output_columns: Union[Tuple[str], List[str]] = ["video", "path"],
    ):
        if data_file_path is not None:
            logger.info(f"loading videos from data file {data_file_path} ...")
            self.parse_data_file(data_file_path)
            self.read_from_data_file = True
        else:
            logger.info(f"loading videos from video folder {data_folder} recursively...")
            self.dataset = get_video_path_list(data_folder)
            self.read_from_data_file = False

        self.length = len(self.dataset)
        logger.info(f"Num data samples: {self.length}")
        logger.info(f"sample_n_frames: {sample_n_frames}")

        self.data_folder = data_folder
        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames
        self.return_image = return_image
        self.dynamic_sample = dynamic_sample
        self.dynamic_start_index = dynamic_start_index
        if not self.dynamic_start_index:
            logger.info(
                f"Always using the first frame as the start index for {sample_n_frames} frames sampling. Better to use it for inference not training!"
            )

        self.pixel_transforms = create_video_transforms(
            size=size,
            crop_size=crop_size,
            random_crop=random_crop,
            disable_flip=disable_flip,
            num_frames=sample_n_frames,
        )
        self.target_size = list(crop_size) if isinstance(crop_size, (tuple, list)) else [crop_size, crop_size]

        self.transform_backend = transform_backend
        self.video_column = video_column
        self.output_columns = list(output_columns)
        assert "video" in self.output_columns, "At least video should be returned"

        # prepare replacement data
        max_attempts = 100
        self.prev_ok_sample = self.get_replace_data(max_attempts)
        self.require_update_prev = False

    def parse_data_file(self, data_file_path):
        if data_file_path.endswith(".csv"):
            with open(data_file_path, "r") as csvfile:
                self.dataset = list(csv.DictReader(csvfile))
        elif data_file_path.endswith(".json"):
            with open(data_file_path, "r") as f:
                self.dataset = json.load(f)
        else:
            raise ValueError("Only support json and csv file now!")

    def get_replace_data(self, max_attempts=100):
        replace_data = None
        attempts = min(max_attempts, self.length)
        for idx in range(attempts):
            try:
                pixel_values, video_path = self.get_batch(idx)
                replace_data = copy.deepcopy(pixel_values)
                break
            except Exception as e:
                print("\tError msg: {}".format(e))

        assert replace_data is not None, f"Fail to preload sample in {attempts} attempts."

        return replace_data, video_path

    def get_batch(self, idx):
        # get video raw pixels (batch of frame) and its caption
        if self.read_from_data_file:
            video_dict = self.dataset[idx]
            video_fn = video_dict[self.video_column]
            video_path = os.path.join(self.data_folder, video_fn)
        else:
            video_path = self.dataset[idx]

        if video_path.endswith(".gif"):
            video_reader = read_gif(video_path, mode="RGB")
        else:
            video_reader = VideoReader(video_path)

        video_length = len(video_reader)

        if not self.return_image:
            if self.dynamic_sample:
                sample_stride = random.randint(1, self.sample_stride)
            else:
                sample_stride = self.sample_stride
            clip_length = min(video_length, (self.sample_n_frames - 1) * sample_stride + 1)
            if self.dynamic_start_index:
                start_idx = random.randint(0, video_length - clip_length)
            else:
                start_idx = 0
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        else:
            batch_index = [random.randint(0, video_length - 1)]

        if video_path.endswith(".gif"):
            pixel_values = video_reader[batch_index]  # shape: (f, h, w, c)
        else:
            pixel_values = video_reader.get_batch(batch_index).asnumpy()  # shape: (f, h, w, c)

        del video_reader

        return pixel_values, video_path

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Returns:
            video: preprocessed video frames in shape (f, c, h, w), normalized to [-1, 1]
        """
        try:
            pixel_values, video_path = self.get_batch(idx)
            if (self.prev_ok_sample is None) or (self.require_update_prev):
                self.prev_ok_sample = copy.deepcopy(pixel_values)
                self.require_update_prev = False
        except Exception as e:
            logger.warning(f"Fail to get sample of idx {idx}. The corrupted video will be replaced.")
            print("\tError msg: {}".format(e), flush=True)
            assert self.prev_ok_sample is not None
            pixel_values, video_path = self.prev_ok_sample  # unless the first sample is already not ok
            self.require_update_prev = True

            if idx >= self.length:
                raise IndexError  # needed for checking the end of dataset iteration

        num_frames = len(pixel_values)
        # pixel value: (f, h, w, 3) -> transforms -> (f 3 h' w')
        if self.transform_backend == "al":
            # NOTE:it's to ensure augment all frames in a video in the same way.
            # ref: https://albumentations.ai/docs/examples/example_multi_target/

            inputs = {"image": pixel_values[0]}
            for i in range(num_frames - 1):
                inputs[f"image{i}"] = pixel_values[i + 1]

            output = self.pixel_transforms(**inputs)

            pixel_values = np.stack(list(output.values()), axis=0)
            # (t h w c) -> (c t h w)
            pixel_values = np.transpose(pixel_values, (3, 0, 1, 2))
        else:
            raise NotImplementedError

        if self.return_image:
            pixel_values = pixel_values[1]

        pixel_values = (pixel_values / 127.5 - 1.0).astype(np.float32)

        if "path" in self.output_columns:
            return pixel_values, video_path
        else:
            return pixel_values


# TODO: parse in config dict
def check_sanity(x, save_fp="./tmp.gif"):
    # reverse normalization and visulaize the transformed video
    # (c, t, h, w) -> (t, h, w, c)
    if len(x.shape) == 3:
        x = np.expand_dims(x, axis=0)
    x = np.transpose(x, (1, 2, 3, 0))

    x = (x + 1.0) / 2.0  # -1,1 -> 0,1
    x = (x * 255).astype(np.uint8)

    imageio.mimsave(save_fp, x, duration=1 / 8.0, loop=1)


class BatchTransform:
    def __init__(
        self,
        mixed_strategy: Literal["mixed_video_image", "mixed_video_random", "image_only"],
        mixed_image_ratio: float = 0.2,
    ):
        if mixed_strategy == "mixed_video_image":
            self._trans_fn = self._mixed_video_image
        elif mixed_strategy == "mixed_video_random":
            self._trans_fn = self._mixed_video_random
        elif mixed_strategy == "image_only":
            self._trans_fn = self._image_only
        else:
            raise NotImplementedError(f"Unknown mixed_strategy: {mixed_strategy}")
        self.mixed_image_ratio = mixed_image_ratio

    def _mixed_video_image(self, x: np.ndarray) -> np.ndarray:
        if random.random() < self.mixed_image_ratio:
            x = x[:, :, :1, :, :]
        return x

    @staticmethod
    def _mixed_video_random(x: np.ndarray) -> np.ndarray:
        # TODO: somehow it's slow. consider do it with tensor in NetWithLoss
        length = random.randint(1, x.shape[2])
        return x[:, :, :length, :, :]

    @staticmethod
    def _image_only(x: np.ndarray) -> np.ndarray:
        return x[:, :, :1, :, :]

    def __call__(self, x):
        # x: (bs, c, t, h, w)
        return self._trans_fn(x)


if __name__ == "__main__":
    ds_config = dict(
        data_folder="../videocomposer/datasets/webvid5",
        random_crop=True,
        flip=True,
    )
    # test source dataset
    ds = VideoDataset(**ds_config)
    sample = ds.__getitem__(0)
    print(sample.shape)

    check_sanity(sample)
