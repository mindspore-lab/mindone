import csv
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import cv2
import imageio
import numpy as np
from decord import VideoReader

from mindone.data import BaseDataset

from .transforms import HorizontalFlip, ResizeCrop

__all__ = ["VideoDataset", "BatchTransform"]

logger = logging.getLogger()


def get_video_path_list(folder: str, video_column: str) -> List[Dict[str, str]]:
    """
    Constructs a list of images and videos in the given directory (recursively).

    Args:
        folder: path to a directory containing images and videos.
        video_column: name of the column to store video paths.
    Returns:
        A list of paths to images and videos in the given directory (absolute and relative).
    """
    exts = (".jpg", ".jpeg", ".png", ".gif", ".mp4", ".avi")
    data = [
        {video_column: str(item), "rel_path": str(item.relative_to(folder))}
        for item in Path(folder).rglob("*")
        if (item.is_file() and item.suffix.lower() in exts)
    ]
    return sorted(data, key=lambda x: x[video_column])


class VideoDataset(BaseDataset):
    def __init__(
        self,
        csv_path: Optional[str],
        folder: str,
        size: Tuple[int, int] = (256, 256),
        random_crop: bool = False,
        random_flip: bool = False,
        sample_stride: int = 1,
        sample_n_frames: int = 16,
        deterministic_sample: bool = False,
        return_image: bool = False,
        video_column: str = "video",
        *,
        output_columns: List[str],
    ):
        if csv_path is not None:
            logger.info(f"Loading annotations from {csv_path} ...")
            with open(csv_path, "r") as csvfile:
                self.dataset = [
                    {**item, video_column: os.path.join(folder, item[video_column]), "rel_path": item[video_column]}
                    for item in csv.DictReader(csvfile)
                ]
        else:
            logger.info(f"Loading video list from {folder} ...")
            self.dataset = get_video_path_list(folder, video_column)

        logger.info(f"Num data samples: {len(self.dataset)}")
        logger.info(f"sample_n_frames: {sample_n_frames}")

        self.folder = folder
        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames
        self.return_image = return_image
        self._deterministic = deterministic_sample

        self._transforms = self.train_transforms(size=size, random_crop=random_crop, random_flip=random_flip)
        self.video_column = video_column
        self.output_columns = output_columns

        # prepare replacement data
        max_attempts = 100
        self.prev_ok_sample = self.get_replace_data(max_attempts)
        self.require_update_prev = False

    def get_replace_data(self, max_attempts=100):
        replace_data = None
        attempts = min(max_attempts, len(self.dataset))
        for idx in range(attempts):
            try:
                replace_data = self.get_batch(idx)
                break
            except Exception as e:
                print(f"\tError msg: {e}")

        assert replace_data is not None, f"Fail to preload sample in {attempts} attempts."

        return replace_data

    def get_batch(self, idx):
        # get video raw pixels (batch of frame) and its caption
        video_dict = self.dataset[idx].copy()
        video_path = video_dict[self.video_column]

        video_reader = VideoReader(video_path)
        video_length = len(video_reader)

        if not self.return_image:
            clip_length = video_length
            if self.sample_n_frames > 0:
                clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
            start_idx = 0 if self._deterministic else random.randint(0, video_length - clip_length)
            batch_index = list(range(start_idx, start_idx + clip_length, self.sample_stride))
        else:
            batch_index = [0] if self._deterministic else [random.randint(0, video_length - 1)]

        if video_path.endswith(".gif"):
            video_dict[self.video_column] = video_reader[batch_index]  # shape: (f, h, w, c)
        else:
            video_dict[self.video_column] = video_reader.get_batch(batch_index).asnumpy()  # shape: (f, h, w, c)

        del video_reader

        video_dict = self._apply_transforms(video_dict, self._transforms)

        return tuple(video_dict[c] for c in self.output_columns)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns:
            video: preprocessed video frames in shape (f, c, h, w), normalized to [-1, 1]
        """
        try:
            data = self.get_batch(idx)
            if self.prev_ok_sample is None or self.require_update_prev:
                self.prev_ok_sample = data
                self.require_update_prev = False
        except Exception as e:
            logger.warning(f"Fail to get sample of idx {idx}. The corrupted video will be replaced.")
            print(f"\tError msg: {e}", flush=True)
            assert self.prev_ok_sample is not None
            data = self.prev_ok_sample  # unless the first sample is already not ok
            self.require_update_prev = True

            if idx >= len(self.dataset):
                raise IndexError  # needed for checking the end of dataset iteration

        return data

    @staticmethod
    def train_transforms(
        size: Tuple[int, int],
        random_crop: bool = False,
        random_flip: bool = False,
        interpolation: int = cv2.INTER_CUBIC,
    ) -> List[dict]:
        if random_crop:
            raise NotImplementedError("Random cropping is not supported yet.")

        operations = [ResizeCrop(size, interpolation=interpolation)]
        if random_flip:
            operations.append(HorizontalFlip(p=0.5))
        operations.append(lambda x: np.transpose(x, (3, 0, 1, 2)))  # T H W C -> C T H W
        operations.append(lambda x: x.astype(np.float32) / 127.5 - 1)

        return [{"operations": operations, "input_columns": ["video"]}]


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
    from mindone.data import create_dataloader

    test = "dl"
    if test == "dataset":
        ds_config = dict(
            folder="../videocomposer/datasets/webvid5",
            random_crop=True,
            flip=True,
        )
        # test source dataset
        ds = VideoDataset(**ds_config)
        sample = ds.__getitem__(0)
        print(sample.shape)

        check_sanity(sample)
    else:
        import math
        import time

        from tqdm import tqdm

        ds_config = dict(
            csv_path="../videocomposer/datasets/webvid5_copy.csv",
            folder="../videocomposer/datasets/webvid5",
            sample_n_frames=17,
            size=128,
            crop_size=128,
        )
        ds = VideoDataset(**ds_config)
        bt = BatchTransform(mixed_strategy="mixed_video_random", mixed_image_ratio=0.2)

        # test loader
        dl = create_dataloader(ds, batch_size=4, batch_transforms={"operations": bt, "input_columns": ["video"]})

        num_batches = dl.get_dataset_size()
        # ms.set_context(mode=0)
        print(num_batches)

        steps = 50
        iterator = dl.create_dict_iterator(100)  # create 100 repeats
        tot = 0

        progress_bar = tqdm(range(steps))
        progress_bar.set_description("Steps")

        start = time.time()
        for epoch in range(math.ceil(steps / num_batches)):
            for i, batch in enumerate(iterator):
                print("epoch", epoch, "step", i)
                dur = time.time() - start
                tot += dur

                if epoch * num_batches + i < 50:
                    for k in batch:
                        print(k, batch[k].shape, batch[k].dtype)  # , batch[k].min(), batch[k].max())
                    print(f"time cost: {dur * 1000} ms")

                progress_bar.update(1)
                if i + 1 > steps:  # in case the data size is too large
                    break
                start = time.time()

        mean = tot / steps
        print("Avg batch loading time: ", mean)
