import logging
import os
import random
import sys
from typing import Any, Callable, List, Tuple

import numpy as np

from mindspore.dataset.vision import CenterCrop, Resize, HWC2CHW

sys.path.append("../../")  # FIXME: remove in future when mindone is ready for install
from mindone.data import BaseDataset

from .video_reader import VideoReader

_logger = logging.getLogger(__name__)


class VideoDataset(BaseDataset):
    def __init__(self, data_dir: str, metadata: str, frames: int, step: int = 1, output_columns: List[str] = None):
        self._data = self._read_data(data_dir, metadata)
        self._frames = frames
        self._step = step
        self._filter_videos()
        # self.output_columns = output_columns or ["frames", "caption"]
        # self.output_columns = ["frames", "txt", "fps_id", "motion_bucket_id", "cond_aug"]
        self.output_columns = [
            "frames",
            "cond_frames_without_noise",
            "fps_id",
            "motion_bucket_id",
            "cond_frames",
            "cond_aug",
        ]

    @staticmethod
    def _read_data(data_dir: str, metadata: str) -> List[dict]:
        data = []
        with open(metadata, "r") as file:
            file.readline()  # skip the header
            for line in file:
                line = line.strip().split(",", maxsplit=2)
                data.append({"path": os.path.join(data_dir, line[0]), "length": int(line[1]), "caption": line[2]})
        return data

    def _filter_videos(self):
        min_length = self._frames * self._step
        old_len = len(self._data)
        self._data = [item for item in self._data if item["length"] >= min_length]
        if len(self._data) < old_len:
            _logger.info(
                f"Filtered out {old_len - len(self._data)} videos as they don't match the minimum length"
                f" requirement: {min_length} frames (num frames x step)"
            )

    def __getitem__(self, idx: int) -> Tuple[Any, ...]:
        data = self._data[idx].copy()
        with VideoReader(data["path"]) as reader:
            start_pos = random.randint(0, len(reader) - self._frames * self._step)
            data["frames"] = reader.fetch_frames(num=self._frames, start_pos=start_pos, step=self._step)
            data.update({"fps": reader.fps, "width": reader.shape[0], "height": reader.shape[1]})

        noise_strength = np.random.lognormal(-3.0, 0.5**2)

        cond_frames_without_noise = data["frames"][0]
        cond_frames = cond_frames_without_noise  # FIXME

        # return tuple(data[col] for col in self.output_columns)
        # return data["frames"], data["caption"], data["fps"], 127, noise_strength
        return data["frames"], cond_frames_without_noise, data["fps"], 127, cond_frames, noise_strength

    def __len__(self):
        return len(self._data)

    @staticmethod
    def train_transforms(tokenizer: Callable[[str], np.ndarray], frames_num: int) -> List[dict]:
        return [
            {
                "operations": [
                    Resize(384),
                    CenterCrop((384, 640)),
                    lambda x: np.transpose(x, (0, 3, 1, 2)),  # ms.HWC2CHW() doesn't support 4D data
                    lambda x: (x / 127.5 - 1.0).astype(np.float32),
                ],
                "input_columns": ["frames"],
            },
            # {
            #     "operations": [
            #         tokenizer,
            #         # expand the number of prompts to match the number of frames
            #         lambda prompt, length: np.tile(prompt, (frames_num, 1)),
            #     ],
            #     "input_columns": ["txt"],
            # },
            {
                "operations": [
                    Resize(384),
                    CenterCrop((384, 640)),
                    HWC2CHW(),
                    lambda x: (x / 127.5 - 1.0).astype(np.float32),
                ],
                "input_columns": ["cond_frames_without_noise"],
            },
            {
                "operations": [lambda x: np.tile(x, (frames_num, 1)).astype(np.float32)],
                "input_columns": ["fps_id"],
            },
            {
                "operations": [lambda x: np.tile(x, (frames_num, 1)).astype(np.float32)],
                "input_columns": ["motion_bucket_id"],
            },
            {
                "operations": [
                    Resize(384),
                    CenterCrop((384, 640)),
                    HWC2CHW(),
                    lambda x: (x / 127.5 - 1.0).astype(np.float32),
                ],
                "input_columns": ["cond_frames"],
            },
            {
                "operations": [lambda frames, aug: (frames + aug * np.random.randn(*frames.shape), aug)],
                "input_columns": ["cond_frames", "cond_aug"],
            },
            {
                "operations": [lambda x: np.tile(x, (frames_num, 1)).astype(np.float32)],
                "input_columns": ["cond_aug"],
            },
        ]

    def val_transforms(self, **kwargs):
        raise NotImplementedError("Validation transforms are not supported yet.")
