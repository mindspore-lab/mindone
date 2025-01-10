import logging
import os
import random
import sys
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from mindspore.dataset.vision import CenterCrop, Resize

sys.path.append("../../")  # FIXME: remove in future when mindone is ready for install
from mindone.data import BaseDataset, VideoReader

_logger = logging.getLogger(__name__)


class VideoDataset(BaseDataset):
    def __init__(self, data_dir: str, metadata: str, frames: int, step: int = 1):
        self._data = self._read_data(data_dir, metadata)
        self._frames = frames
        self._step = step
        self._filter_videos()
        self.output_columns = ["frames", "fps_id", "motion_bucket_id", "cond_aug"]

    @staticmethod
    def _read_data(data_dir: str, metadata: str) -> List[dict]:
        data = []
        with open(metadata, "r") as file:
            file.readline()  # skip the header
            for line in file:
                line = line.strip().split(",", maxsplit=2)
                data.append(
                    {"path": os.path.join(data_dir, line[0]), "length": int(line[1]), "motion_bucket_id": int(line[2])}
                )
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
            data.update({"fps": reader.fps / self._step, "width": reader.shape[0], "height": reader.shape[1]})

        noise_strength = np.random.lognormal(-3.0, 0.5**2)

        return data["frames"], data["fps"] - 1, data["motion_bucket_id"], noise_strength

    def __len__(self):
        return len(self._data)

    def train_transforms(self, tokenizer: Optional[Callable[[str], np.ndarray]] = None) -> List[dict]:
        return [
            {
                "operations": [
                    Resize(576),
                    CenterCrop((576, 1024)),
                    lambda x: np.transpose(x, (0, 3, 1, 2)),  # ms.HWC2CHW() doesn't support 4D data
                    lambda x: (x / 127.5 - 1.0).astype(np.float32),
                ],
                "input_columns": ["frames"],
            },
            {
                "operations": [
                    lambda frames, aug: (
                        frames,
                        frames[0],
                        frames[0] + aug * np.random.randn(*frames[0].shape).astype(np.float32),
                        np.tile(aug, (self._frames, 1)).astype(np.float32),
                    )
                ],
                "input_columns": ["frames", "cond_aug"],
                "output_columns": ["frames", "cond_frames_without_noise", "cond_frames", "cond_aug"],
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
                    lambda fps_id, motion_bucket_id: (
                        np.tile(fps_id, (self._frames, 1)).astype(np.float32),
                        np.tile(motion_bucket_id, (self._frames, 1)).astype(np.float32),
                    )
                ],
                "input_columns": ["fps_id", "motion_bucket_id"],
            },
        ]
