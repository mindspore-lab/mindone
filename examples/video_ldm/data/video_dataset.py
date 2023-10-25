import random
from typing import Tuple

import numpy as np

from .video_reader import VideoReader


def get_video_paths_captions(metadata_dir: str, data_dir: str):
    return [], []


class VideoDataset:
    def __init__(self, data_dir: str, metadata_dir: str, frames: int = 8, step: int = 1):
        self._videos, self._captions = get_video_paths_captions(metadata_dir, data_dir)
        self._frames = frames
        self._step = step
        self.output_columns = ["frames", "caption"]

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, str]:
        with VideoReader(self._videos[idx]) as reader:
            try:
                start_pos = random.randint(0, len(reader) - self._frames * self._step)
            except ValueError:
                raise ValueError(
                    f"Number of frames to fetch ({self._frames * self._step})"
                    f" must be less than video length ({len(reader)})."
                )
            frames = reader.fetch_frames(num=self._frames, start_pos=start_pos, step=self._step)

        return frames, self._captions[idx]

    def __len__(self):
        return len(self._videos)
