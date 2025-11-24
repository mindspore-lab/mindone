from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import decord
import numpy as np
import pandas as pd

from mindspore.dataset import GeneratorDataset, vision
from mindspore.dataset.vision import Inter

from mindone.diffusers.utils import get_logger

logger = get_logger(__name__)


class VideoDataset:
    def __init__(
        self,
        data_root: str,
        dataset_file: Optional[str] = None,
        caption_column: str = "text",
        video_column: str = "video",
        frame_num: int = 81,
        size_buckets: Tuple[Tuple[int, int], ...] = ((1280, 704), (704, 1280)),
        image_to_video: bool = False,
        text_drop_prob: float = 0.1,
    ) -> None:
        super().__init__()

        self.data_root = Path(data_root)
        self.dataset_file = dataset_file
        self.caption_column = caption_column
        self.video_column = video_column
        self.frame_num = frame_num
        self.size_buckets = size_buckets
        self.image_to_video = image_to_video
        self.text_drop_prob = text_drop_prob

        if self.image_to_video:
            raise NotImplementedError("image_to_video is not supported yet.")

        # Two methods of loading data are supported.
        #   - Using a CSV: caption_column and video_column must be some column in the CSV. One could
        #     make use of other columns too, such as a motion score or aesthetic score, by modifying the
        #     logic in CSV processing.
        #   - Using two files containing line-separate captions and relative paths to videos.
        # For a more detailed explanation about preparing dataset format, checkout the README.
        if dataset_file is None:
            (
                self.prompts,
                self.video_paths,
            ) = self._load_dataset_from_local_path()
        else:
            (
                self.prompts,
                self.video_paths,
            ) = self._load_dataset_from_csv()

        if len(self.video_paths) != len(self.prompts):
            raise ValueError(
                f"Expected length of prompts and videos to be the same but found len(self.prompts)={len(self.prompts)} and len(self.video_paths)={len(self.video_paths)}. Please ensure that the number of caption prompts and videos match in your dataset."  # noqa: E501
            )

        self.valid_indices = set()

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        try:
            item = self.read_video(index)
            self.valid_indices.add(index)
        except Exception as e:
            logger.warning(
                f"Failed to read video at index {index} due to error: {e}. Sampling a random valid video instead."
            )
            item = self.read_video(np.random.choice(list(self.valid_indices)) if self.valid_indices else 0)
        return item

    def read_video(self, index: int) -> Dict[str, Any]:
        if np.random.rand() < self.text_drop_prob:
            prompt = ""
        else:
            prompt = self.prompts[index]

        video_path = self.video_paths[index]

        image, video = self._preprocess_video(video_path)

        item = {"prompt": prompt, "video": video}
        if image is not None:
            item["image"] = image
        return tuple(item.values())

    def _load_dataset_from_local_path(self) -> Tuple[List[str], List[str]]:
        if not self.data_root.exists():
            raise ValueError("Root folder for videos does not exist")

        prompt_path = self.data_root.joinpath(self.caption_column)
        video_path = self.data_root.joinpath(self.video_column)

        if not prompt_path.exists() or not prompt_path.is_file():
            raise ValueError(
                "Expected `--caption_column` to be path to a file in `--data_root` containing line-separated text prompts."
            )
        if not video_path.exists() or not video_path.is_file():
            raise ValueError(
                "Expected `--video_column` to be path to a file in `--data_root` containing line-separated paths to video data in the same directory."
            )

        with open(prompt_path, "r", encoding="utf-8") as file:
            prompts = [line.strip() for line in file.readlines() if len(line.strip()) > 0]
        with open(video_path, "r", encoding="utf-8") as file:
            video_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]

        if any(not path.is_file() for path in video_paths):
            raise ValueError(
                f"Expected `self.video_column={self.video_column}` to be a path to a file in `self.data_root={self.data_root}` containing line-separated paths to video data but found atleast one path that is not a valid file."  # noqa: E501
            )

        return prompts, video_paths

    def _load_dataset_from_csv(self) -> Tuple[List[str], List[str]]:
        df = pd.read_csv(self.dataset_file)
        prompts = df[self.caption_column].tolist()
        video_paths = df[self.video_column].tolist()
        video_paths = [self.data_root.joinpath(line.strip()) for line in video_paths]

        if any(not path.is_file() for path in video_paths):
            raise ValueError(
                f"Expected `self.video_column={self.video_column}` to be a path to a file in `self.data_root={self.data_root}` containing line-separated paths to video data but found atleast one path that is not a valid file."  # noqa: E501
            )

        return prompts, video_paths

    def _preprocess_video(self, path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        video_reader = decord.VideoReader(uri=path.as_posix())
        video_num_frames = len(video_reader)

        if video_num_frames < self.frame_num:
            raise ValueError(
                f"Video at {path} has only {video_num_frames} frames, but expected at least {self.frame_num} frames."
            )

        # random samples
        start_ind = np.random.randint(0, video_num_frames - self.frame_num)
        indices = np.arange(start_ind, start_ind + self.frame_num)
        frames = video_reader.get_batch(indices).asnumpy()
        bucket = self._find_nearest_resolution(frames.shape[2], frames.shape[3])
        frames = np.stack(
            [vision.Resize(size=(bucket[1], bucket[0]), interpolation=Inter.BICUBIC)(frame) for frame in frames], axis=0
        )
        frames = frames.transpose(3, 0, 1, 2)  # C, F, H, W

        # normalize to [−1, 1]
        frames = frames.astype(np.float32) / 127.5 - 1.0
        image = frames[:1].copy() if self.image_to_video else None

        return image, frames

    def _find_nearest_resolution(self, height, width):
        return min(self.size_buckets, key=lambda x: abs(x[0] - width) + abs(x[1] - height))


def create_video_dataset(
    data_root: str,
    dataset_file: Optional[str] = None,
    caption_column: str = "text",
    video_column: str = "video",
    frame_num: int = 81,
    size_buckets: Tuple[Tuple[int, int], ...] = ((1280, 704), (704, 1280)),
    image_to_video: bool = False,
    batch_size: int = 1,
    num_parallel_workers: int = 4,
    num_shards=None,
    shard_id=None,
    text_drop_prob: float = 0.1,
):
    dataset = VideoDataset(
        data_root=data_root,
        dataset_file=dataset_file,
        caption_column=caption_column,
        video_column=video_column,
        frame_num=frame_num,
        size_buckets=size_buckets,
        image_to_video=image_to_video,
        text_drop_prob=text_drop_prob,
    )

    column_names = ["prompt", "video"]
    if image_to_video:
        column_names.append("image")

    generator = GeneratorDataset(
        dataset,
        column_names=column_names,
        num_parallel_workers=num_parallel_workers,
        num_shards=num_shards,
        shard_id=shard_id,
    )
    generator = generator.batch(batch_size, drop_remainder=True, num_parallel_workers=2)
    return generator
