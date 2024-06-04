import csv
import logging
import os
import time
from typing import Dict

import albumentations
import cv2
import numpy as np
from decord import VideoReader

import mindspore as ms

logger = logging.getLogger()


class _ResizeByMaxValue:
    def __init__(
        self, max_size: int = 256, vae_scale: int = 8, patch_size: int = 2, interpolation: str = "bicubic"
    ) -> None:
        self.max_size = max_size
        self.scale = vae_scale * patch_size
        self.interpolation = {"bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC}[interpolation]

    def __call__(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        h, w, _ = image.shape
        image_area = w * h
        max_area = self.max_size * self.max_size
        if image_area > max_area:
            ratio = max_area / image_area
            new_w = w * np.sqrt(ratio)
            new_h = h * np.sqrt(ratio)
        else:
            new_w = w
            new_h = h

        round_w, round_h = (np.round(np.array([new_w, new_h]) / self.scale) * self.scale).astype(int).tolist()
        if round_w * round_h > max_area:
            round_w, round_h = (np.floor(np.array([new_w, new_h]) / self.scale) * self.scale).astype(int).tolist()

        round_w, round_h = max(round_w, self.scale), max(round_h, self.scale)
        image = cv2.resize(image, (round_w, round_h), interpolation=self.interpolation)
        return dict(image=image)


def create_video_transforms(h, w, interpolation="bicubic"):
    """
    h, w : target resize height, weight
    """
    # expect rgb image in range 0-255, shape (h w c)
    from albumentations import CenterCrop, SmallestMaxSize

    mapping = {"bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC}
    pixel_transforms = albumentations.Compose(
        [
            SmallestMaxSize(max_size=h, interpolation=mapping[interpolation]),
            CenterCrop(h, w),
        ],
    )

    return pixel_transforms


class VideoDataset:
    def __init__(
        self,
        csv_path,
        video_folder,
        video_column="video",
        caption_column="caption",
        sample_size=512,
        sample_stride=1,
        return_frame_data=False,
        micro_batch_size=None,
        resize_by_max_value=False,
    ):
        logger.info(f"loading annotations from {csv_path} ...")
        with open(csv_path, "r") as csvfile:
            self.dataset = list(csv.DictReader(csvfile))

        self.length = len(self.dataset)
        logger.info(f"Num data samples: {self.length}")

        self.video_folder = video_folder
        self.caption_column = caption_column
        self.video_column = video_column
        self.return_frame_data = return_frame_data
        self.sample_stride = sample_stride
        self.micro_batch_size = micro_batch_size

        if resize_by_max_value:
            if isinstance(sample_size, (tuple, list)):
                if len(sample_size) != 1:
                    raise ValueError(f"`sample_size` must be length 1 list, but get `{sample_size}`.")
            self.pixel_transforms = _ResizeByMaxValue(max_size=sample_size[0], interpolation="bicubic")
        else:
            if isinstance(sample_size, (int, float)):
                sample_size = (sample_size, sample_size)
            elif isinstance(sample_size, (tuple, list)):
                if len(sample_size) == 1:
                    sample_size = list(sample_size) * 2
            self.pixel_transforms = create_video_transforms(
                sample_size[0],
                sample_size[1],
                interpolation="bicubic",
            )

    def __len__(self):
        return self.length

    def apply_transform(self, pixel_values):
        # pixel value: (f, h, w, 3) -> transforms -> (f 3 h' w')
        num_frames = len(pixel_values)
        output = []
        # TODO: use parallel transform
        for i in range(num_frames):
            trans = self.pixel_transforms(image=pixel_values[i])["image"]
            output.append(trans)
        pixel_values = np.stack(output, axis=0)
        # (f h w c) -> (f c h w)
        pixel_values = np.transpose(pixel_values, (0, 3, 1, 2))
        return pixel_values

    def get_video_frames_in_batch(self, video_path, micro_batch_size=64, sample_stride=1, do_transform=True):
        if not video_path.endswith(".mp4") or video_path.endswith(".gif"):
            if video_path[-4] != ".":
                video_path = video_path + ".mp4"
            else:
                raise ValueError(f"video file format is not verified: {video_path}")

        video_reader = VideoReader(video_path)
        video_length = len(video_reader)
        # print("D--: video_length ", video_length)

        bs = micro_batch_size
        for i in range(0, video_length, bs):
            frame_indice = list(range(i, min(i + bs, video_length), sample_stride))
            pixel_values = video_reader.get_batch(frame_indice).asnumpy()  # shape: (f, h, w, c)
            if do_transform:
                pixel_values = self.apply_transform(pixel_values)
                pixel_values = (pixel_values / 127.5 - 1.0).astype(np.float32)

            yield pixel_values

    def __getitem__(self, idx):
        row = self.dataset[idx]
        caption = row[self.caption_column]
        video_path = row[self.video_column]
        # video_path = os.path.join(self.video_folder, video_path)

        if self.return_frame_data:
            all_frames = []
            for pixel_values in self.get_video_frames_in_batch(
                os.path.join(self.video_folder, video_path), self.micro_batch_size, self.sample_stride
            ):
                all_frames.append(pixel_values)
            all_frames = np.concatenate(all_frames, axis=0)

            return video_path, caption, all_frames
        else:
            return video_path, caption


def create_dataloader(
    ds_config,
    batch_size,
    ds_name="video",
    num_parallel_workers=12,
    max_rowsize=32,
    shuffle=False,
    device_num=1,
    rank_id=0,
    drop_remainder=False,
    return_dataset=False,
):
    if ds_name == "video":
        dataset = VideoDataset(**ds_config)
        if ds_config["return_frame_data"]:
            column_names = ["video_path", "caption", "frame_data"]
        else:
            column_names = ["video_path", "caption"]
    else:
        raise NotImplementedError

    dataloader = ms.dataset.GeneratorDataset(
        source=dataset,
        column_names=column_names,
        num_shards=device_num,
        shard_id=rank_id,
        python_multiprocessing=True,
        shuffle=shuffle,
        num_parallel_workers=num_parallel_workers,
        max_rowsize=max_rowsize,
    )

    dl = dataloader.batch(
        batch_size,
        drop_remainder=drop_remainder,
    )
    if return_dataset:
        return dl, dataset
    else:
        return dl, None


if __name__ == "__main__":
    return_frame_data = True
    video_folder = "../videocomposer/datasets/webvid5"
    ds_config = dict(
        csv_path="../videocomposer/datasets/webvid5/video_caption.csv",
        video_folder=video_folder,
        video_column="video",
        caption_column="caption",
        sample_size=512,
        return_frame_data=return_frame_data,
        sample_stride=1,
        micro_batch_size=64,
    )
    dl, ds = create_dataloader(
        ds_config,
        batch_size=1,
        max_rowsize=512,
        return_dataset=True,
    )

    ds_iter = dl.create_dict_iterator(1, output_numpy=True)

    for step, data in enumerate(ds_iter):
        vp = data["video_path"]
        cap = data["caption"]
        if step == 1:
            start = time.time()
        print(vp[0], cap[0])
        if return_frame_data:
            frame_data = data["frame_data"]
        else:
            all_frames = []
            num_videos = data["video_path"].shape[0]
            print(num_videos)
            video_path = os.path.join(video_folder, vp[0])
            for clip in ds.get_video_frames_in_batch(video_path, micro_batch_size=64, sample_stride=1):
                all_frames.append(clip)
            all_frames = np.concatenate(all_frames, axis=0)
            frame_data = [all_frames]
        print(frame_data[0].shape)

    cost = time.time() - start
    print(f"Time cost: {cost:.3f}")
