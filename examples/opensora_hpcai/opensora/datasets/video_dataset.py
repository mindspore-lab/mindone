import csv
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import albumentations as A
import cv2
import imageio
import numpy as np
from decord import VideoReader
from tqdm import tqdm

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


class MinCropAndResize:
    """
    (tar_h, tar_w) as a cropping box centered at the image center point,
    you can resize it while keeping its AR until it touches one side of the image boundary.
    Then crop it and resize to (tar_h, tar_w)
    """

    def __init__(self, tar_h, tar_w, interpolation="bicubic"):
        self.th = tar_h
        self.tw = tar_w
        self.interpolation = interpolation

    def __call__(self, image):
        h, w, c = image.shape
        if (self.tw / self.th) > (w / h):
            scale = w / self.tw
        else:
            scale = h / self.th
        crop_h = int(scale * self.th)
        crop_w = int(scale * self.tw)

        trans = A.Compose([A.CenterCrop(crop_h, crop_w), A.Resize(self.th, self.tw, interpolation=self.interpolation)])
        out = trans(image=image)["image"]

        return {"image": out}


def create_video_transforms(h, w, interpolation="bicubic", name="center"):
    """
    h, w : target resize height, weight
    if h < w: (512, 1024)
        if ch < cw: (512, 768)
            cannot crop, unless crop and resize

    """
    # expect rgb image in range 0-255, shape (h w c)
    from albumentations import CenterCrop, SmallestMaxSize

    mapping = {"bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC}
    if name == "center":
        pixel_transforms = A.Compose(
            [
                SmallestMaxSize(max_size=h, interpolation=mapping[interpolation]),
                CenterCrop(h, w),
            ],
        )
    elif name == "crop_resize":
        pixel_transforms = A.Compose(
            [
                MinCropAndResize(h, w, interpolation=mapping[interpolation]),
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
        transform_name="center",
        filter_data: bool = False,
    ):
        logger.info(f"loading annotations from {csv_path} ...")
        self.dataset = self._read_data(video_folder, csv_path, video_column, filter_data)

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
                name=transform_name,
            )

    @staticmethod
    def _read_data(data_dir: str, csv_path: str, video_column: str, filter_data: bool = False) -> List[dict]:
        def _filter_data(sample_):
            if not os.path.isfile(sample_[video_column]):
                logger.warning(f"Video not found: {sample_[video_column]}")
                return None
            return sample_

        with open(csv_path, "r") as csv_file:
            try:
                data = [
                    {**item, video_column: os.path.join(data_dir, item[video_column])}
                    for item in csv.DictReader(csv_file)
                ]
            except KeyError as e:
                logger.error(
                    f"The video column `{video_column}` was not found."
                    f" Please specify the correct name with `--video_column` argument."
                )
                raise e

        if filter_data:
            with ThreadPoolExecutor(max_workers=10) as executor:
                data = [
                    item
                    for item in tqdm(executor.map(_filter_data, data), total=len(data), desc="Filtering data")
                    if item is not None
                ]

        logger.info(f"Number of data samples: {len(data)}")
        return data

    def __len__(self):
        return len(self.dataset)

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
        fps = video_reader.get_avg_fps()

        bs = micro_batch_size
        for i in range(0, video_length, bs):
            frame_indice = list(range(i, min(i + bs, video_length), sample_stride))
            pixel_values = video_reader.get_batch(frame_indice).asnumpy()  # shape: (f, h, w, c)
            ori_size = pixel_values.shape[-3:-1]
            if do_transform:
                pixel_values = self.apply_transform(pixel_values)

                # efficient implement
                pixel_values = np.divide(pixel_values, 127.5, dtype=np.float32)
                pixel_values = np.subtract(pixel_values, 1.0, dtype=np.float32)

            yield pixel_values, fps, ori_size

    def __getitem__(self, idx):
        row = self.dataset[idx]
        caption = row[self.caption_column]
        video_path = row[self.video_column]
        # video_path = os.path.join(self.video_folder, video_path)

        if self.return_frame_data:
            all_frames = []
            fps, ori_size = None, None
            for pixel_values, fps, ori_size in self.get_video_frames_in_batch(
                video_path, self.micro_batch_size, self.sample_stride
            ):
                all_frames.append(pixel_values)
                fps = fps
                ori_size = ori_size
            all_frames = np.concatenate(all_frames, axis=0)

            return video_path, caption, all_frames, fps, ori_size
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
            column_names = ["video_path", "caption", "frame_data", "fps", "ori_size"]
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


def check_sanity(x, save_fp="./tmp.gif"):
    # reverse normalization and visulaize the transformed video
    # (f, c, h, w) -> (f, h, w, c)
    if len(x.shape) == 3:
        x = np.expand_dims(x, axis=0)
    x = np.transpose(x, (0, 2, 3, 1))

    x = (x + 1.0) / 2.0  # -1,1 -> 0,1
    x = (x * 255).astype(np.uint8)

    imageio.mimsave(save_fp, x, duration=1 / 8.0, loop=1)


if __name__ == "__main__":
    return_frame_data = True
    video_folder = "../videocomposer/datasets/webvid5"
    ds_config = dict(
        csv_path="../videocomposer/datasets/webvid5/video_caption.csv",
        video_folder=video_folder,
        video_column="video",
        caption_column="caption",
        sample_size=(512, 1024),
        return_frame_data=return_frame_data,
        sample_stride=1,
        micro_batch_size=32,
        transform_name="crop_resize",
        filter_data=True,
    )
    dl, ds = create_dataloader(
        ds_config,
        batch_size=1,
        max_rowsize=256,
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
            fps = np.array(data["fps"], dtype=np.float32)
            ori_size = np.array(data["ori_size"], dtype=np.int32)
            print("fps: ", fps)
            print("ori size: ", ori_size)
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
        check_sanity(frame_data[0][:64])

    cost = time.time() - start
    print(f"Time cost: {cost:.3f}")
