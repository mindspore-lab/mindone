import copy
import csv
import glob
import logging
import os
import random

import imageio
import numpy as np
from decord import VideoReader
from opensora.utils.dataset_utils import create_video_transforms
from PIL import Image, ImageSequence

import mindspore as ms

logger = logging.getLogger()


def read_gif(gif_path, mode="RGB"):
    with Image.open(gif_path) as fp:
        frames = np.array([np.array(frame.convert(mode)) for frame in ImageSequence.Iterator(fp)])
    return frames


def get_video_path_list(folder):
    # TODO: find recursively
    fmts = ["avi", "mp4", "gif"]
    out = []
    for fmt in fmts:
        out += glob.glob(os.path.join(folder, f"*.{fmt}"))
    return sorted(out)


class VideoDataset:
    def __init__(
        self,
        csv_path=None,
        data_folder=None,
        size=384,
        crop_size=256,
        random_crop=False,
        sample_stride=4,
        sample_n_frames=16,
        return_image=False,
        transform_backend="al",
        video_column="video",
        disable_flip=True,
        dynamic_sample=False,
        output_columns=["video", "path"],
    ):
        logger.info(f"loading annotations from {csv_path} ...")

        if csv_path is not None:
            with open(csv_path, "r") as csvfile:
                self.dataset = list(csv.DictReader(csvfile))
            self.read_from_csv = True
        else:
            self.dataset = get_video_path_list(data_folder)
            self.read_from_csv = False

        self.length = len(self.dataset)
        logger.info(f"Num data samples: {self.length}")
        logger.info(f"sample_n_frames: {sample_n_frames}")

        self.data_folder = data_folder
        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames
        self.return_image = return_image
        self.dynamic_sample = dynamic_sample

        self.pixel_transforms = create_video_transforms(
            size=size,
            crop_size=crop_size,
            random_crop=random_crop,
            disable_flip=disable_flip,
            num_frames=sample_n_frames,
        )
        self.transform_backend = transform_backend
        self.video_column = video_column
        self.output_columns = output_columns
        assert "video" in self.output_columns, "At least video should be returned"

        # prepare replacement data
        max_attempts = 100
        self.prev_ok_sample = self.get_replace_data(max_attempts)
        self.require_update_prev = False

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
        if self.read_from_csv:
            video_dict = self.dataset[idx]
            video_fn = video_dict[list(video_dict.keys())[0]]
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
            start_idx = random.randint(0, video_length - clip_length)
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


def create_dataloader(
    dataset,
    batch_size,
    ds_name="video",
    num_parallel_workers=12,
    max_rowsize=32,
    shuffle=True,
    device_num=1,
    rank_id=0,
    drop_remainder=True,
):
    """
    Args:
        ds_config, dataset config, args for ImageDataset or VideoDataset
        ds_name: dataset name, image or video
    """
    column_names = dataset.get("output_columns", ["video"])
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

    return dl


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
