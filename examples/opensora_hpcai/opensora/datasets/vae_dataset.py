import copy
import csv
import glob
import logging
import os
import random

import albumentations
import cv2
import imageio
import numpy as np
from decord import VideoReader

import mindspore as ms

logger = logging.getLogger()


def create_video_transforms(
    size=384, crop_size=256, interpolation="bicubic", backend="al", random_crop=False, flip=False, num_frames=None
):
    if backend == "al":
        # expect rgb image in range 0-255, shape (h w c)
        from albumentations import CenterCrop, HorizontalFlip, RandomCrop, SmallestMaxSize

        # NOTE: to ensure augment all frames in a video in the same way.
        assert num_frames is not None, "num_frames must be parsed"
        targets = {"image{}".format(i): "image" for i in range(num_frames)}
        mapping = {"bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC}
        transforms = [
            SmallestMaxSize(max_size=size, interpolation=mapping[interpolation]),
            CenterCrop(crop_size, crop_size) if not random_crop else RandomCrop(crop_size, crop_size),
        ]
        if flip:
            transforms += [HorizontalFlip(p=0.5)]

        pixel_transforms = albumentations.Compose(
            transforms,
            additional_targets=targets,
        )
    else:
        raise NotImplementedError

    return pixel_transforms


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
        flip=False,
        sample_stride=4,
        sample_n_frames=16,
        return_image=False,
        transform_backend="al",
        video_column="video",
    ):
        """
        size: image resize size
        crop_size: crop size after resize operation
        """
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

        self.pixel_transforms = create_video_transforms(
            size=size,
            crop_size=crop_size,
            random_crop=random_crop,
            flip=flip,
            num_frames=sample_n_frames,
        )
        self.transform_backend = transform_backend
        self.video_column = video_column

        # prepare replacement data
        max_attempts = 100
        self.prev_ok_sample = self.get_replace_data(max_attempts)
        self.require_update_prev = False

    def get_replace_data(self, max_attempts=100):
        replace_data = None
        attempts = min(max_attempts, self.length)
        for idx in range(attempts):
            try:
                pixel_values = self.get_batch(idx)
                replace_data = copy.deepcopy(pixel_values)
                break
            except Exception as e:
                print("\tError msg: {}".format(e))

        assert replace_data is not None, f"Fail to preload sample in {attempts} attempts."

        return replace_data

    def get_batch(self, idx):
        # get video raw pixels (batch of frame) and its caption
        if self.read_from_csv:
            video_dict = self.dataset[idx]
            video_fn = video_dict[list(video_dict.keys())[0]]
            video_path = os.path.join(self.data_folder, video_fn)
        else:
            video_path = self.dataset[idx]

        video_reader = VideoReader(video_path)

        video_length = len(video_reader)

        if not self.return_image:
            clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
            start_idx = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        else:
            batch_index = [random.randint(0, video_length - 1)]

        if video_path.endswith(".gif"):
            pixel_values = video_reader[batch_index]  # shape: (f, h, w, c)
        else:
            pixel_values = video_reader.get_batch(batch_index).asnumpy()  # shape: (f, h, w, c)

        del video_reader

        return pixel_values

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Returns:
            video: preprocessed video frames in shape (f, c, h, w), normalized to [-1, 1]
        """
        try:
            pixel_values = self.get_batch(idx)
            if (self.prev_ok_sample is None) or (self.require_update_prev):
                self.prev_ok_sample = copy.deepcopy(pixel_values)
                self.require_update_prev = False
        except Exception as e:
            logger.warning(f"Fail to get sample of idx {idx}. The corrupted video will be replaced.")
            print("\tError msg: {}".format(e), flush=True)
            assert self.prev_ok_sample is not None
            pixel_values = self.prev_ok_sample  # unless the first sample is already not ok
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
    def __init__(self, mixed_strategy, mixed_image_ratio=0.2):
        self.mixed_strategy = mixed_strategy
        self.mixed_image_ratio = mixed_image_ratio

    def __call__(self, x):
        # x: (bs, c, t, h, w)
        if self.mixed_strategy == "mixed_video_image":
            if random.random() < self.mixed_image_ratio:
                x = x[:, :, :1, :, :]
        elif self.mixed_strategy == "mixed_video_random":
            # TODO: somehow it's slow. consider do it with tensor in NetWithLoss
            length = random.randint(1, x.shape[2])
            x = x[:, :, :length, :, :]
        elif self.mixed_strategy == "image_only":
            x = x[:, :, :1, :, :]
        else:
            raise ValueError
        return x


def create_dataloader(
    ds_config,
    batch_size,
    mixed_strategy=None,
    mixed_image_ratio=0.0,
    num_parallel_workers=12,
    max_rowsize=32,
    shuffle=True,
    device_num=1,
    rank_id=0,
    drop_remainder=True,
):
    """
    Args:
        mixed_strategy:
            None - all output batches are videoes [bs, c, T, h, w]
            mixed_video_image - with prob of mixed_image_ratio, output batch are images [b, c, 1, h, w]
            mixed_video_random - output batch has a random number of frames [bs, c, t, h, w],  t is the same of samples in a batch
        mixed_image_ratio:
        ds_config, dataset config, args for ImageDataset or VideoDataset
        ds_name: dataset name, image or video
    """
    dataset = VideoDataset(**ds_config)
    print("Total number of samples: ", len(dataset))

    # Larger value leads to more memory consumption. Default: 16
    # prefetch_size = config.get("prefetch_size", 16)
    # ms.dataset.config.set_prefetch_size(prefetch_size)

    dataloader = ms.dataset.GeneratorDataset(
        source=dataset,
        column_names=["video"],
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

    if mixed_strategy is not None:
        batch_map_fn = BatchTransform(mixed_strategy, mixed_image_ratio)
        dl = dl.map(
            operations=batch_map_fn,
            input_columns=["video"],
            num_parallel_workers=1,
        )

    return dl


if __name__ == "__main__":
    test = "dl"
    if test == "dataset":
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
    else:
        import math
        import time

        from tqdm import tqdm

        ds_config = dict(
            csv_path="../videocomposer/datasets/webvid5_copy.csv",
            data_folder="../videocomposer/datasets/webvid5",
            sample_n_frames=17,
            size=128,
            crop_size=128,
        )

        # test loader
        dl = create_dataloader(
            ds_config,
            4,
            mixed_strategy="mixed_video_random",
            mixed_image_ratio=0.2,
        )

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
