import csv
import os
import random

import albumentations
import cv2
import numpy as np
from decord import VideoReader

from mindone.metrics.utils import get_video_path


def create_video_transforms(h, w, num_frames, interpolation="bicubic", backend="al", disable_flip=True):
    """
    pipeline: flip -> resize -> crop
    h, w : target resize height, weight
    NOTE: we change interpolation to bicubic for its better precision and used in SD. TODO: check impact on performance
    """
    if backend == "al":
        # expect rgb image in range 0-255, shape (h w c)
        from albumentations import CenterCrop, HorizontalFlip, SmallestMaxSize

        targets = {"image{}".format(i): "image" for i in range(num_frames)}
        mapping = {"bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC}
        if disable_flip:
            # flip is not proper for horizontal motion learning
            pixel_transforms = albumentations.Compose(
                [
                    SmallestMaxSize(max_size=h, interpolation=mapping[interpolation]),
                    CenterCrop(h, w),
                ],
                additional_targets=targets,
            )
        else:
            # originally used in torch ad code, but not good for non-square video data
            # also conflict the learning of left-right camera movement
            pixel_transforms = albumentations.Compose(
                [
                    HorizontalFlip(p=0.5),
                    SmallestMaxSize(max_size=h, interpolation=mapping[interpolation]),
                    CenterCrop(h, w),
                ],
                additional_targets=targets,
            )
    elif backend == "ms":
        # TODO: MindData doesn't support batch transform. can NOT make sure all frames are flipped the same
        from mindspore.dataset import transforms, vision
        from mindspore.dataset.vision import Inter

        from .transforms import CenterCrop

        mapping = {"bilinear": Inter.BILINEAR, "bicubic": Inter.BICUBIC}
        pixel_transforms = transforms.Compose(
            [
                vision.RandomHorizontalFlip(),
                vision.Resize(h, interpolation=mapping[interpolation]),
                CenterCrop(h, w),
            ]
        )
    else:
        raise NotImplementedError

    return pixel_transforms


class TextVideoDataset:
    def __init__(
        self,
        video_folder,
        csv_path=None,
        sample_size=512,
        sample_stride=4,
        sample_n_frames=16,
        is_image=False,
        transform_backend="al",  # ms,  al
        video_column="video",
        caption_column="caption",
        disable_flip=True,
    ):
        """
        text_emb_folder: root dir of text embed saved in npz files. Expected to have the same file name and directory strcutre as videos. e.g.
            video_folder:
                001.mp4
                002.mp4

        """
        if csv_path is not None:
            print(f"loading annotations from {csv_path} ...")
            with open(csv_path, "r") as csvfile:
                self.dataset = list(csv.DictReader(csvfile))
            self.length = len(self.dataset)
            print(f"Num data samples: {self.length}")
            self.video_folder = video_folder
        else:
            self.video_folder = get_video_path(video_folder)
            self.dataset = None
            self.length = len(self.video_folder)
            print(f"Num data samples: {self.length}")

        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames
        self.is_image = is_image
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)

        # it should match the transformation used in SD/VAE pretraining, especially for normalization
        self.pixel_transforms = create_video_transforms(
            sample_size[0],
            sample_size[1],
            sample_n_frames,
            interpolation="bicubic",
            backend=transform_backend,
            disable_flip=disable_flip,
        )
        self.transform_backend = transform_backend
        self.video_column = video_column
        self.caption_column = caption_column

    def get_batch(self, idx):
        if self.dataset is not None:
            # get video raw pixels (batch of frame) and its caption
            video_dict = self.dataset[idx]
            video_fn, caption = video_dict[self.video_column], video_dict[self.caption_column]
            video_path = os.path.join(self.video_folder, video_fn)

            # in case missing .mp4 in csv file
            if not video_path.endswith(".mp4") or video_path.endswith(".gif"):
                if video_path[-4] != ".":
                    video_path = video_path + ".mp4"
                else:
                    raise ValueError(f"video file format is not verified: {video_path}")
        else:
            video_path = self.video_folder[idx]
            caption = None

        video_reader = VideoReader(video_path)

        video_length = len(video_reader)

        if not self.is_image:
            clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
            start_idx = 0
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        else:
            batch_index = [random.randint(0, video_length - 1)]

        if video_path.endswith(".gif"):
            pixel_values = video_reader[batch_index]  # shape: (f, h, w, c)
        else:
            pixel_values = video_reader.get_batch(batch_index).asnumpy()  # shape: (f, h, w, c)
        # print("D--: video clip shape ", pixel_values.shape, pixel_values.dtype)
        # pixel_values = pixel_values / 255. # let's keep uint8 for fast compute
        del video_reader

        return pixel_values, caption

    def __len__(self):
        return self.length

    def apply_transform(self, pixel_values):
        # pixel value: (f, h, w, 3) -> transforms -> (f 3 h' w')
        if self.transform_backend == "al":
            inputs = {"image": pixel_values[0]}
            num_frames = len(pixel_values)
            for i in range(num_frames - 1):
                inputs[f"image{i}"] = pixel_values[i + 1]

            output = self.pixel_transforms(**inputs)

            pixel_values = np.stack(list(output.values()), axis=0)
            # (f h w c) -> (f c h w)
            pixel_values = np.transpose(pixel_values, (0, 3, 1, 2))
        else:
            raise NotImplementedError
        return pixel_values

    def get_video_frame(self, idx):
        """
        Returns:
        - video (np.float32): preprocessed video frames in shape (f, c, h, w) for vae encoding
        """

        pixel_values, caption = self.get_batch(idx)

        # apply visual transform and normalization
        pixel_values = self.apply_transform(pixel_values)

        if self.is_image:
            pixel_values = pixel_values[0]

        pixel_values = (pixel_values / 127.5 - 1.0).astype(np.float32)

        return pixel_values, caption
