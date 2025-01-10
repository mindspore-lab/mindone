import copy
import csv
import logging
import os
import random

import albumentations
import cv2
import imageio
import numpy as np
from decord import VideoReader
from PIL import Image, ImageSequence

import mindspore as ms

logger = logging.getLogger()


def read_gif(gif_path, mode="RGB"):
    with Image.open(gif_path) as fp:
        frames = np.array([np.array(frame.convert(mode)) for frame in ImageSequence.Iterator(fp)])
    return frames


def create_video_transforms(h, w, num_frames, interpolation="bicubic", backend="al", disable_flip=True):
    """
    pipeline: flip -> resize -> crop
    h, w : target resize height, weight
    NOTE: we change interpolation to bicubic for its better precision and used in SD. TODO: check impact on performance
    """
    if backend == "pt":
        from torchvision import transforms
        from torchvision.transforms.functional import InterpolationMode

        mapping = {"bilinear": InterpolationMode.BILINEAR, "bicubic": InterpolationMode.BICUBIC}

        pixel_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.Resize(h, interpolation=mapping[interpolation]),
                transforms.CenterCrop((h, w)),
            ]
        )
    elif backend == "al":
        # expect rgb image in range 0-255, shape (h w c)
        from albumentations import CenterCrop, HorizontalFlip, SmallestMaxSize

        # NOTE: to ensure augment all frames in a video in the same way.
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


# TODO: rm csv_path, find csv under video_folder
class TextVideoDataset:
    def __init__(
        self,
        csv_path,
        video_folder,
        sample_size=256,
        sample_stride=4,
        sample_n_frames=16,
        is_image=False,
        transform_backend="al",  # ms, pt, al
        tokenizer=None,
        video_column="video",
        caption_column="caption",
        random_drop_text=True,
        random_drop_text_ratio=0.1,
        disable_flip=True,
        n_samples=None,
    ):
        logger.info(f"loading annotations from {csv_path} ...")
        with open(csv_path, "r") as csvfile:
            self.dataset = list(csv.DictReader(csvfile))

        if n_samples and n_samples < len(self.dataset):
            self.dataset = np.random.choice(self.dataset, n_samples, replace=False)

        self.length = len(self.dataset)
        logger.info(f"Num data samples: {self.length}")

        self.video_folder = video_folder
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
        self.tokenizer = tokenizer
        self.video_column = video_column
        self.caption_column = caption_column

        self.random_drop_text = random_drop_text
        self.random_drop_text_ratio = random_drop_text_ratio

        # prepare replacement data
        max_attempts = 100
        self.prev_ok_sample = self.get_replace_data(max_attempts)
        self.require_update_prev = False
        self.sample_size = sample_size

    def get_replace_data(self, max_attempts=100):
        replace_data = None
        attempts = min(max_attempts, self.length)
        for idx in range(attempts):
            try:
                pixel_values, caption = self.get_batch(idx)
                replace_data = copy.deepcopy((pixel_values, caption))
                break
            except Exception as e:
                print("\tError msg: {}".format(e), flush=True)

        assert replace_data is not None, f"Fail to preload sample in {attempts} attempts."

        return replace_data

    def get_batch(self, idx):
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

        # TODO: Add error data replacement!!!
        if video_path.endswith(".gif"):
            video_reader = read_gif(video_path, mode="RGB")
        else:
            video_reader = VideoReader(video_path)

        video_length = len(video_reader)

        if not self.is_image:
            clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
            start_idx = random.randint(0, video_length - clip_length)
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

    def __getitem__(self, idx):
        """
        Returns:
            tuple (video, text_data)
                - video: preprocessed video frames in shape (f, c, h, w)
                - text_data: if tokenizer provided, tokens shape (context_max_len,), otherwise text string
        """
        try:
            pixel_values, caption = self.get_batch(idx)
            h_ = pixel_values.shape[1]
            w_ = pixel_values.shape[2]
            assert h_ >= self.sample_size[0], f"size not large enough, h: {h_}"
            assert w_ >= self.sample_size[1], f"size not large enough, w: {w_}"
            assert w_ >= h_, "weight is smaller than height, skip"
            if (self.prev_ok_sample is None) or (self.require_update_prev):
                self.prev_ok_sample = copy.deepcopy((pixel_values, caption))
                self.require_update_prev = False
        except Exception as e:
            logger.warning(f"Fail to get sample of idx {idx}. The corrupted video will be replaced.")
            print("\tError msg: {}".format(e), flush=True)
            assert self.prev_ok_sample is not None
            pixel_values, caption = self.prev_ok_sample  # unless the first sample is already not ok
            self.require_update_prev = True

            if idx >= self.length:
                raise IndexError  # needed for checking the end of dataset iteration

        # pixel value: (f, h, w, 3) -> transforms -> (f 3 h' w')
        if self.transform_backend == "pt":
            import torch

            pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
            pixel_values = self.pixel_transforms(pixel_values)
            pixel_values = pixel_values.numpy()
        elif self.transform_backend == "al":
            # NOTE:it's to ensure augment all frames in a video in the same way.
            # ref: https://albumentations.ai/docs/examples/example_multi_target/

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

        if self.is_image:
            pixel_values = pixel_values[0]

        pixel_values = (pixel_values / 127.5 - 1.0).astype(np.float32)

        # randomly set caption to be empty
        if self.random_drop_text:
            if random.random() <= self.random_drop_text_ratio:
                caption = ""

        if self.tokenizer is not None:
            tokens, _ = self.tokenizer(caption)
            # print("D--: ", type(text_data))
            if isinstance(tokens, list):
                tokens = np.array(tokens, dtype=np.int64)
            if len(tokens.shape) == 2:  # in case, the tokenizer output [1, 77]
                tokens = tokens[0]
            text_data = tokens
        else:
            text_data = caption

        return pixel_values, caption, text_data


# TODO: parse in config dict
def create_dataloader(config, tokenizer=None, is_image=False, device_num=1, rank_id=0, n_samples=None):
    dataset = TextVideoDataset(
        config["csv_path"],
        config["video_folder"],
        sample_size=config["sample_size"],
        sample_stride=config["sample_stride"],
        sample_n_frames=config["sample_n_frames"],
        is_image=is_image,
        tokenizer=tokenizer,
        n_samples=n_samples,
        # disable_flip=config["disable_flip"],
        # video_column=config["video_column"],
        # caption_column=config["caption_column"],
        # random_drop_text=config["random_drop_text"],
        # random_drop_text_ratio=config["random_drop_text_ratio"],
    )
    print("Total number of samples: ", len(dataset))

    # Larger value leads to more memory consumption. Default: 16
    # prefetch_size = config.get("prefetch_size", 16)
    # ms.dataset.config.set_prefetch_size(prefetch_size)

    dataloader = ms.dataset.GeneratorDataset(
        source=dataset,
        column_names=[
            "video",
            "caption",
            "tokens",
        ],
        num_shards=device_num,
        shard_id=rank_id,
        python_multiprocessing=True,
        shuffle=config["shuffle"],
        num_parallel_workers=config["num_parallel_workers"],
        max_rowsize=config["max_rowsize"],  # video data require larger rowsize
    )

    dl = dataloader.batch(
        config["batch_size"],
        drop_remainder=True,
    )

    return dl


def check_sanity(x, save_fp="./tmp.gif"):
    # reverse normalization and visulaize the transformed video
    # (f, c, h, w) -> (f, h, w, c)
    if len(x.shape) == 3:
        x = np.expand_dims(x, axis=0)
    x = np.transpose(x, (0, 2, 3, 1))

    x = (x + 1.0) / 2.0  # -1,1 -> 0,1
    x = (x * 255).astype(np.uint8)

    imageio.mimsave(save_fp, x, duration=1 / 8.0, loop=1)


class SelectFrameMap:
    def __init__(self, is_image=False, sample_n_frames=16, sample_stride=4):
        self.is_image = is_image
        self.sample_n_frames = sample_n_frames
        self.sample_stride = sample_stride

    def __call__(self, video_latent):
        video_length = len(video_latent)
        if not self.is_image:
            clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
            start_idx = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        else:
            batch_index = [random.randint(0, video_length - 1)]

        video_emb_train = video_latent[batch_index]
        return video_emb_train
