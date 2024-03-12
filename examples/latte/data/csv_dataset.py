import csv
import logging
import os
import random

import numpy as np
from decord import VideoReader
from PIL import Image, ImageSequence

import mindspore as ms

from .dataset import create_video_transforms

logger = logging.getLogger()


def read_gif(gif_path, mode="RGB"):
    with Image.open(gif_path) as fp:
        frames = np.array([np.array(frame.convert(mode)) for frame in ImageSequence.Iterator(fp)])
    return frames


class CSVDataset:
    def __init__(
        self,
        csv_path,
        video_folder,
        sample_size=256,
        sample_stride=4,
        sample_n_frames=16,
        transform_backend="al",  # ms, pt, al
        tokenizer=None,
        video_column=None,
        caption_column=None,
        class_column=None,
        use_safer_augment=False,
        image_video_joint=False,
        use_image_num=None,
    ):
        logger.info(f"loading annotations from {csv_path} ...")
        with open(csv_path, "r") as csvfile:
            self.dataset = list(csv.DictReader(csvfile))
        self.length = len(self.dataset)
        logger.info(f"Num data samples: {self.length}")

        self.video_folder = video_folder
        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)

        # it should match the transformation used in SD/VAE pretraining, especially for normalization
        self.pixel_transforms = create_video_transforms(
            sample_size[0],
            sample_size[1],
            sample_n_frames,
            interpolation="bicubic",
            backend=transform_backend,
            use_safer_augment=use_safer_augment,
        )
        self.transform_backend = transform_backend
        self.tokenizer = tokenizer
        self.video_column = video_column
        assert self.video_column is not None, "The input csv file must specifiy the video column"

        self.caption_column = caption_column
        self.class_column = class_column
        if self.caption_column is not None and self.tokenizer is None:
            logger.warning(
                f"The caption column is provided as {self.caption_column}, but tokenizer is None",
                "The text tokens will be dummy placeholders!",
            )
        self.image_video_joint = image_video_joint
        self.use_image_num = use_image_num
        if image_video_joint:
            # image video joint training not supported here because the total number of frames is unknown
            raise NotImplementedError
        self.image_transforms = None

    def get_batch(self, idx):
        video_dict = self.dataset[idx]
        video_fn = video_dict[self.video_column]
        # load caption if needed, otherwise replace it with a dummy value
        if self.caption_column is not None:
            caption = video_dict[self.caption_column]
        else:
            caption = ""
        # load class labels if needed, otherwise replace it with a dummy value
        if self.class_column is not None:
            class_label = int(video_dict[self.class_column])
        else:
            class_label = 0  # a dummy class label as a placeholder

        # pixel transformation
        video_path = os.path.join(self.video_folder, video_fn)
        if video_path.endswith(".gif"):
            video_reader = read_gif(video_path, mode="RGB")
        else:
            video_reader = VideoReader(video_path)
        # randomly sample video frames
        video_length = len(video_reader)
        clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
        start_idx = random.randint(0, video_length - clip_length)
        batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)

        if video_path.endswith(".gif"):
            pixel_values = video_reader[batch_index]  # shape: (f, h, w, c)
        else:
            pixel_values = video_reader.get_batch(batch_index).asnumpy()  # shape: (f, h, w, c)
        del video_reader

        return pixel_values, class_label, caption

    def __len__(self):
        return self.length

    def apply_transform(self, pixel_values, video_transform=True):
        if video_transform:
            transform_func = self.pixel_transforms
        else:
            assert self.image_video_joint, "image transform requires to set image-video-joint training on"
            transform_func = self.image_transforms
        if self.transform_backend == "pt":
            import torch

            pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
            pixel_values = transform_func(pixel_values)
            pixel_values = pixel_values.numpy()
        elif self.transform_backend == "al":
            # NOTE:it's to ensure augment all frames in a video in the same way.
            # ref: https://albumentations.ai/docs/examples/example_multi_target/

            inputs = {"image": pixel_values[0]}
            num_frames = len(pixel_values)
            for i in range(num_frames - 1):
                inputs[f"image{i}"] = pixel_values[i + 1]

            output = transform_func(**inputs)

            pixel_values = np.stack(list(output.values()), axis=0)
            # (f h w c) -> (f c h w)
            pixel_values = np.transpose(pixel_values, (0, 3, 1, 2))
        else:
            raise NotImplementedError
        return pixel_values

    def __getitem__(self, idx):
        """
        Returns:
            tuple (video, text_data)
                - video: preprocessed video frames in shape (f, c, h, w)
                - text_data: if tokenizer provided, tokens shape (context_max_len,), otherwise text string
        """
        pixel_values, class_label, caption = self.get_batch(idx)
        pixel_values = self.apply_transform(pixel_values, video_transform=True)
        pixel_values = (pixel_values / 127.5 - 1.0).astype(np.float32)

        if self.tokenizer is not None:
            tokens = self.tokenize(caption)
            # print("D--: ", type(text_data))
            if isinstance(tokens, list):
                tokens = np.array(tokens, dtype=np.int64)
            if len(tokens.shape) == 2:  # in case, the tokenizer output [1, 77]
                tokens = tokens[0]
            text_data = tokens
        else:
            text_data = np.array([49407], dtype=np.int64)  # dummy token ids as a placeholder. Do not return a string.
        return pixel_values, class_label, text_data

    def tokenize(self, text):
        # a hack to determine if use transformers.CLIPTokenizer
        # should handle it better
        if type(self.tokenizer).__name__ == "CLIPTokenizer":
            return self._clip_tokenize(text)

        SOT_TEXT = self.tokenizer.sot_text  # "[CLS]"
        EOT_TEXT = self.tokenizer.eot_text  # "[SEP]"
        CONTEXT_LEN = self.tokenizer.context_length

        sot_token = self.tokenizer.encoder[SOT_TEXT]
        eot_token = self.tokenizer.encoder[EOT_TEXT]
        tokens = [sot_token] + self.tokenizer.encode(text) + [eot_token]
        result = np.zeros([CONTEXT_LEN]) + eot_token
        if len(tokens) > CONTEXT_LEN:
            tokens = tokens[: CONTEXT_LEN - 1] + [eot_token]
        result[: len(tokens)] = tokens

        return result.astype(np.int64)

    def _clip_tokenize(self, texts):
        batch_encoding = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.tokenizer.context_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
        )
        tokens = np.array(batch_encoding["input_ids"], dtype=np.int32)
        return tokens


# TODO: parse in config dict
def create_dataloader(
    config,
    tokenizer=None,
    device_num=1,
    rank_id=0,
):
    dataset = CSVDataset(
        config["csv_path"],
        config["data_folder"],
        sample_size=config.get("sample_size", 256),
        sample_stride=config.get("sample_stride", 4),
        sample_n_frames=config.get("sample_n_frames", 16),
        video_column=config["video_column"],
        caption_column=config["caption_column"],
        class_column=config["class_column"],
        tokenizer=tokenizer,
    )

    dataloader = ms.dataset.GeneratorDataset(
        source=dataset,
        column_names=[
            "video",
            "label",
            "caption",
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
