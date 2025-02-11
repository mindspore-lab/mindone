import copy
import csv
import logging
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

import albumentations
import cv2
import numpy as np
from decord import VideoReader
from tqdm import tqdm

import mindspore as ms

logger = logging.getLogger()


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
        csv_path,
        video_folder,
        text_emb_folder=None,
        vae_latent_folder=None,
        return_text_emb=False,
        return_vae_latent=False,
        vae_scale_factor=0.18215,
        sample_size=256,
        sample_stride=4,
        sample_n_frames=16,
        is_image=False,
        transform_backend="al",  # ms,  al
        tokenizer=None,
        video_column="video",
        caption_column="caption",
        random_drop_text=False,
        random_drop_text_ratio=0.1,
        disable_flip=True,
        filter_data: bool = False,
    ):
        """
        text_emb_folder: root dir of text embed saved in npz files. Expected to have the same file name and directory strcutre as videos. e.g.
            video_folder:
                folder1/
                    001.mp4
                    002.mp4
            text_emb_folder:
                folder1/
                    001.npz
                    002.npz
            video_folder and text_emb_folder can be the same folder for simplicity.

        tokenizer: a function, e.g. partial(get_text_tokens_and_mask, return_tensor=False), input text string, return text emb and mask
        """
        assert (
            not random_drop_text
        ), "Cfg training is already done in CaptionEmbedder, please adjust class_dropout_prob in STDiT args if needed."
        logger.info(f"loading annotations from {csv_path} ...")
        self.dataset = self._read_data(
            video_folder, csv_path, video_column, text_emb_folder, vae_latent_folder, filter_data
        )

        self.video_folder = video_folder
        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames
        self.is_image = is_image
        self.vae_scale_factor = vae_scale_factor

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

        self.return_text_emb = return_text_emb
        if return_text_emb:
            assert text_emb_folder is not None
        self.text_emb_folder = text_emb_folder
        self.return_vae_latent = return_vae_latent
        if return_vae_latent:
            assert vae_latent_folder is not None
        self.vae_latent_folder = vae_latent_folder

        # prepare replacement data
        max_attempts = 100
        self.prev_ok_sample = self.get_replace_data(max_attempts)
        self.require_update_prev = False

    @staticmethod
    def _read_data(
        data_dir: str,
        csv_path: str,
        video_column: str,
        text_emb_folder: Optional[str] = None,
        vae_latent_folder: Optional[str] = None,
        filter_data: bool = False,
    ) -> List[dict]:
        def _filter_data(sample_):
            if not os.path.isfile(sample_[video_column]):
                logger.warning(f"Video not found: {sample_[video_column]}")
                return None
            elif "text_emb" in sample_ and not os.path.isfile(sample_["text_emb"]):
                logger.warning(f"Text embedding not found: {sample_['text_emb']}")
                return None
            elif "vae_latent" in sample_ and not os.path.isfile(sample_["vae_latent"]):
                logger.warning(f"Text embedding not found: {sample_['vae_latent']}")
                return None
            return sample_

        with open(csv_path, "r") as csv_file:
            try:
                data = []
                for item in csv.DictReader(csv_file):
                    sample = {**item, video_column: os.path.join(data_dir, item[video_column])}
                    if text_emb_folder:
                        sample["text_emb"] = os.path.join(text_emb_folder, Path(item[video_column]).with_suffix(".npz"))
                    if vae_latent_folder:
                        sample["vae_latent"] = os.path.join(
                            vae_latent_folder, Path(item[video_column]).with_suffix(".npz")
                        )
                    data.append(sample)
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

    def get_replace_data(self, max_attempts=100):
        replace_data = None
        attempts = min(max_attempts, len(self))
        for idx in range(attempts):
            try:
                pixel_values, text, mask = self.get_batch(idx)
                replace_data = copy.deepcopy((pixel_values, text, mask))
                break
            except Exception as e:
                print("\tError msg: {}".format(e), flush=True)

        assert replace_data is not None, f"Fail to preload sample in {attempts} attempts."

        return replace_data

    def parse_text_emb(self, npz):
        if not os.path.exists(npz):
            raise ValueError(
                f"text embedding file {npz} not found. Please check the text_emb_folder and make sure the text embeddings are already generated"
            )
        td = np.load(npz)
        text_emb = td["text_emb"]
        mask = td["mask"]
        # tokens = td['tokens']

        return text_emb, mask

    def get_batch(self, idx):
        # get video raw pixels (batch of frame) and its caption
        video_dict = self.dataset[idx]
        video_path, caption = video_dict[self.video_column], video_dict[self.caption_column]

        if self.return_text_emb:
            text_emb, mask = self.parse_text_emb(self.dataset[idx]["text_emb"])

        if not self.return_vae_latent:
            # in case missing .mp4 in csv file
            if not video_path.endswith(".mp4") or video_path.endswith(".gif"):
                if video_path[-4] != ".":
                    video_path = video_path + ".mp4"
                else:
                    raise ValueError(f"video file format is not verified: {video_path}")

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
            # pixel_values = pixel_values / 255. # let's keep uint8 for fast compute
            del video_reader

            if self.return_text_emb:
                return pixel_values, text_emb, mask
            else:
                return pixel_values, caption, None

        else:
            vae_latent_data = np.load(self.dataset[idx]["vae_latent"])
            latent_mean = vae_latent_data["latent_mean"]
            video_length = len(latent_mean)
            if not self.is_image:
                clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
                start_idx = random.randint(0, video_length - clip_length)
                batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
            else:
                batch_index = [random.randint(0, video_length - 1)]
            latent_mean = latent_mean[batch_index]

            sample_from_distribution = "latent_std" in vae_latent_data.keys()
            if sample_from_distribution:
                latent_std = vae_latent_data["latent_std"]
                latent_std = latent_std[batch_index]
                vae_latent = latent_mean + latent_std * np.random.standard_normal(latent_mean.shape)
            else:
                vae_latent = latent_mean
            vae_latent = vae_latent * self.vae_scale_factor
            vae_latent = vae_latent.astype(np.float32)

            if self.return_text_emb:
                return vae_latent, text_emb, mask
            else:
                return vae_latent, caption, None

    def __len__(self):
        return len(self.dataset)

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

    def __getitem__(self, idx):
        """
        Returns:
            tuple (video, text_data)
                - video (np.float32): preprocessed video frames in shape (f, c, h, w) for vae encoding
                - text_data: np.float32 if return embedding, tokens shape (context_max_len,), otherwise np.int64
        """
        try:
            pixel_values, text, mask = self.get_batch(idx)
            if (self.prev_ok_sample is None) or (self.require_update_prev):
                self.prev_ok_sample = copy.deepcopy((pixel_values, text, mask))
                self.require_update_prev = False
        except Exception as e:
            logger.warning(f"Fail to get sample of idx {idx}. The corrupted video will be replaced.")
            print("\tError msg: {}".format(e), flush=True)
            assert self.prev_ok_sample is not None
            pixel_values, text, mask = self.prev_ok_sample  # unless the first sample is already not ok
            self.require_update_prev = True

            if idx >= len(self):
                raise IndexError  # needed for checking the end of dataset iteration
        if not self.return_vae_latent:
            # apply visual transform and normalization
            pixel_values = self.apply_transform(pixel_values)

            if self.is_image:
                pixel_values = pixel_values[0]

            pixel_values = np.divide(pixel_values, 127.5, dtype=np.float32)
            pixel_values = np.subtract(pixel_values, 1.0, dtype=np.float32)
        else:
            # pixel_values is the vae encoder's output sample * scale_factor
            pass

        # randomly set caption to be empty
        if self.random_drop_text:
            if random.random() <= self.random_drop_text_ratio:
                if self.return_text_emb:
                    # TODO: check the t5 embedding for for empty caption
                    text = np.zeros_like(text)
                    mask = np.zeros_like(mask)
                    assert len(mask.shape) == 1
                    mask[0] = 1
                else:
                    text = ""

        if self.return_text_emb:
            text_data = text.astype(np.float32)
        else:
            if self.tokenizer is not None:
                tokens, mask = self.tokenizer(text)
                if isinstance(tokens, list):
                    tokens = np.array(tokens, dtype=np.int64)
                if isinstance(tokens, ms.Tensor):
                    tokens = tokens.asnumpy()
                if isinstance(mask, ms.Tensor):
                    mask = mask.asnumpy()
                text_data = tokens
            else:
                raise ValueError("tokenizer must be provided to generate text mask if text embeddings are not cached.")

        return pixel_values, text_data, mask.astype(np.uint8)

    def traverse_single_video_frames(self, video_index):
        video_dict = self.dataset[video_index]
        video_path = video_dict[self.video_column]

        # video_name = os.path.basename(video_fn).split(".")[0]
        # read video
        # in case missing .mp4 in csv file
        if not video_path.endswith(".mp4") or video_path.endswith(".gif"):
            if video_path[-4] != ".":
                video_path = video_path + ".mp4"
            else:
                raise ValueError(f"video file format is not verified: {video_path}")

        video_reader = VideoReader(video_path)
        video_length = len(video_reader)

        # Sampling video frames
        clips_indices = []
        start_idx = 0
        while start_idx + self.sample_n_frames < video_length:
            clips_indices.append([start_idx, start_idx + self.sample_n_frames])
            start_idx += self.sample_n_frames
        if start_idx < video_length:
            clips_indices.append([start_idx, video_length])
        assert len(clips_indices) > 0 and clips_indices[-1][-1] == video_length, "incorrect sampled clips!"

        for clip_indices in clips_indices:
            i, j = clip_indices
            frame_indice = list(range(i, j, 1))
            select_video_frames = [
                f"{index}" for index in frame_indice
            ]  # return indexes as strings, for the purpose of saving frame-wise embedding cache
            if video_path.endswith(".gif"):
                pixel_values = video_reader[frame_indice]  # shape: (f, h, w, c)
            else:
                pixel_values = video_reader.get_batch(frame_indice).asnumpy()  # shape: (f, h, w, c)
            pixel_values = self.apply_transform(pixel_values)
            pixel_values = (pixel_values / 127.5 - 1.0).astype(np.float32)
            return_dict = {"video": pixel_values}
            yield video_path, select_video_frames, return_dict


def create_dataloader(
    ds_config,
    batch_size,
    ds_name="text_video",
    num_parallel_workers=12,
    max_rowsize=64,
    shuffle=True,
    device_num=1,
    rank_id=0,
    drop_remainder=True,
    return_dataset=False,
):
    if ds_name == "text_video":
        dataset = TextVideoDataset(**ds_config)
        column_names = ["video", "text", "mask"]
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
    return dl
