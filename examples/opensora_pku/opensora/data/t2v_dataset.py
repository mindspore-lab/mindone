import copy
import csv
import html
import json
import logging
import os
import random
import re
import urllib.parse as ul
from pathlib import Path

import albumentations
import cv2
import ftfy
import numpy as np
from bs4 import BeautifulSoup
from decord import VideoReader

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
        data_file_path,
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
        use_image_num=False,
        use_img_from_vid=True,
        transform_backend="al",  # ms,  al
        tokenizer=None,
        video_column="video",
        caption_column="caption",
        random_drop_text=False,
        random_drop_text_ratio=0.1,
        disable_flip=True,
        token_max_length=120,
        use_text_preprocessing=True,
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
        logger.info(f"loading annotations from {data_file_path} ...")

        if data_file_path.endswith(".csv"):
            with open(data_file_path, "r") as csvfile:
                self.dataset = list(csv.DictReader(csvfile))
        elif data_file_path.endswith(".json"):
            with open(data_file_path, "r") as f:
                self.dataset = json.load(f)

        self.length = len(self.dataset)
        logger.info(f"Num data samples: {self.length}")

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

        self.use_image_num = use_image_num
        self.use_img_from_vid = use_img_from_vid
        if self.use_image_num != 0 and not self.use_img_from_vid:
            self.img_cap_list = self.get_img_cap_list()
        # prepare replacement data
        max_attempts = 100
        self.prev_ok_sample = self.get_replace_data(max_attempts)
        self.require_update_prev = False
        self.token_max_length = token_max_length
        self.use_text_preprocessing = use_text_preprocessing

    def get_img_cap_list(self):
        raise NotImplementedError

    def get_replace_data(self, max_attempts=100):
        replace_data = None
        attempts = min(max_attempts, self.length)
        for idx in range(attempts):
            # TODO: uncomment after training verified
            # try:
            pixel_values, text, mask = self.get_batch(idx)
            replace_data = copy.deepcopy((pixel_values, text, mask))
            #    break
            # except Exception as e:
            #     print("\tError msg: {}".format(e), flush=True)

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

    def video_read(self, video_path):
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
        return pixel_values, video_reader

    def vae_latent_read(self, vae_latent_path):
        vae_latent_data = np.load(vae_latent_path)
        latent_mean, latent_std = vae_latent_data["latent_mean"], vae_latent_data["latent_std"]
        video_length = len(latent_mean)
        if not self.is_image:
            clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
            start_idx = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        else:
            batch_index = [random.randint(0, video_length - 1)]
        latent_mean = latent_mean[batch_index]
        latent_std = latent_std[batch_index]
        vae_latent = latent_mean + latent_std * np.random.standard_normal(latent_mean.shape)
        vae_latent = vae_latent * self.vae_scale_factor
        vae_latent = vae_latent.astype(np.float32)
        return vae_latent  # (f, h, w, c)

    def get_batch(self, idx):
        # get video raw pixels (batch of frame) and its caption
        video_dict = self.dataset[idx]
        video_fn, caption = video_dict[self.video_column], video_dict[self.caption_column]
        if isinstance(caption, (tuple, list)):
            caption = caption[0]
        video_path = os.path.join(self.video_folder, video_fn)

        if self.return_text_emb:
            text_emb_path = Path(os.path.join(self.text_emb_folder, video_fn)).with_suffix(".npz")
            text_emb, mask = self.parse_text_emb(text_emb_path)
            text_data = text_emb
        else:
            mask = None
            text_data = caption

        if not self.return_vae_latent:
            video, video_reader = self.video_read(video_path)
            del video_reader
        else:
            vae_latent_path = Path(os.path.join(self.vae_latent_folder, video_fn)).with_suffix(".npz")
            video = self.vae_latent_read(vae_latent_path)
        return video, text_data, mask

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

    def __getitem__(self, idx):
        """
        Returns:
            tuple (video, text_data)
                - video: preprocessed video frames in shape (f, c, h, w) for vae encoding
                - text_data: if tokenizer provided, tokens shape (context_max_len,), otherwise text string
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

            if idx >= self.length:
                raise IndexError  # needed for checking the end of dataset iteration
        if not self.return_vae_latent:
            # apply visual transform and normalization
            pixel_values = self.apply_transform(pixel_values)

            if self.is_image:
                pixel_values = pixel_values[0]

            pixel_values = (pixel_values / 127.5 - 1.0).astype(np.float32)
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
                tokens, mask = self.get_token_ids_mask(text)
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

    def get_token_ids_mask(self, caption):
        text = self.text_preprocessing(caption)
        text_tokens_and_mask = self.tokenizer(
            text,
            max_length=self.token_max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors=None,
        )
        input_ids = text_tokens_and_mask["input_ids"].squeeze(0)
        mask = text_tokens_and_mask["attention_mask"].squeeze(0)
        return input_ids, mask

    def traverse_single_video_frames(self, video_index):
        video_dict = self.dataset[video_index]
        video_fn = video_dict[self.video_column]
        video_path = os.path.join(self.video_folder, video_fn)

        video_name = os.path.basename(video_fn).split(".")[0]
        # read video
        video_path = os.path.join(self.video_folder, video_fn)
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
            yield video_name, select_video_frames, return_dict

    def text_preprocessing(self, text):
        if self.use_text_preprocessing:
            # The exact text cleaning as was in the training stage:
            text = self.clean_caption(text)
            text = self.clean_caption(text)
            return text
        else:
            return text.lower().strip()

    @staticmethod
    def basic_clean(text):
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        return text.strip()

    def clean_caption(self, caption):
        caption = str(caption)
        caption = ul.unquote_plus(caption)
        caption = caption.strip().lower()
        caption = re.sub("<person>", "person", caption)
        # urls:
        caption = re.sub(
            r"\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
            "",
            caption,
        )  # regex for urls
        caption = re.sub(
            r"\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
            "",
            caption,
        )  # regex for urls
        # html:
        caption = BeautifulSoup(caption, features="html.parser").text

        # @<nickname>
        caption = re.sub(r"@[\w\d]+\b", "", caption)

        # 31C0—31EF CJK Strokes
        # 31F0—31FF Katakana Phonetic Extensions
        # 3200—32FF Enclosed CJK Letters and Months
        # 3300—33FF CJK Compatibility
        # 3400—4DBF CJK Unified Ideographs Extension A
        # 4DC0—4DFF Yijing Hexagram Symbols
        # 4E00—9FFF CJK Unified Ideographs
        caption = re.sub(r"[\u31c0-\u31ef]+", "", caption)
        caption = re.sub(r"[\u31f0-\u31ff]+", "", caption)
        caption = re.sub(r"[\u3200-\u32ff]+", "", caption)
        caption = re.sub(r"[\u3300-\u33ff]+", "", caption)
        caption = re.sub(r"[\u3400-\u4dbf]+", "", caption)
        caption = re.sub(r"[\u4dc0-\u4dff]+", "", caption)
        caption = re.sub(r"[\u4e00-\u9fff]+", "", caption)
        #######################################################

        # все виды тире / all types of dash --> "-"
        caption = re.sub(
            r"[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+",  # noqa
            "-",
            caption,
        )

        # кавычки к одному стандарту
        caption = re.sub(r"[`´«»“”¨]", '"', caption)
        caption = re.sub(r"[‘’]", "'", caption)

        # &quot;
        caption = re.sub(r"&quot;?", "", caption)
        # &amp
        caption = re.sub(r"&amp", "", caption)

        # ip adresses:
        caption = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ", caption)

        # article ids:
        caption = re.sub(r"\d:\d\d\s+$", "", caption)

        # \n
        caption = re.sub(r"\\n", " ", caption)

        # "#123"
        caption = re.sub(r"#\d{1,3}\b", "", caption)
        # "#12345.."
        caption = re.sub(r"#\d{5,}\b", "", caption)
        # "123456.."
        caption = re.sub(r"\b\d{6,}\b", "", caption)
        # filenames:
        caption = re.sub(r"[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)", "", caption)

        #
        caption = re.sub(r"[\"\']{2,}", r'"', caption)  # """AUSVERKAUFT"""
        caption = re.sub(r"[\.]{2,}", r" ", caption)  # """AUSVERKAUFT"""

        caption = re.sub(self.bad_punct_regex, r" ", caption)  # ***AUSVERKAUFT***, #AUSVERKAUFT
        caption = re.sub(r"\s+\.\s+", r" ", caption)  # " . "

        # this-is-my-cute-cat / this_is_my_cute_cat
        regex2 = re.compile(r"(?:\-|\_)")
        if len(re.findall(regex2, caption)) > 3:
            caption = re.sub(regex2, " ", caption)

        caption = self.basic_clean(caption)

        caption = re.sub(r"\b[a-zA-Z]{1,3}\d{3,15}\b", "", caption)  # jc6640
        caption = re.sub(r"\b[a-zA-Z]+\d+[a-zA-Z]+\b", "", caption)  # jc6640vc
        caption = re.sub(r"\b\d+[a-zA-Z]+\d+\b", "", caption)  # 6640vc231

        caption = re.sub(r"(worldwide\s+)?(free\s+)?shipping", "", caption)
        caption = re.sub(r"(free\s)?download(\sfree)?", "", caption)
        caption = re.sub(r"\bclick\b\s(?:for|on)\s\w+", "", caption)
        caption = re.sub(r"\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?", "", caption)
        caption = re.sub(r"\bpage\s+\d+\b", "", caption)

        caption = re.sub(r"\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b", r" ", caption)  # j2d1a2a...

        caption = re.sub(r"\b\d+\.?\d*[xх×]\d+\.?\d*\b", "", caption)

        caption = re.sub(r"\b\s+\:\s+", r": ", caption)
        caption = re.sub(r"(\D[,\./])\b", r"\1 ", caption)
        caption = re.sub(r"\s+", " ", caption)

        caption.strip()

        caption = re.sub(r"^[\"\']([\w\W]+)[\"\']$", r"\1", caption)
        caption = re.sub(r"^[\'\_,\-\:;]", r"", caption)
        caption = re.sub(r"[\'\_,\-\:\-\+]$", r"", caption)
        caption = re.sub(r"^\.\S+$", "", caption)

        return caption.strip()


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
