import logging
import os
from collections import defaultdict

import numpy as np
from ldm.data.dataset import (
    ImageDataset,
    MetaLoader,
    build_dataloader_ft,
    filter_small_image,
    list_image_files_captions_recursively,
)
from PIL import Image
from toolz.sandbox import unzip
from transformers import CLIPImageProcessor

from mindspore.communication.management import get_local_rank, get_local_rank_size
from mindspore.dataset import GeneratorDataset

_logger = logging.getLogger(__name__)


data_column = ["img_feat", "txt_tokens", "clip_img_feat"]


def t2i_collate(inputs):
    """
    Return:
    :img_feat     (batch_size, height, weight, 3)
    :txt_tokens   (n, max_txt_len)
    """
    img_feat, txt_tokens, clip_img_feat = map(list, unzip(inputs))
    batch = {
        "img_feat": img_feat,
        "txt_tokens": txt_tokens,
        "clip_img_feat": clip_img_feat,
    }
    return batch


def load_data(
    data_path,
    batch_size,
    tokenizer,
    image_size=512,
    image_filter_size=256,
    random_crop=False,
    filter_small_size=True,
    device_num=1,
    rank_id=0,
    replace=True,
    sample_num=-1,
    enable_modelarts=False,
    drop_text_prob=0.0,
):
    if not os.path.exists(data_path):
        raise ValueError(f"Data directory {data_path} does not exist!")

    all_images, all_captions = list_image_files_captions_recursively(data_path, enable_modelarts)
    if filter_small_size:
        all_images, all_captions = filter_small_image(all_images, all_captions, image_filter_size, replace)

    _logger.debug(f"The first image path is {all_images[0]}, and the caption is {all_captions[0]}")
    _logger.info(f"Total number of training samples: {len(all_images)}")

    dataloaders = {}
    dataset = IPAdapterImageDataset(
        batch_size,
        all_images,
        all_captions,
        tokenizer,
        image_size,
        image_filter_size,
        random_crop=random_crop,
        filter_small_size=filter_small_size,
        drop_text_prob=drop_text_prob,
    )
    datalen = dataset.__len__
    if enable_modelarts:
        device_num = get_local_rank_size()
        rank_id = get_local_rank() % 8
    loader = build_dataloader_ft(dataset, datalen, t2i_collate, batch_size, device_num, rank_id=rank_id)
    dataloaders["ftT2I"] = loader
    if sample_num == -1:
        batchlen = datalen // (batch_size * device_num)
    else:
        batchlen = sample_num
    metaloader = IPAdapterMetaLoader(dataloaders, datalen=batchlen, task_num=len(dataloaders.keys()))

    dataset = GeneratorDataset(metaloader, column_names=data_column, shuffle=True)

    _logger.info("dataset size per shard: {}".format(dataset.get_dataset_size()))
    return dataset


class IPAdapterImageDataset(ImageDataset):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.clip_image_processor = CLIPImageProcessor()

    def __getitem__(self, idx):
        # images preprocess
        img_path = self.local_images[idx]
        image_input, clip_image_input = self.preprocess_image(img_path)

        # caption preprocess
        if np.random.rand() < self.drop_text_prob:
            caption = ""
        else:
            caption = self.local_captions[idx]

        caption_input = self.tokenize(caption)
        return (
            np.asarray(image_input, dtype=np.float32),
            np.asarray(caption_input, dtype=np.int32),
            np.asarray(clip_image_input, dtype=np.float32),
        )

    def preprocess_image(self, image_path):
        try:
            image = Image.open(image_path)
            if not image.mode == "RGB":
                image = image.convert("RGB")
            image = np.array(image).astype(np.uint8)
        except Exception:
            print("image file open failed or not exist, path:", image_path, flush=True)
            image = np.zeros((512, 512, 3)).astype(np.uint8)

        # clip image preprocess
        clip_image = self.clip_image_processor(image).pixel_values[0]

        image = self.preprocessor(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)

        return image, clip_image


class IPAdapterMetaLoader(MetaLoader):
    def get_batch(self, batch, task):
        """get_batch"""
        batch = defaultdict(lambda: None, batch)
        img_feat = batch.get("img_feat", None)
        txt_tokens = batch.get("txt_tokens", None)
        clip_img_feat = batch.get("clip_img_feat", None)
        output = (img_feat, txt_tokens, clip_img_feat)

        return output


def build_dataset(
    data_path,
    train_batch_size,
    tokenizer,
    image_size,
    image_filter_size,
    device_num,
    rank_id,
    random_crop,
    filter_small_size,
    replace,
    enable_modelarts,
    drop_text_prob=0.0,
):
    dataset = load_data(
        data_path=data_path,
        batch_size=train_batch_size,
        tokenizer=tokenizer,
        image_size=image_size,
        image_filter_size=image_filter_size,
        device_num=device_num,
        rank_id=rank_id,
        random_crop=random_crop,
        filter_small_size=filter_small_size,
        replace=replace,
        sample_num=-1,
        enable_modelarts=enable_modelarts,
        drop_text_prob=drop_text_prob,
    )
    _logger.info(f"Num batches for rank {rank_id}: {dataset.get_dataset_size()}")

    return dataset
