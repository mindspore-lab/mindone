# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import gc
import logging
import os
from collections import defaultdict
from random import randint

import albumentations
import imagesize
import numpy as np
import pandas as pd
from ldm.data.t2i_collate import data_column, t2i_collate
from PIL import Image

from mindspore.communication.management import get_local_rank, get_local_rank_size
from mindspore.dataset import GeneratorDataset

_logger = logging.getLogger(__name__)


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
    dataset = ImageDataset(
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
    metaloader = MetaLoader(dataloaders, datalen=batchlen, task_num=len(dataloaders.keys()))

    dataset = GeneratorDataset(metaloader, column_names=data_column, shuffle=True)

    _logger.info("dataset size per shard: {}".format(dataset.get_dataset_size()))
    return dataset


def build_dataloader_ft(dataset, datalens, collate_fn, batch_size, device_num, rank_id=0):
    sampler = BatchSampler(datalens, batch_size=batch_size, device_num=device_num)
    loader = DataLoader(
        dataset, batch_sampler=sampler, collate_fn=collate_fn, device_num=device_num, drop_last=True, rank_id=rank_id
    )
    return loader


def list_image_files_captions_recursively(data_path, enable_modelarts=False):
    anno_dir = data_path
    if enable_modelarts:
        anno_list = [os.path.join(data_path, "merged_imgp_text.csv")]
    else:
        anno_list = sorted(
            [os.path.join(anno_dir, f) for f in list(filter(lambda x: x.endswith(".csv"), os.listdir(anno_dir)))]
        )
    db_list = [pd.read_csv(f) for f in anno_list]
    all_images = []
    all_captions = []
    for db in db_list:
        all_images.extend(list(db["dir"]))
        all_captions.extend(list(db["text"]))
    assert len(all_images) == len(all_captions)
    all_images = [os.path.join(data_path, f) for f in all_images]
    _logger.info(f"Before filter, Total number of training samples: {len(all_images)}")
    return all_images, all_captions


def filter_small_image(all_images, all_captions, image_filter_size, replace):
    filted_images = []
    filted_captions = []
    filter_count = 0
    for image, caption in zip(all_images, all_captions):
        try:
            w, h = imagesize.get(image)
        except Exception:
            filter_count += 1
            _logger.info(f"image file open failed or not exist(replace with others), path: {image}")
            continue
        if min(w, h) < image_filter_size:
            _logger.info(f"The size of image {image}: {w}x{h} < `image_filter_size` and excluded from training.")
            filter_count += 1
            continue
        else:
            filted_images.append(image)
            filted_captions.append(caption)
    _logger.info(f"filter image count: {filter_count}")
    if replace:
        while filter_count > 0:
            filted_images.append(filted_images[filter_count])
            filted_captions.append(filted_captions[filter_count])
            filter_count -= 1
    _logger.info("complete image list, size: " + str(len(filted_images)))
    return filted_images, filted_captions


def check_data(all_iamges):
    print("===================\n Checking data...")
    bad_path_num = 0
    good_path_num = 0
    for file in all_iamges:
        if os.path.exists(file):
            good_path_num += 1
        else:
            bad_path_num += 1
            print(f"bad images path: {file}")
    print(
        f"There are {len(all_iamges)} pairs of data, including {good_path_num} pairs of good data and {bad_path_num} "
        f"pairs of bad data"
    )


class ImageDataset:
    def __init__(
        self,
        batch_size,
        image_paths,
        captions,
        tokenizer,
        image_size,
        image_filter_size,
        shuffle=True,
        random_crop=False,
        filter_small_size=False,
        drop_text_prob=0.0,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.image_filter_size = image_filter_size
        self.local_images = image_paths
        self.local_captions = captions
        self.shuffle = shuffle
        self.random_crop = random_crop
        self.filter_small_size = filter_small_size
        self.drop_text_prob = drop_text_prob

        self.rescaler = albumentations.SmallestMaxSize(max_size=self.image_size)
        if not self.random_crop:
            self.cropper = albumentations.CenterCrop(height=self.image_size, width=self.image_size)
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        else:
            self.cropper = albumentations.RandomCrop(height=self.image_size, width=self.image_size)
            self.preprocessor = albumentations.Compose(
                [self.rescaler, self.cropper, albumentations.HorizontalFlip(p=0.5)]
            )
            print("apply random crop and horizontal flip")

    @property
    def __len__(self):
        return len(self.local_images)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, idx):
        # images preprocess
        img_path = self.local_images[idx]
        image_input = self.preprocess_image(img_path)

        # caption preprocess
        if np.random.rand() < self.drop_text_prob:
            caption = ""
        else:
            caption = self.local_captions[idx]
        caption_input = self.tokenize(caption)
        return np.array(image_input, dtype=np.float32), np.array(caption_input, dtype=np.int64)

    def preprocess_image(self, image_path):
        try:
            image = Image.open(image_path)
            if not image.mode == "RGB":
                image = image.convert("RGB")
            image = np.array(image).astype(np.uint8)
        except Exception:
            print("image file open failed or not exist, path:", image_path, flush=True)
            image = np.zeros((512, 512, 3)).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        return image

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

        return result

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


class BatchSampler:
    """
    Batch Sampler
    """

    def __init__(self, lens, batch_size, device_num):
        self._lens = lens
        self._batch_size = batch_size * device_num

    def _create_ids(self):
        return list(range(self._lens))

    def __iter__(self):
        ids = self._create_ids()
        batches = [ids[i : i + self._batch_size] for i in range(0, len(ids), self._batch_size)]
        gc.collect()
        return iter(batches)

    def __len__(self):
        raise ValueError("NOT supported. " "This has some randomness across epochs")


class DataLoader:
    """DataLoader"""

    def __init__(self, dataset, batch_sampler, collate_fn, device_num=1, drop_last=True, rank_id=0):
        self.dataset = dataset
        self.batch_sampler = batch_sampler
        self.collat_fn = collate_fn
        self.device_num = device_num
        self.rank_id = rank_id
        self.drop_last = drop_last
        self.batch_size = len(next(iter(self.batch_sampler)))

    def __iter__(self):
        self.step_index = 0
        self.batch_indices = iter(self.batch_sampler)

        return self

    def __next__(self):
        try:
            indices = next(self.batch_indices)
            if len(indices) != self.batch_size and self.drop_last:
                return self.__next__()
        except StopIteration:
            self.batch_indices = iter(self.batch_sampler)
            indices = next(self.batch_indices)
        data = []
        per_batch = len(indices) // self.device_num
        index = indices[self.rank_id * per_batch : (self.rank_id + 1) * per_batch]
        for idx in index:
            data.append(self.dataset[idx])

        data = self.collat_fn(data)
        return data


class MetaLoader:
    """wraps multiple data loaders"""

    def __init__(self, loaders, datalen, task_num=1):
        assert isinstance(loaders, dict)
        self.task_num = task_num
        self.name2loader = {}
        self.name2iter = {}
        self.sampling_pools = []
        self.loaders = loaders
        self.datalen = datalen
        for n, l in loaders.items():
            if isinstance(l, tuple):
                l, r = l
            elif isinstance(l, DataLoader):
                r = 1
            else:
                raise ValueError()
            self.name2loader[n] = l
            self.name2iter[n] = iter(l)
            self.sampling_pools.extend([n] * r)

        self.task = self.sampling_pools[0]
        self.task_label = [0] * self.task_num
        self.step = 0
        self.step_cnt = 0
        self.task_index_list = np.random.permutation(self.task_num)
        self.all_ids = []

    def init_iter(self, task_name):
        self.name2iter[task_name] = iter(self.name2loader[task_name])

    def return_ids(self):
        return self.all_ids

    def get_batch(self, batch, task):
        """get_batch"""
        batch = defaultdict(lambda: None, batch)
        img_feat = batch.get("img_feat", None)
        txt_tokens = batch.get("txt_tokens", None)
        output = (img_feat, txt_tokens)

        return output

    def __getitem__(self, index):
        if self.step_cnt == self.task_num:
            self.task_index_list = np.random.permutation(self.task_num)
            self.step_cnt = 0
        task_index = self.task_index_list[self.step_cnt]
        local_task = self.sampling_pools[task_index]

        iter_ = self.name2iter[local_task]

        name = local_task
        try:
            batch = next(iter_)
        except StopIteration:
            self.init_iter(local_task)
            iter_ = self.name2iter[local_task]
            batch = next(iter_)

        task = name.split("_")[0]
        for key, val in batch.items():
            if isinstance(val, np.ndarray):
                if val.dtype == np.int64:
                    batch[key] = val.astype(np.int32)

        output = self.get_batch(batch, task)
        self.step_cnt += 1
        return output

    def __len__(self):
        return self.datalen


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
