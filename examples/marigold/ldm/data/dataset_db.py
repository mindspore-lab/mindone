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
import os
from collections import defaultdict
from functools import partial
from random import choice, randint

import albumentations
import imagesize
import numpy as np
import pandas as pd
from ldm.data.t2i_collate import data_column_db, t2i_collate_db
from PIL import Image

from mindspore.dataset import GeneratorDataset


def load_data(
    train_data_path,
    reg_data_path,
    instance_prompt,
    class_prompt,
    batch_size,
    tokenizer,
    image_size=512,
    image_filter_size=256,
    train_data_repeats=1,
    device_num=1,
    random_crop=False,
    rank_id=0,
    sample_num=-1,
    with_prior_preservation=True,
):
    if not os.path.exists(train_data_path):
        raise ValueError("Training data path directory does not exist!")
    train_images = list_image_files(train_data_path)
    print(f"Total training images: {len(train_images)}")
    if with_prior_preservation:
        if not os.path.exists(reg_data_path):
            raise ValueError("Regularization data path directory does not exist!")
        reg_images = list_image_files(reg_data_path)
        print(f"Total regularization images: {len(reg_images)}")
    train_images = repeat_data(train_images, train_data_repeats)
    print(f"The training data is repeated {train_data_repeats} times, and the total number is {len(train_images)}")

    dataloaders = {}
    dataset = ImageDataset(
        batch_size,
        train_images,
        reg_images if with_prior_preservation else None,
        instance_prompt,
        class_prompt if with_prior_preservation else None,
        tokenizer,
        image_size,
        image_filter_size,
        random_crop=random_crop,
        with_prior_preservation=with_prior_preservation,
    )
    datalen = dataset.__len__
    collate_func = partial(t2i_collate_db, with_prior_preservation=with_prior_preservation)
    loader = build_dataloader_ft(dataset, datalen, collate_func, batch_size, device_num, rank_id=rank_id)
    dataloaders["ftT2I"] = loader
    if sample_num == -1:
        batchlen = datalen // (batch_size * device_num)
    else:
        batchlen = sample_num
    metaloader = MetaLoader(
        dataloaders, datalen=batchlen, task_num=len(dataloaders.keys()), with_prior_preservation=with_prior_preservation
    )
    fetch_columns = data_column_db(with_prior_preservation=with_prior_preservation)
    dataset = GeneratorDataset(metaloader, column_names=fetch_columns, shuffle=True)

    return dataset


def build_dataloader_ft(dataset, datalens, collate_fn, batch_size, device_num, rank_id=0):
    sampler = BatchSampler(datalens, batch_size=batch_size, device_num=device_num)
    loader = DataLoader(
        dataset, batch_sampler=sampler, collate_fn=collate_fn, device_num=device_num, drop_last=True, rank_id=rank_id
    )
    return loader


def list_image_files(data_path):
    all_images = []
    for file_name in os.listdir(data_path):
        if (not file_name.endswith(".csv")) and (not file_name.endswith(".json")):
            imges_path = os.path.join(data_path, file_name)
            all_images.append(imges_path)
    return all_images


def repeat_data(data_list, repeats):
    return data_list * repeats


def list_image_files_captions_recursively(data_path):
    anno_dir = data_path
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

    return all_images, all_captions


def filter_small_image(all_images, all_captions, image_filter_size):
    filted_images = []
    filted_captions = []
    for image, caption in zip(all_images, all_captions):
        try:
            w, h = imagesize.get(image)
        except ValueError:
            continue
        if min(w, h) < image_filter_size:
            continue
        else:
            filted_images.append(image)
            filted_captions.append(caption)
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
        train_images,
        reg_images,
        instance_prompt,
        class_prompt,
        tokenizer,
        image_size,
        image_filter_size,
        shuffle=True,
        random_crop=False,
        with_prior_preservation=True,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.image_filter_size = image_filter_size
        self.train_images = train_images
        self.reg_images = reg_images
        self.shuffle = shuffle
        self.random_crop = random_crop
        self.class_prompt = class_prompt
        self.instance_prompt = instance_prompt
        self.with_prior_preservation = with_prior_preservation

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
        return len(self.train_images)

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
        train_img_path = self.train_images[idx]
        train_image_input = self.preprocess_image(train_img_path)
        if self.with_prior_preservation:
            reg_image_path = choice(self.reg_images)
            reg_image_input = self.preprocess_image(reg_image_path)

        # caption preprocess
        train_caption = self.instance_prompt
        train_caption_input = self.tokenize(train_caption)
        train_image_input = np.array(train_image_input, dtype=np.float32)
        train_caption_input = np.array(train_caption_input, dtype=np.int32)
        if self.with_prior_preservation:
            reg_caption = self.class_prompt
            reg_caption_input = self.tokenize(reg_caption)
            reg_image_input = np.array(reg_image_input, dtype=np.float32)
            reg_caption_input = np.array(reg_caption_input, dtype=np.int32)
            return train_image_input, train_caption_input, reg_image_input, reg_caption_input
        else:
            return train_image_input, train_caption_input

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        return image

    def tokenize(self, text):
        SOT_TEXT = self.tokenizer.sot_text  # "[CLS]"
        EOT_TEXT = self.tokenizer.eot_text  # "[SEP]"
        CONTEXT_LEN = 77  # TODO: get from self.tokenizer.context_len

        sot_token = self.tokenizer.encoder[SOT_TEXT]
        eot_token = self.tokenizer.encoder[EOT_TEXT]
        tokens = [sot_token] + self.tokenizer.encode(text) + [eot_token]
        result = np.zeros([CONTEXT_LEN]) + eot_token
        if len(tokens) > CONTEXT_LEN:
            tokens = tokens[: CONTEXT_LEN - 1] + [eot_token]
        result[: len(tokens)] = tokens

        return result


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

    def __init__(self, loaders, datalen, task_num=1, with_prior_preservation=True):
        assert isinstance(loaders, dict)
        self.task_num = task_num
        self.name2loader = {}
        self.name2iter = {}
        self.sampling_pools = []
        self.loaders = loaders
        self.datalen = datalen
        self.with_prior_preservation = with_prior_preservation
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
        train_img_feat = batch.get("train_img_feat", None)
        train_txt_tokens = batch.get("train_txt_tokens", None)
        if self.with_prior_preservation:
            reg_img_feat = batch.get("reg_img_feat", None)
            reg_txt_tokens = batch.get("reg_txt_tokens", None)
            output = (train_img_feat, train_txt_tokens, reg_img_feat, reg_txt_tokens)
        else:
            output = (train_img_feat, train_txt_tokens)

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
