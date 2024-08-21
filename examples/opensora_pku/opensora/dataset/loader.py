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
import random
from collections import defaultdict

import numpy as np
import pandas as pd

import mindspore as ms
from mindspore.communication.management import get_local_rank, get_local_rank_size
from mindspore.dataset import GeneratorDataset

from .t2v_dataset import TextVideoDataset

_logger = logging.getLogger(__name__)


def create_dataloader(
    ds_config,
    batch_size,
    ds_name="text_video",
    column_names=["video"],
    num_parallel_workers=12,
    max_rowsize=64,
    shuffle=True,
    device_num=1,
    rank_id=0,
    drop_last=True,
    return_dataset=False,
    prefetch_size=None,
    enable_modelarts=False,
    collate_fn=None,
):
    if ds_name == "text_video":
        dataset = TextVideoDataset(**ds_config)
        column_names = ["video", "text", "mask"]
    else:
        raise NotImplementedError
    datalen = dataset.__len__

    if prefetch_size is not None:
        assert isinstance(prefetch_size, int)
        ms.dataset.config.set_prefetch_size(prefetch_size)

    dataloaders = {}

    if enable_modelarts:
        device_num = get_local_rank_size()
        rank_id = get_local_rank() % 8
    loader = build_dataloader(
        dataset,
        datalen,
        collate_fn,
        batch_size,
        device_num,
        rank_id=rank_id,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    dataloaders[ds_name] = loader

    metaloader = MetaLoader(dataloaders, datalen=batch_size, task_num=len(dataloaders.keys()))

    dl = GeneratorDataset(
        metaloader,
        column_names=column_names,
        shuffle=shuffle,
        num_parallel_workers=num_parallel_workers,
        max_rowsize=max_rowsize,
    )

    _logger.info("dataset size per shard: {}".format(dl.get_dataset_size()))

    if return_dataset:
        return dl, dataset
    return dl


def build_dataloader(dataset, datalens, collate_fn, batch_size, device_num, rank_id=0, shuffle=True, drop_last=True):
    sampler = BatchSampler(datalens, batch_size=batch_size, device_num=device_num, shuffle=shuffle)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        device_num=device_num,
        drop_last=drop_last,
        rank_id=rank_id,
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


class BatchSampler:
    """
    Batch Sampler
    """

    def __init__(self, lens, batch_size, device_num, shuffle):
        self._lens = lens
        self._batch_size = batch_size * device_num
        self.shuffle = shuffle

    def _create_ids(self):
        return list(range(self._lens))

    def __iter__(self):
        ids = self._create_ids()
        if self.shuffle:
            random.shuffle(ids)
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
        if self.collat_fn is not None:
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
