import logging
from collections import defaultdict

import numpy as np

import mindspore as ms
from mindspore.communication.management import get_local_rank, get_local_rank_size
from mindspore.dataset import GeneratorDataset

_logger = logging.getLogger(__name__)


def create_dataloader(
    dataset,
    batch_size,
    column_names=["video"],
    num_workers=12,
    max_rowsize=32,
    shuffle=True,
    device_num=1,
    rank_id=0,
    drop_last=True,
    prefetch_size=None,
    enable_modelarts=False,
    collate_fn=None,
    sampler=None,
    batch_sampler=None,
    dataset_iterator_no_copy=False,
):
    # do_copy=False enables the dataset iterator to not do copy when creating a tensor which takes less time.
    # Currently the default value of do_copy is True,
    # it is expected that the default value of do_copy will be changed to False in MindSpore 2.7.0.
    if dataset_iterator_no_copy:
        ms.dataset.config.set_iterator_mode(do_copy=False)
    if prefetch_size is not None:
        assert isinstance(prefetch_size, int)
        ms.dataset.config.set_prefetch_size(prefetch_size)

    if enable_modelarts:
        device_num = get_local_rank_size()
        rank_id = get_local_rank() % 8

    dl = GeneratorDataset(
        dataset,
        column_names=column_names,
        shuffle=shuffle,
        num_parallel_workers=num_workers,
        max_rowsize=max_rowsize,
        sampler=sampler,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        num_shards=None if device_num == 1 else device_num,
        shard_id=None if device_num == 1 else rank_id,
    ).batch(batch_size)

    _logger.info("dataset size per shard: {}".format(dl.get_dataset_size()))

    return dl


class CombinedLoader:
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
            elif isinstance(l, ms.dataset.GeneratorDataset):
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
