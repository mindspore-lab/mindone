import gc
import logging
import random
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
    num_parallel_workers=12,
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
):
    datalen = len(dataset)

    if prefetch_size is not None:
        assert isinstance(prefetch_size, int)
        ms.dataset.config.set_prefetch_size(prefetch_size)

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
        sampler=sampler,
        batch_sampler=batch_sampler,
    )
    dl = GeneratorDataset(
        loader,
        column_names=column_names,
        shuffle=shuffle,
        num_parallel_workers=num_parallel_workers,
        max_rowsize=max_rowsize,
    )
    dl.dataset_size = len(loader)

    _logger.info("dataset size per shard: {}".format(dl.get_dataset_size()))

    return dl


def build_dataloader(
    dataset,
    datalens,
    collate_fn,
    batch_size,
    device_num,
    rank_id=0,
    sampler=None,
    batch_sampler=None,
    shuffle=True,
    drop_last=True,
):
    if batch_sampler is None and sampler is None:
        # use batch sampler if not specified
        batch_sampler = BatchSampler(datalens, batch_size=batch_size, device_num=device_num, shuffle=shuffle)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        device_num=device_num,
        drop_last=drop_last,
        rank_id=rank_id,
    )
    return loader


class BatchSampler:
    """
    Batch Sampler that return batches of indices instead of single indices
    """

    def __init__(self, lens, batch_size, device_num, shuffle):
        self._lens = lens
        self._batch_size = batch_size * device_num
        self.shuffle = shuffle
        self.remainder = len(self) * self._batch_size != self._lens

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
        ids = list(range(0, self._lens, self._batch_size))
        return len(ids)


class DataLoader:
    """DataLoader"""

    def __init__(
        self,
        dataset,
        batch_size,
        sampler=None,
        batch_sampler=None,
        collate_fn=None,
        device_num=1,
        drop_last=True,
        rank_id=0,
    ):
        self.dataset = dataset
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        if self.sampler is not None and self.batch_sampler is not None:
            raise ValueError("Cannot specify both a sampler and a batch sampler simultaneously!")
        self.collate_fn = collate_fn
        self.device_num = device_num
        self.rank_id = rank_id
        self.drop_last = drop_last
        self.batch_size = batch_size
        self._batch_size = batch_size * device_num

    def __iter__(self):
        if self.batch_sampler is not None:
            # Use batch_sampler to get batches directly
            return iter(self.batch_sampler)
        else:
            # Use sampler to get indices and create batches
            self.sampler_iter = iter(self.sampler)
            return self

    def __next__(self):
        if self.batch_sampler is not None:
            # Get the next batch directly from the batch sampler
            batch_indices = next(self.batch_sampler)
        else:
            # Get the next indices from the sampler
            batch_indices = [next(self.sampler_iter) for _ in range(self._batch_size)]
        if len(batch_indices) != self._batch_size and self.drop_last:
            raise StopIteration()

        data = []
        per_batch = len(batch_indices) // self.device_num
        index = batch_indices[self.rank_id * per_batch : (self.rank_id + 1) * per_batch]
        for idx in index:
            data.append(self.dataset[idx])
        if self.collate_fn is not None:
            data = self.collate_fn(data)
        return data

    def __len__(self):
        if self.batch_sampler is not None:
            batch_sampler_len = len(self.batch_sampler)
            remainder = self.batch_sampler.remainder
            if remainder and self.drop_last:
                return batch_sampler_len - 1
            else:
                return batch_sampler_len
        else:
            remainder = len(self.sampler) % self._batch_size != 0
            if remainder and not self.drop_last:
                return len(self.sampler) // self._batch_size + 1
            else:
                return len(self.sampler) // self._batch_size


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
