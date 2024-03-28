from typing import List, Optional, Union

import mindspore as ms
from mindspore.communication import get_local_rank, get_local_rank_size

from .dataset import BaseDataset


def create_dataloader(
    dataset: BaseDataset,
    batch_size: int,
    transforms: Optional[Union[List[dict], dict]] = None,
    shuffle: bool = False,
    num_workers: int = 4,
    num_workers_dataset: int = 4,
    num_workers_batch: int = 2,
    drop_remainder: bool = True,
    python_multiprocessing: bool = True,
    prefetch_size: int = 16,
    max_rowsize: int = 64,
    device_num: int = 1,
    rank_id: int = 0,
    debug: bool = False,
    enable_modelarts: bool = False,
) -> ms.dataset.BatchDataset:
    """
    Builds and returns a DataLoader for the given dataset.

    Args:
        dataset: A dataset instance, must have `output_columns` member.
        batch_size: Number of samples per batch.
        transforms: Optional transformations to apply to the dataset. It can be a list of transform dictionaries or
                    a single transform dictionary. The dictionary must have the following structure:
                    {
                        "operations": [List of transform operations],               # Required
                        "input_columns": [List of columns to apply transforms to],  # Optional
                        "output_columns": [List of output columns]                  # Optional, only used if different from the `input columns`
                    }
        shuffle: Whether to randomly sample data. Default is False.
        num_workers: The number of workers used for data transformations. Default is 4.
        num_workers_dataset: The number of workers used for reading data from the dataset. Default is 4.
        num_workers_batch: The number of workers used for batch aggregation. Default is 2.
        drop_remainder: Whether to drop the remainder of the dataset if it doesn't divide evenly by `batch_size`.
                        Default is True.
        python_multiprocessing: Whether to use Python multiprocessing for data transformations. This option could be
                                beneficial if the Python operation is computational heavy. Default is True.
        prefetch_size: The number of samples to prefetch (per device). Default is 16.
        max_rowsize: Maximum size of row in MB that is used for shared memory allocation to copy data between processes.
                     This is only used if `python_multiprocessing` is set to `True`. Default is 64.
        device_num: The number of devices to distribute the dataset across. Default is 1.
        rank_id: The rank ID of the current device. Default is 0.
        debug: Whether to enable debug mode. Default is False.
        enable_modelarts: Whether to enable modelarts (OpenI) support. Default is False.

    Returns:
        ms.dataset.BatchDataset: The DataLoader for the given dataset.
    """
    if not hasattr(dataset, "output_columns"):
        raise AttributeError(f"{type(dataset).__name__} must have `output_columns` attribute.")

    ms.dataset.config.set_prefetch_size(prefetch_size)
    ms.dataset.config.set_enable_shared_mem(True)
    ms.dataset.config.set_debug_mode(debug)

    if enable_modelarts:
        device_num = get_local_rank_size()
        rank_id = get_local_rank() % 8

    dataloader = ms.dataset.GeneratorDataset(
        dataset,
        column_names=dataset.output_columns,
        num_parallel_workers=num_workers_dataset,
        num_shards=device_num,
        shard_id=rank_id,
        # file reading is not CPU bounded => use multithreading for reading images and labels
        python_multiprocessing=False,
        shuffle=shuffle,
    )

    if transforms is not None:
        if not isinstance(transforms, list):
            transforms = [transforms]

        for transform in transforms:
            dataloader = dataloader.map(
                **transform,
                python_multiprocessing=python_multiprocessing,
                num_parallel_workers=num_workers,
                max_rowsize=max_rowsize,
            )

    dataloader = dataloader.batch(batch_size, drop_remainder=drop_remainder, num_parallel_workers=num_workers_batch)

    return dataloader
