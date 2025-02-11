from typing import List, Optional, Union

import mindspore as ms
from mindspore.communication import get_local_rank, get_local_rank_size

from ..utils.version_control import MS_VERSION
from .dataset import BaseDataset


def create_dataloader(
    dataset: BaseDataset,
    batch_size: int = 1,
    transforms: Optional[Union[List[dict], dict]] = None,
    batch_transforms: Optional[Union[List[dict], dict]] = None,
    project_columns: Optional[List[str]] = None,
    shuffle: bool = False,
    num_workers: int = 4,
    num_workers_dataset: int = 4,
    num_workers_batch: int = 2,
    drop_remainder: bool = True,
    python_multiprocessing: bool = True,
    prefetch_size: int = 16,
    max_rowsize: Optional[int] = None,
    device_num: int = 1,
    rank_id: int = 0,
    debug: bool = False,
    enable_modelarts: bool = False,
) -> ms.dataset.BatchDataset:
    """
    Builds and returns a DataLoader for the given dataset.

    Args:
        dataset: A dataset instance, must have `output_columns` member.
        batch_size: Number of samples per batch. Set to 0 to disable batching. Default is 1.
        transforms: Optional transformations to apply to the dataset. It can be a list of transform dictionaries or
                    a single transform dictionary. The dictionary must have the following structure:
                    {
                        "operations": [List of transform operations],               # Required
                        "input_columns": [List of columns to apply transforms to],  # Optional
                        "output_columns": [List of output columns]                  # Optional, only used if different from the `input columns`
                    }
        batch_transforms: Optional transformations to apply to the dataset. Identical to `transforms` but applied to
                          batches.
        project_columns: Optional list of output columns names from transformations.
                         These names can be used for column selection or sorting in a specific order.
        shuffle: Whether to randomly sample data. Default is False.
        num_workers: The number of workers used for data transformations. Default is 4.
        num_workers_dataset: The number of workers used for reading data from the dataset. Default is 4.
        num_workers_batch: The number of workers used for batch aggregation. Default is 2.
        drop_remainder: Whether to drop the remainder of the dataset if it doesn't divide evenly by `batch_size`.
                        Default is True.
        python_multiprocessing: Whether to use Python multiprocessing for data transformations. This option could be
                                beneficial if the Python operation is computational heavy. Default is True.
        prefetch_size: The number of samples to prefetch (per device). Default is 16.
        max_rowsize: Maximum size of row in MB for shared memory allocation to copy data among processes.
                     This is only used if `python_multiprocessing` is set to `True`.
                     Values:
                        - `None` (default):
                            - For MindSpore 2.3 and above: Uses -1 (dynamic allocation).
                            - For MindSpore 2.2 and below: Uses 64MB.
                        - `-1`: (MindSpore 2.3+ only) Allocates memory dynamically.
                        - Positive integer: Sets a specific maximum row size in MB.
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
    # ms.dataset.config.set_enable_shared_mem(True)   # shared memory is ON by default
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

    if max_rowsize is None:
        # MS 2.3 and above: allocate memory dynamically
        max_rowsize = -1 if MS_VERSION >= "2.3" else 64

    if transforms is not None:
        if isinstance(transforms, dict):
            transforms = [transforms]

        for transform in transforms:
            dataloader = dataloader.map(
                **transform,
                python_multiprocessing=python_multiprocessing,
                num_parallel_workers=num_workers,
                max_rowsize=max_rowsize,
            )

    if project_columns:
        dataloader = dataloader.project(project_columns)

    if getattr(dataset, "pad_info", None):
        if batch_size > 0:
            dataloader = dataloader.padded_batch(
                batch_size,
                drop_remainder=drop_remainder,
                num_parallel_workers=num_workers_batch,
                pad_info=dataset.pad_info,
            )
    else:
        if batch_size > 0:
            dataloader = dataloader.batch(
                batch_size, drop_remainder=drop_remainder, num_parallel_workers=num_workers_batch
            )
            if batch_transforms is not None:
                if isinstance(batch_transforms, dict):
                    batch_transforms = [batch_transforms]

                for batch_transform in batch_transforms:
                    dataloader = dataloader.map(
                        **batch_transform,
                        python_multiprocessing=python_multiprocessing,
                        num_parallel_workers=num_workers,
                        max_rowsize=max_rowsize,
                    )

    return dataloader
