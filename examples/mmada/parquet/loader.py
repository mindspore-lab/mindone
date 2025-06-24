import itertools
import random
from typing import Any, Dict, Iterator, Tuple

import numpy as np

import mindspore as ms
from mindspore.dataset import GeneratorDataset


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

    return dl


class CombinedLoader:
    """
    A loader that combines multiple dataloaders into a single iterator.

    Args:
        iterables: Dictionary of dataloaders with string keys
        mode: How to combine the dataloaders. Options:
            - "max_size_cycle": Cycle through shorter dataloaders until the longest is exhausted
            - "min_size": Stop when the shortest dataloader is exhausted
            - "sequential": Iterate through dataloaders sequentially
            - "random": Randomly sample from dataloaders at each step
        output_numpy: if True, returns numpy arrays for each element, except for strings
    """

    def __init__(self, iterables: Dict[str, Iterator], mode: str = "max_size_cycle", output_numpy=True):
        self.iterables = iterables
        self.mode = mode
        self.dataloader_names = list(iterables.keys())

        # Calculate lengths for different modes
        self.lengths = {name: loader.dataset_size for name, loader in iterables.items()}
        self.max_length = max(self.lengths.values())
        self.min_length = min(self.lengths.values())
        self.output_numpy = output_numpy

        if mode == "max_size_cycle":
            self.total_length = self.max_length
        elif mode == "min_size":
            self.total_length = self.min_length
        elif mode == "sequential":
            self.total_length = sum(self.lengths.values())
        elif mode == "random":
            self.total_length = self.max_length
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def __len__(self) -> int:
        return self.total_length

    def _convert_to_numpy(self, input):
        if isinstance(input, str):
            return input
        else:
            output = np.array(input)
            assert isinstance(output, np.ndarray)
            return output

    def convert_to_numpy(self, input_dict):
        return {key: self._convert_to_numpy(value) for key, value in input_dict.items()}

    def __iter__(self) -> Iterator[Tuple[Dict[str, Any], int, int]]:
        if self.mode == "max_size_cycle":
            return self._iter_max_size_cycle()
        elif self.mode == "min_size":
            return self._iter_min_size()
        elif self.mode == "sequential":
            return self._iter_sequential()
        elif self.mode == "random":
            return self._iter_random()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _iter_max_size_cycle(self) -> Iterator[Tuple[Dict[str, Any], int, int]]:
        """Cycle through shorter dataloaders until the longest is exhausted."""
        iterators = {}
        for name, loader in self.iterables.items():
            iterators[name] = itertools.cycle(loader)

        for batch_idx in range(self.max_length):
            combined_batch = {}
            for dataloader_idx, name in enumerate(self.dataloader_names):
                try:
                    batch_data = next(iterators[name])
                    combined_batch[name] = batch_data if not self.output_numpy else self.convert_to_numpy(batch_data)
                except StopIteration:
                    # This shouldn't happen with cycle, but just in case
                    iterators[name] = itertools.cycle(self.iterables[name])
                    batch_data = next(iterators[name])
                    combined_batch[name] = batch_data if not self.output_numpy else self.convert_to_numpy(batch_data)

            yield combined_batch

    def _iter_min_size(self) -> Iterator[Tuple[Dict[str, Any], int, int]]:
        """Stop when the shortest dataloader is exhausted."""
        iterators = {name: iter(loader) for name, loader in self.iterables.items()}

        for batch_idx in range(self.min_length):
            combined_batch = {}
            for dataloader_idx, name in enumerate(self.dataloader_names):
                try:
                    batch_data = next(iterators[name])
                    combined_batch[name] = batch_data if not self.output_numpy else self.convert_to_numpy(batch_data)
                except StopIteration:
                    return

            yield combined_batch

    def _iter_sequential(self) -> Iterator[Tuple[Dict[str, Any], int, int]]:
        """Iterate through dataloaders sequentially."""
        batch_idx = 0

        for dataloader_idx, (name, loader) in enumerate(self.iterables.items()):
            for loader_batch_idx, batch_data in enumerate(loader):
                # Create a combined batch with only the current dataloader's data
                # Fill others with None or empty structures
                combined_batch = {}
                for loader_name in self.dataloader_names:
                    if loader_name == name:
                        combined_batch[loader_name] = batch_data
                    else:
                        # Create empty batch structure - this might need adjustment based on your data structure
                        combined_batch[loader_name] = self._create_empty_batch(loader_name)

                yield combined_batch
                batch_idx += 1

    def _iter_random(self) -> Iterator[Tuple[Dict[str, Any], int, int]]:
        """Randomly sample from dataloaders at each step."""
        iterators = {}
        exhausted = set()

        # Initialize iterators
        for name, loader in self.iterables.items():
            iterators[name] = iter(loader)

        for batch_idx in range(self.max_length):
            # Choose a random dataloader that's not exhausted
            available_loaders = [name for name in self.dataloader_names if name not in exhausted]

            if not available_loaders:
                # All exhausted, restart the shorter ones
                exhausted.clear()
                for name, loader in self.iterables.items():
                    if name in exhausted:
                        iterators[name] = iter(loader)
                available_loaders = self.dataloader_names

            chosen_loader = random.choice(available_loaders)
            dataloader_idx = self.dataloader_names.index(chosen_loader)

            combined_batch = {}
            for name in self.dataloader_names:
                if name == chosen_loader:
                    try:
                        batch_data = next(iterators[name])
                        combined_batch[name] = (
                            batch_data if not self.output_numpy else self.convert_to_numpy(batch_data)
                        )
                    except StopIteration:
                        exhausted.add(name)
                        # Try to get from another loader
                        remaining = [n for n in available_loaders if n != name and n not in exhausted]
                        if remaining:
                            chosen_loader = random.choice(remaining)
                            dataloader_idx = self.dataloader_names.index(chosen_loader)
                            batch_data = next(iterators[chosen_loader])
                            combined_batch[chosen_loader] = batch_data
                            combined_batch[name] = self._create_empty_batch(name)
                        else:
                            return
                else:
                    combined_batch[name] = self._create_empty_batch(name)

            yield combined_batch

    def _create_empty_batch(self, loader_name: str) -> Dict[str, Any]:
        """Create an empty batch structure for a given loader."""
        # This is a placeholder - you might need to adjust based on your actual data structure
        if "t2i" in loader_name:
            return {"images": [], "input_ids": []}
        elif "lm" in loader_name:
            return {"input_ids": []}
        elif "mmu" in loader_name:
            return {"images": [], "input_ids": [], "labels": []}
        else:
            return {}
