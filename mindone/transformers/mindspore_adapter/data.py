from typing import Generic, Iterator, Optional, Sized, TypeVar, Union

import numpy as np

import mindspore as ms

T_co = TypeVar("T_co", covariant=True)


class Sampler(Generic[T_co]):
    r"""Base class for all Samplers.

    reference to torch.utils.data.Sampler

    Every Sampler subclass has to provide an :meth:`__iter__` method, providing a
    way to iterate over indices of dataset elements, and a :meth:`__len__` method
    that returns the length of the returned iterators.
    """

    def __init__(self, data_source: Optional[Sized]) -> None:
        pass

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError

    # NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
    #
    # Many times we have an abstract class representing a collection/iterable of
    # data, e.g., `.data.Sampler`, with its subclasses optionally
    # implementing a `__len__` method. In such cases, we must make sure to not
    # provide a default implementation, because both straightforward default
    # implementations have their issues:
    #
    #   + `return NotImplemented`:
    #     Calling `len(subclass_instance)` raises:
    #       TypeError: 'NotImplementedType' object cannot be interpreted as an integer
    #
    #   + `raise NotImplementedError()`:
    #     This prevents triggering some fallback behavior. E.g., the built-in
    #     `list(X)` tries to call `len(X)` first, and executes a different code
    #     path if the method is not found or `NotImplemented` is returned, while
    #     raising an `NotImplementedError` will propagate and and make the call
    #     fail where it could have use `__iter__` to complete the call.
    #
    # Thus, the only two sensible things to do are
    #
    #   + **not** provide a default `__len__`.
    #
    #   + raise a `TypeError` instead, which is what Python uses when users call
    #     a method that is not defined on an object.
    #     (@ssnl verifies that this works on at least Python 3.7.)


class SequentialSampler(Sampler[int]):
    r"""Samples elements sequentially, always in the same order.

    Args:
        data_source (Dataset): dataset to sample from
    """
    data_source: Sized

    def __init__(self, data_source: Sized) -> None:
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)


class RandomSampler(Sampler[int]):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """
    data_source: Sized
    replacement: bool

    def __init__(
        self, data_source: Sized, replacement: bool = False, num_samples: Optional[int] = None, generator=None
    ) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator

        if not isinstance(self.replacement, bool):
            raise TypeError("replacement should be a boolean value, but got " "replacement={}".format(self.replacement))

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer " "value, but got num_samples={}".format(self.num_samples)
            )

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)

        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from np.random.randint(low=0, high=n, size=(32,), dtype=np.int64).tolist()
            yield from np.random.randint(low=0, high=n, size=(self.num_samples % 32,), dtype=np.int64).tolist()
        else:
            for _ in range(self.num_samples // n):
                yield from np.random.permutation(n).tolist()
            yield from np.random.permutation(n).tolist()[: self.num_samples % n]

    def __len__(self) -> int:
        return self.num_samples


class Dataset(Generic[T_co]):
    r"""An abstract class representing a :class:`Dataset`.

    reference to torch.utils.data.Dataset

    All datasets that represent a map from keys to data samples should subclass
    it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses could also optionally overwrite
    :meth:`__len__`, which is expected to return the size of the dataset by many
    :class:`~mindspore_adapter.data.Sampler` implementations and the default options
    of :class:`~mindspore_adapter.data.DataLoader`.

    .. note::
      :class:`~mindspore_adapter.data.DataLoader` by default constructs a index
      sampler that yields integral indices.  To make it work with a map-style
      dataset with non-integral indices/keys, a custom sampler must be provided.
    """

    def __getitem__(self, index) -> T_co:
        raise NotImplementedError

    # No `def __len__(self)` default?
    # See NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
    # in mindone/transformers/mindspore_adapter/data.py:Sampler


class TensorDataset(Dataset):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Args:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, *tensors: Union[ms.Tensor, np.ndarray]) -> None:
        assert all(tensors[0].shape[0] == tensor.shape[0] for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(
            tensor[index] if isinstance(tensor, ms.Tensor) else ms.Tensor(tensor[index]) for tensor in self.tensors
        )

    def __len__(self):
        return self.tensors[0].shape[0]


class HF2MSDataset:
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, item):
        return self.dataset[int(item)]

    def __len__(self):
        return len(self.dataset)
