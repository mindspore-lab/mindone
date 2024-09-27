import abc
from typing import List, Optional, Tuple

import mindspore.ops as ops
from mindspore import Tensor


class Cache(abc.ABC):
    @abc.abstractmethod
    def __getitem__(self, layer_idx: int) -> List[Tuple[Tensor]]:
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, key_states: Tensor, value_states: Tensor, layer_idx: int) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self, key_states: Tensor, value_states: Tensor, layer_idx: int) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    @property
    def seen_tokens(self) -> Optional[int]:
        return getattr(self, "_seen_tokens", None)


class DynamicCache(Cache):
    def __init__(self) -> None:
        self.key_cache: List[Tensor] = []
        self.value_cache: List[Tensor] = []
        self._seen_tokens = 0

    def __getitem__(self, layer_idx: int) -> List[Tuple[Tensor]]:
        if layer_idx < len(self.key_cache):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def update(self, key_states: Tensor, value_states: Tensor, layer_idx: int) -> Tuple[Tensor, Tensor]:
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if len(self.key_cache) == layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        elif len(self.key_cache) > layer_idx:
            self.key_cache[layer_idx] = ops.concat([self.key_cache[layer_idx], key_states], axis=-2)
            self.value_cache[layer_idx] = ops.concat([self.value_cache[layer_idx], value_states], axis=-2)
        else:
            raise KeyError(
                f"Layer index {layer_idx} is larger than the current key_cache length {len(self.key_cache)}."
            )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def reset(self) -> None:
        self.key_cache = []
        self.value_cache = []
        self._seen_tokens = 0
