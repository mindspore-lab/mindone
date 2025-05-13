"""
Cache utils.
"""
import copy
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from transformers.configuration_utils import PretrainedConfig

import mindspore as ms
from mindspore import nn, ops


def init_static_cache(config: PretrainedConfig, max_batch_size: int, max_cache_len: int, dtype=None):
    max_cache_len = config.max_position_embeddings if max_cache_len is None else max_cache_len
    # Some model define a custom `head_dim` != config.hidden_size // config.num_attention_heads
    head_dim = config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads

    dtype = dtype if dtype is not None else ms.float32
    num_key_value_heads = (
        config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
    )

    key_value_cache: Tuple[Tuple[ms.Tensor, ms.Tensor]] = []
    cache_shape = (max_batch_size, num_key_value_heads, max_cache_len, head_dim)
    for _layer_index in range(config.num_hidden_layers):
        # Note: `mark_static_address` is used to tag the cache as an fixed data pointer, preventing cuda graph
        # breaks when updating the cache.
        new_layer_key_cache = ms.Tensor(np.zeros(cache_shape), dtype=dtype)
        new_layer_value_cache = ms.Tensor(np.zeros(cache_shape), dtype=dtype)
        key_value_cache += [(new_layer_key_cache, new_layer_value_cache)]
    key_value_cache = tuple(key_value_cache)

    return key_value_cache


# TODO backup code for a future implementation of the graph mode static cache
# _pad_ops = ops.operations.PadV3()
# _sub_ops = ops.operations.Sub()
# _concat_ops = ops.operations.Concat(axis=0)  # for setting up arg
# _cache_padding_dim_preorder = Tensor([0, 0, 0], ms.int32)
# _cache_padding_dim_subsequence = Tensor([0, 0, 0, 0, 0], ms.int32)


# def kv_padding_subsequence(cache_length, state_seq_length, key, value, cache_position, dtype):
#     _pad_zero = Tensor([0,], dtype)
#     pad_length = _sub_ops(cache_length, state_seq_length)[None].to(ms.int32)
#     pad_config = _concat_ops((_cache_padding_dim_preorder, pad_length, _cache_padding_dim_subsequence))
#     key_padded = _pad_ops(key, pad_config, _pad_zero)
#     value_padded = _pad_ops(value, pad_config, _pad_zero)
#     cache_position_padded = _pad_ops(
#         cache_position,
#         _concat_ops((Tensor([0,], ms.int32), pad_length)),
#         Tensor([0,], ms.int32)
#     )
#     return key_padded, value_padded, cache_position_padded


# Notes: Only return the updated value, do not modify the original `past_key_value` in-place !
def update(
    past_key_value: Tuple[ms.Tensor, ms.Tensor],
    key_states: ms.Tensor,
    value_states: ms.Tensor,
    cache_position: Optional[ms.Tensor] = None,
) -> Tuple[ms.Tensor, ms.Tensor]:
    """
    Notes: Only return the updated value, do not modifying the original `past_key_value` in-place !

    Get the cache with the new `key_states` and `value_states` for cur layer.

    Parameters:
        past_key_value (`Tuple[ms.Tensor, ms.Tensor]`):
            Past key/value states cache.
        key_states (`ms.Tensor`):
            The new key states to cache.
        value_states (`ms.Tensor`):
            The new value states to cache.
        cache_position (`ms.Tensor`, `optional`):
            Additional arguments for the cache subclass, needs the `cache_position` input
            to know how where to write in the cache.

    Return:
        A tuple containing the updated key and value states.
    """
    k_out, v_out = past_key_value[0], past_key_value[1]

    k_out = ops.select(
        (ops.arange(k_out.shape[2]) == cache_position)[None, None, :, None],
        key_states,
        k_out,
    )
    v_out = ops.select(
        (ops.arange(v_out.shape[2]) == cache_position)[None, None, :, None],
        value_states,
        v_out,
    )

    return k_out, v_out


def get_seq_length(past_key_values, layer_idx: Optional[int] = 0) -> int:
    """Returns the sequence length of the cached states that were seen by the model."""
    # Occupied cache == any slot in the 3rd dim (sequence length) holds a non-zero value. To save on compute, let's
    # limit the check to the first batch member and head dimension.
    # TODO: deprecate this function in favor of `cache_position`
    return (past_key_values[layer_idx][0][0, 0].any(axis=-1)).sum()


def get_max_length(past_key_values) -> Optional[int]:
    """Returns the maximum sequence length of the cached states."""
    return past_key_values[0][0].shape[2]


def reset(past_key_values):
    """Resets the cache values while preserving the objects"""
    for layer_idx in range(len(past_key_values)):
        # In-place ops prevent breaking the static address
        past_key_values[layer_idx][0] = ops.zeros_like(past_key_values[layer_idx][0])  # key
        past_key_values[layer_idx][1] = ops.zeros_like(past_key_values[layer_idx][1])  # value

    return past_key_values


class Cache(nn.Cell):
    """
    Base, abstract class for all caches. The actual data structure is specific to each subclass.
    """

    def update(
        self,
        key_states: ms.Tensor,
        value_states: ms.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ms.Tensor, ms.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`ms.Tensor`):
                The new key states to cache.
            value_states (`ms.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. These are specific to each subclass and allow new types of
                cache to be created.

        Return:
            A tuple containing the updated key and value states.
        """
        raise NotImplementedError("Make sure to implement `update` in a subclass.")

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        raise NotImplementedError("Make sure to implement `get_seq_length` in a subclass.")

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states, if there is any."""
        raise NotImplementedError("Make sure to implement `get_max_length` in a subclass.")

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        """Given the sequence length of the new inputs, returns the usable length of the cache."""
        # Cache without size limit -> all cache is usable
        # Cache with size limit -> if the length cache plus the length of the new inputs is larger the maximum cache
        #   length, we will need to evict part of the cache (and thus not all cache is usable)
        max_length = self.get_max_length()
        previous_seq_length = self.get_seq_length(layer_idx)
        if max_length is not None and previous_seq_length + new_seq_length > max_length:
            return max_length - new_seq_length
        return previous_seq_length

    def reorder_cache(self, beam_idx: ms.Tensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx] != []:
                self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx)
            if self.value_cache[layer_idx] != []:
                self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx)

    @property
    def seen_tokens(self):
        if hasattr(self, "_seen_tokens"):
            return self._seen_tokens
        else:
            return None


class StaticCache(Cache):
    """
    Static Cache class to be used with `static shape`.

    Parameters:
        config (`PretrainedConfig):
            The configuration file defining the shape-related attributes required to initialize the static cache.
        max_batch_size (`int`):
            The maximum batch size with which the model will be used.
        max_cache_len (`int`):
            The maximum sequence length with which the model will be used.
        dtype (*optional*, defaults to `ms.float32`):
            The default `dtype` to use when initializing the layer.
    """

    is_compileable = True

    def __init__(self, config: PretrainedConfig, max_batch_size: int, max_cache_len: int, dtype=None) -> None:
        super().__init__()
        self.max_batch_size = max_batch_size
        self.max_cache_len = config.max_position_embeddings if max_cache_len is None else max_cache_len
        # Some model define a custom `head_dim` != config.hidden_size // config.num_attention_heads
        self.head_dim = (
            config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
        )

        self.dtype = dtype if dtype is not None else ms.float32
        self.num_key_value_heads = (
            config.num_attention_heads if getattr(config, "num_key_value_heads", None) is None else config.num_key_value_heads
        )

        key_cache: List[ms.Parameter] = []
        value_cache: List[ms.Parameter] = []
        cache_shape = (max_batch_size, self.num_key_value_heads, self.max_cache_len, self.head_dim)
        for _layer_index in range(config.num_hidden_layers):
            # Note: `mark_static_address` is used to tag the cache as an fixed data pointer, preventing cuda graph
            # breaks when updating the cache.
            new_layer_key_cache = ms.Parameter(
                ms.Tensor(np.zeros(cache_shape), dtype=self.dtype),
                name=f"key_cache_{_layer_index}",
                requires_grad=False,
            )
            new_layer_value_cache = ms.Parameter(
                ms.Tensor(np.zeros(cache_shape), dtype=self.dtype),
                name=f"value_cache_{_layer_index}",
                requires_grad=False,
            )
            key_cache.append(new_layer_key_cache)
            value_cache.append(new_layer_value_cache)

        self.key_cache = ms.ParameterTuple(key_cache)
        self.value_cache = ms.ParameterTuple(value_cache)

    def update(
        self,
        key_states: ms.Tensor,
        value_states: ms.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ms.Tensor, ms.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`ms.Tensor`):
                The new key states to cache.
            value_states (`ms.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The `StaticCache` needs the `cache_position` input
                to know how where to write in the cache.

        Return:
            A tuple containing the updated key and value states.
        """
        cache_position = cache_kwargs.get("cache_position")
        k_out = self.key_cache[layer_idx]
        v_out = self.value_cache[layer_idx]

        if cache_position is None:
            k_out.copy_(key_states)
            v_out.copy_(value_states)
        else:
            k_out[:, :, cache_position] = key_states
            v_out[:, :, cache_position] = value_states

        return k_out, v_out

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states that were seen by the model."""
        # Occupied cache == any slot in the 3rd dim (sequence length) holds a non-zero value. To save on compute, let's
        # limit the check to the first batch member and head dimension.
        # TODO: deprecate this function in favor of `cache_position`
        return (self.key_cache[layer_idx][0, 0].any(axis=-1)).sum()

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states."""
        return self.max_cache_len

    def reset(self):
        """Resets the cache values while preserving the objects"""
        for layer_idx in range(len(self.key_cache)):
            # In-place ops prevent breaking the static address
            ops.assign(self.key_cache[layer_idx], ms.Tensor(0.0))
            ops.assign(self.value_cache[layer_idx], ms.Tensor(0.0))


class DynamicCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(self, num_hidden_layers: Optional[int] = None) -> None:
        # in hf transformers there is no `num_hidden_layers` but `_distributed_cache_data`
        # it was originally added for compatibility with `torch.distributed` (DDP). See #36121
        # in mindspore there is no DDP, so we keep `num_hidden_layers`
        super().__init__()
        if num_hidden_layers is None:
            self.key_cache: List[ms.Tensor] = []
            self.value_cache: List[ms.Tensor] = []
        else:
            self.key_cache: List[ms.Tensor] = [[] for _ in range(num_hidden_layers)]
            self.value_cache: List[ms.Tensor] = [[] for _ in range(num_hidden_layers)]
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen

    def __getitem__(self, layer_idx: int) -> List[Tuple[ms.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def update(
        self,
        key_states: ms.Tensor,
        value_states: ms.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ms.Tensor, ms.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`ms.Tensor`):
                The new key states to cache.
            value_states (`ms.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            # There may be skipped layers, fill them with empty lists
            for _ in range(len(self.key_cache), layer_idx):
                self.key_cache.append([])
                self.value_cache.append([])
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        # content on layer cache can be a tensor and checking not tensor causes errors
        # so we explicitly check for the empty list
        elif len(self.key_cache[layer_idx]) == 0:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = ops.cat([self.key_cache[layer_idx], key_states], axis=-2)
            self.value_cache[layer_idx] = ops.cat([self.value_cache[layer_idx], value_states], axis=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        if len(self.key_cache) <= layer_idx or (len(self.key_cache) > layer_idx and self.key_cache[layer_idx] == []):
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None

    def to_legacy_cache(self) -> Tuple[Tuple[ms.Tensor], Tuple[ms.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format. Used for
        backward compatibility."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[ms.Tensor]]] = None) -> "DynamicCache":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`. Used for
        backward compatibility."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache

    def crop(self, max_length: int):
        """Crop the past key values up to a new `max_length` in terms of tokens. `max_length` can also be
        negative to remove `max_length` tokens. This is used in assisted decoding and contrastive search."""
        # In case it is negative
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)

        if self.get_seq_length() <= max_length:
            return

        self._seen_tokens = max_length
        for idx in range(len(self.key_cache)):
            self.key_cache[idx] = self.key_cache[idx][..., :max_length, :]
            self.value_cache[idx] = self.value_cache[idx][..., :max_length, :]

    def batch_split(self, full_batch_size: int, split_size: int) -> List["DynamicCache"]:
        """Split the current instance into a list of `DynamicCache` by the batch size. This will be used by
        `_split_model_inputs()` in `generation.utils`"""
        out = []
        for i in range(0, full_batch_size, split_size):
            current_split = DynamicCache()
            current_split._seen_tokens = self._seen_tokens
            current_split.key_cache = [tensor[i : i + split_size] for tensor in self.key_cache]
            current_split.value_cache = [tensor[i : i + split_size] for tensor in self.value_cache]
            out.append(current_split)
        return out

    @classmethod
    def from_batch_splits(cls, splits: List["DynamicCache"]) -> "DynamicCache":
        """This is the opposite of the above `batch_split()` method. This will be used by `stack_model_outputs` in
        `generation.utils`"""
        cache = cls()
        for idx in range(len(splits[0])):
            layer_keys = ops.cat([current.key_cache[idx] for current in splits], dim=0)
            layer_values = ops.cat([current.value_cache[idx] for current in splits], dim=0)
            cache.update(layer_keys, layer_values, idx)
        return cache

    def batch_repeat_interleave(self, repeats: int):
        """Repeat the cache `repeats` times in the batch dimension. Used in contrastive search."""
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = ops.repeat_interleave(self.key_cache[layer_idx], repeats, dim=0)
            self.value_cache[layer_idx] = ops.repeat_interleave(self.value_cache[layer_idx], repeats, dim=0)

    def batch_select_indices(self, indices: ms.Tensor):
        """Only keep the `indices` in the batch dimension of the cache. Used in contrastive search."""
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][indices, ...]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][indices, ...]


class EncoderDecoderCache(Cache):
    def __init__(self):
        raise NotImplementedError


class SlidingWindowCache(Cache):
    def __init__(self):
        super(SlidingWindowCache, self).__init__()
        raise NotImplementedError
