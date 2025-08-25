"""
Adapted from https://github.com/huggingface/transformers/tree/main/src/transformers/cache_utils.py.

Cache utils.
"""
import copy
import functools
import inspect
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

import mindspore as ms
from mindspore import mint, ops

logger = logging.get_logger(__name__)


def init_static_cache(config: PretrainedConfig, max_batch_size: int, max_cache_len: int, dtype=None):
    # Hack implementation for multimodal models. Only the text part is used.
    if hasattr(config, "text_config"):
        config = config.text_config
        logger.info("Using text_config for static cache")

    max_cache_len = config.max_position_embeddings if max_cache_len is None else max_cache_len
    # Some model define a custom `head_dim` != config.hidden_size // config.num_attention_heads
    head_dim = config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads

    dtype = dtype if dtype is not None else ms.float32
    if hasattr(config, "num_key_value_heads"):
        num_key_value_heads = config.num_key_value_heads
    else:
        num_key_value_heads = config.num_attention_heads

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
    dynamic: bool = False,
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

    if dynamic:
        if len(k_out) == 0:  # first time, prefill the cache
            return key_states, value_states
        k_out = ops.cat((k_out, key_states), axis=-2)
        v_out = ops.cat((v_out, value_states), axis=-2)
        return k_out, v_out

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


def get_seq_length(past_key_values, layer_idx: Optional[int] = 0, dynamic=False) -> int:
    """Returns the sequence length of the cached states that were seen by the model."""
    # Occupied cache == any slot in the 3rd dim (sequence length) holds a non-zero value. To save on compute, let's
    # limit the check to the first batch member and head dimension.
    # TODO: deprecate this function in favor of `cache_position`
    if dynamic:
        if past_key_values is None:
            return 0
        return past_key_values[layer_idx][0].shape[-2]
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


class CacheLayerMixin(ABC):
    """Base, abstract class for a single layer's cache."""

    is_compileable = False

    def __init__(self):
        self.keys, self.values = None, None

    @abstractmethod
    def update(
        self,
        key_states: ms.Tensor,
        value_states: ms.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[ms.Tensor, ms.Tensor]:
        ...

    @abstractmethod
    def get_seq_length(self, cache_position=None) -> int:
        ...

    @abstractmethod
    def get_max_cache_shape(self) -> int:
        ...

    @abstractmethod
    def get_mask_sizes(self, cache_position: ms.Tensor) -> tuple[int, int]:
        ...

    def reset(self) -> None:
        """Resets the cache values while preserving the objects"""
        self.keys.zero_()
        self.values.zero_()

    def reorder_cache(self, beam_idx: ms.Tensor) -> tuple[ms.Tensor, ms.Tensor]:
        """Reorders this layer's cache for beam search."""
        if self.keys.numel():
            self.keys = self.keys.index_select(0, beam_idx)
        if self.values.numel():
            self.values = self.values.index_select(0, beam_idx)


class DynamicLayer(CacheLayerMixin):
    """
    A cache layer that grows dynamically as more tokens are generated. This is the default for generative models.
    It stores the Key and Value states as tensors with shape `[batch_size, num_heads, seq_len, head_dim]`.

    See `CacheLayerMixin` for details on common methods that are implemented by all cache layers.
    """

    is_sliding = False

    def update(
        self,
        key_states: ms.Tensor,
        value_states: ms.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[ms.Tensor, ms.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states`.

        Parameters:
            key_states (`ms.Tensor`):
                The new key states to cache.
            value_states (`ms.Tensor`):
                The new value states to cache.
            cache_kwargs (`dict[str, Any]`, *optional*):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicLayer`.

        Return:
            A tuple containing the updated key and value states.
        """
        if self.keys is None:
            self.keys = key_states
            self.values = value_states
        else:
            self.keys = mint.cat([self.keys, key_states], dim=-2)
            self.values = mint.cat([self.values, value_states], dim=-2)
        return self.keys, self.values

    def get_seq_length(self, cache_position=None) -> int:
        """Returns the sequence length of the cached states."""
        if self.keys is None or self.keys.numel() == 0:
            return 0
        return self.keys.shape[-2]

    def get_max_cache_shape(self) -> int:
        """Returns the maximum sequence length of the cache object. DynamicLayer does not have a maximum length."""
        return -1

    def reorder_cache(self, beam_idx: ms.Tensor) -> None:
        """Reorders the cache for beam search, given the selected beam indices."""
        if self.keys is not None and self.keys.numel():
            self.keys = self.keys.index_select(0, beam_idx)
            self.values = self.values.index_select(0, beam_idx)

    def crop(self, max_length: int) -> None:
        """
        Crop the past key values up to a new `max_length` in terms of tokens. `max_length` can also be
        negative to remove `max_length` tokens.
        """
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)

        if self.get_seq_length() <= max_length:
            return

        if self.keys is not None and self.keys.numel():
            self.keys = self.keys[..., :max_length, :]
            self.values = self.values[..., :max_length, :]

    def batch_repeat_interleave(self, repeats: int) -> None:
        """Repeat the cache `repeats` times in the batch dimension."""
        if self.keys is not None and self.keys.numel():
            self.keys = self.keys.repeat_interleave(repeats, dim=0)
            self.values = self.values.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: ms.Tensor) -> None:
        """Only keep the `indices` in the batch dimension of the cache."""
        if self.keys is not None and self.keys.numel():
            self.keys = self.keys[indices, ...]
            self.values = self.values[indices, ...]

    def get_mask_sizes(self, cache_position: ms.Tensor) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the mask"""
        kv_offset = 0
        query_length = cache_position.shape[0]
        past_seen_tokens = self.get_seq_length()
        kv_length = query_length + past_seen_tokens
        return kv_length, kv_offset

    @classmethod
    def from_tensors(cls, keys: ms.Tensor, values: ms.Tensor) -> "DynamicLayer":
        """
        Build a `DynamicLayer` instance from pre-existing key/value tensors.

        Args:
            keys (`ms.Tensor`):
                Key cache tensor of shape ``[batch_size, num_heads, seq_len, head_dim]``.
            values (`ms.Tensor`):
                Value cache tensor of shape ``[batch_size, num_heads, seq_len, head_dim]``.

        Returns:
            `DynamicLayer`: The newly constructed layer whose internal cache directly references
            the supplied tensors.
        """
        layer = cls()
        layer.keys = keys
        layer.values = values
        return layer


class StaticLayer(CacheLayerMixin):
    """
    A static cache layer that stores the Key and Value states as static tensors with shape `[batch_size, num_heads, seq_len, head_dim]`.
    It allocates its full backing tensors up-front and mutates them in-place. Built for `mindspore.jit` support.

    See `CacheLayerMixin` for details on common methods that are implemented by all cache layers.
    """

    is_compileable = True
    is_sliding = False

    def __init__(
        self,
        max_cache_len: int,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        dtype: ms.Type = ms.float32,
        sliding_window: Optional[int] = None,
    ):
        """
        Args:
            max_cache_len (`int`):
                Maximum number of tokens that can be stored, used for tensor preallocation.
            batch_size (`int`):
                Maximum batch size the cache is pre-allocated for.
            num_heads (`int`):
                Number of attention heads.
            head_dim (`int`):
                Per-head hidden dimension.
            dtype (`ms.Type`, defaults to `ms.float32`):
                Data type of the cache tensors.

        Notes:
            Static layers allocate their full backing tensors up-front and mutate them
            in-place. See the documentation of `Cache` for shared helper methods that
            operate uniformly across all layer types.
        """
        self.max_cache_len = max_cache_len
        self.max_batch_size = batch_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype

        self.keys = mint.zeros(
            (batch_size, num_heads, self.max_cache_len, head_dim),
            dtype=dtype,
        )
        self.values = mint.zeros(
            (batch_size, num_heads, self.max_cache_len, head_dim),
            dtype=dtype,
        )
        # Note: `mark_static_address` is used to tag the cache as a fixed data pointer,
        # preventing compiled graph breaks when updating the cache.
        # fixme there is no implementation for torch._dynamo.mark_static_address

    def get_max_cache_shape(self) -> int:
        """Return the maximum cache shape of the cache"""
        return self.max_cache_len

    def update(
        self,
        key_states: ms.Tensor,
        value_states: ms.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[ms.Tensor, ms.Tensor]:
        """
        Update the static cache tensors in place.

        Args:
            key_states (`ms.Tensor`): The new key states to cache.
            value_states (`ms.Tensor`): The new value states to cache.
            cache_kwargs (`dict[str, Any]`, *optional*): Additional arguments for the cache.

        Returns:
            tuple[`ms.Tensor`, `ms.Tensor`]: The updated key and value states.
        """
        cache_position = cache_kwargs.get("cache_position") if cache_kwargs else None
        key_states = key_states.to(self.keys.dtype)
        value_states = value_states.to(self.values.dtype)

        if cache_position is None:
            # Prefill phase where seq_len potentially equals max_cache_len. Directly copy.
            self.keys.copy_(key_states)
            self.values.copy_(value_states)
        else:
            # Generation phase. Update specific positions.
            # Use index_copy_ for in-place update (compile-friendly).
            try:
                self.keys.index_copy_(2, cache_position, key_states)
                self.values.index_copy_(2, cache_position, value_states)
            except NotImplementedError:
                # Fallback for devices like MPS where index_copy_ might not be supported.
                self.keys[:, :, cache_position] = key_states
                self.values[:, :, cache_position] = value_states
        return self.keys, self.values

    def get_seq_length(self, cache_position=None) -> int:
        """Returns the sequence length of the cached states."""
        if cache_position is not None:
            return int(cache_position[-1] + 1)
        # Occupied cache == any slot in the 3rd dim (sequence length) holds a non-zero value. To save on compute, let's
        # limit the check to the first batch member and head dimension.
        seq_length = (self.keys[0, 0].any(dim=-1)).sum() if self.keys is not None else 0
        return seq_length

    def reorder_cache(self, beam_idx: ms.Tensor) -> None:
        """Reorders the cache for beam search, given the selected beam indices."""
        self.keys = self.keys.index_select(0, beam_idx)
        self.values = self.values.index_select(0, beam_idx)

    def get_mask_sizes(self, cache_position: ms.Tensor) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the attention mask"""
        kv_offset = 0
        kv_length = self.max_cache_len
        return kv_length, kv_offset


class SlidingWindowLayer(StaticLayer):
    """
    A static cache layer that implements sliding window attention caching.

    See `CacheLayerMixin` for details on common methods that are implemented by all cache layers.
    """

    is_sliding = True

    def __init__(self, sliding_window, *args, **kwargs):
        """
        Args:
            sliding_window (`int`):
                Effective window size: number of tokens that are kept on each update call.
        """
        max_cache_len = kwargs.pop("max_cache_len", None)
        max_cache_len = min(sliding_window, max_cache_len) if max_cache_len is not None else sliding_window
        super().__init__(*args, max_cache_len=max_cache_len, *args, **kwargs)

    def update(
        self,
        key_states: ms.Tensor,
        value_states: ms.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[ms.Tensor, ms.Tensor]:
        """
        Update the sliding window cache tensors in place.

        Args:
            key_states (`ms.Tensor`): The new key states to cache.
            value_states (`ms.Tensor`): The new value states to cache.
            cache_kwargs (`dict[str, Any]`, *optional*): Additional arguments for the cache.

        Returns:
            tuple[`ms.Tensor`, `ms.Tensor`]: The updated key and value states.
        """
        cache_position = cache_kwargs.get("cache_position") if cache_kwargs else None
        if cache_position is None:
            raise ValueError("`cache_position` must be provided for SlidingWindowLayer.")

        key_states = key_states.to(self.keys.dtype)
        value_states = value_states.to(self.values.dtype)

        # Handle prefill phase when prompt length > sliding_window_size.
        # Note that we store cropped key/value states in the cache but return the full key/value states.
        if cache_position.shape[0] > self.max_cache_len:
            new_k = key_states[:, :, -self.max_cache_len :, :]
            new_v = value_states[:, :, -self.max_cache_len :, :]
            self.keys.copy_(new_k)
            self.values.copy_(new_v)
            return key_states, value_states

        # Sliding window logic for generation phase or prefill < window
        slicing = mint.arange(self.max_cache_len)
        current_seq_len = cache_position[-1] + 1  # Use last position to determine current length
        to_shift = current_seq_len > self.max_cache_len
        indices = (slicing + to_shift.sum()) % self.max_cache_len

        k_out_shifted = self.keys[:, :, indices]
        v_out_shifted = self.values[:, :, indices]

        # Clamp cache_position to determine the *target index* within the shifted cache view
        update_position = cache_position.clamp(min=0, max=self.max_cache_len - 1)

        try:
            k_out_updated = k_out_shifted.index_copy(2, update_position, key_states)
            v_out_updated = v_out_shifted.index_copy(2, update_position, value_states)
        except NotImplementedError:
            # Fallback for MPS: clone and modify the clone
            k_out_updated = k_out_shifted.clone()
            v_out_updated = v_out_shifted.clone()
            k_out_updated[:, :, update_position] = key_states
            v_out_updated[:, :, update_position] = value_states

        self.keys.copy_(k_out_updated)
        self.values.copy_(v_out_updated)
        return self.keys, self.values

    def get_mask_sizes(self, cache_position: ms.Tensor) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the attention mask"""
        query_length = cache_position.shape[0]
        first_cache_position = cache_position[0]

        kv_offset = mint.clamp(first_cache_position - self.max_cache_len + 1, min=0)
        # This is not general (see HybridChunkedCache for the whole general case), but it's what the cache returns
        kv_length = max(query_length, self.max_cache_len)
        return kv_length, kv_offset


class ChunkedSlidingLayer(SlidingWindowLayer):
    """
    An extended SlidingWindowLayer that supports prefill chunking, originally implemented for Llama 4.

    See `SlidingWindowLayer` for details on common methods that are implemented by all cache layers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cumulative_length = 0

    def update(
        self,
        key_states: ms.Tensor,
        value_states: ms.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[ms.Tensor, ms.Tensor]:
        cache_position = cache_kwargs.get("cache_position") if cache_kwargs else None
        if cache_position is None:
            raise ValueError("`cache_position` must be provided for ChunkedSlidingLayer.")

        cumulative_length = self.cumulative_length
        self.cumulative_length += key_states.shape[-2]
        is_full = cumulative_length >= self.max_cache_len

        if is_full:
            full_key_states = mint.cat((self.keys[:, :, 1:, :], key_states), dim=-2)
            full_value_states = mint.cat((self.values[:, :, 1:, :], value_states), dim=-2)
            # Fast decoding path -> here as the effective size is still sliding window, it is extremely important
            # to return `self.key_cache[layer_idx]` and `self.value_cache[layer_idx]`, as they have the fixed address
            # in memory (the values are the same as the full states, but not the address!!)
            if key_states.shape[-2] == 1:
                self.keys.copy_(full_key_states)
                self.values.copy_(full_value_states)
                return self.keys, self.values
        elif not is_full and cumulative_length + key_states.shape[2] > self.max_cache_len:
            if cumulative_length == 0:
                full_key_states = key_states
                full_value_states = value_states
            else:
                full_key_states = mint.cat((self.keys[:, :, :cumulative_length, :], key_states), dim=-2)
                full_value_states = mint.cat((self.values[:, :, :cumulative_length, :], value_states), dim=-2)
        else:
            try:
                self.keys.index_copy_(2, cache_position, key_states)
                self.values.index_copy_(2, cache_position, value_states)
            except NotImplementedError:
                self.keys[:, :, cache_position] = key_states
                self.values[:, :, cache_position] = value_states
            return self.keys, self.values

        self.keys.copy_(full_key_states[:, :, -self.max_cache_len :, :])
        self.values.copy_(full_value_states[:, :, -self.max_cache_len :, :])
        return full_key_states, full_value_states

    def reset(self) -> None:
        super().reset()
        self.cumulative_length = 0

    def get_mask_sizes(self, cache_position: ms.Tensor) -> tuple[int, int]:
        query_length = cache_position.shape[0]
        first_cache_position = cache_position[0]
        sliding_window = self.max_cache_len

        kv_offset = mint.clamp(first_cache_position - sliding_window + 1, min=0)
        # This is the true general case for any Cache using local attention (sliding or chunked)
        if first_cache_position >= sliding_window:
            # Here the Cache is already full
            kv_length = sliding_window + query_length - 1
        elif first_cache_position < sliding_window and first_cache_position + query_length > sliding_window:
            # Here the Cache becomes full with the new input
            kv_length = first_cache_position + query_length
        else:
            # Here the Cache is still smaller than the local size, but we return the local size as it's static
            kv_length = sliding_window
        return kv_length, kv_offset


class CacheProcessor:
    """
    Base class for cache processors. It defines a pre-update and post-update methods that are called before and after the cache update.
    This class should be subclassed.
    """

    def __init__(self, cache: "Cache", **kwargs) -> None:
        """
        Initialize the processor and perform compatibility checks with the cache.

        Args:
            cache (`Cache`): The cache instance this processor will be applied to.
            **kwargs: Additional arguments that may be needed for initialization.
        """
        raise NotImplementedError(f"Make sure to implement `init` in {self.__class__.__name__}.")

    def pre_update(
        self,
        cache: "Cache",
        key_states: ms.Tensor,
        value_states: ms.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[ms.Tensor, ms.Tensor]:
        """
        Function called before the cache update. Can modify the key/value states.

        Args:
            cache (`Cache`): The cache instance.
            key_states (`ms.Tensor`): The new key states to cache.
            value_states (`ms.Tensor`): The new value states to cache.
            layer_idx (`int`): The index of the layer to cache the states for.
            cache_kwargs (`dict[str, Any]`, *optional*): Additional arguments for the cache.

        Returns:
            The modified key and value states.
        """
        return key_states, value_states

    def post_update(
        self,
        cache: "Cache",
        key_tensors: ms.Tensor,
        value_tensors: ms.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[ms.Tensor, ms.Tensor]:
        """
        Function called after the cache update. Can process the cached data.

        Args:
            cache (`Cache`): The cache instance.
            key_states (`ms.Tensor`): The key states that were cached.
            value_states (`ms.Tensor`): The value states that were cached.
            layer_idx (`int`): The index of the layer that was updated.
            cache_kwargs (`dict[str, Any]`, *optional*): Additional arguments for the cache.

        Returns:
            The final key and value states to return to the model.
        """
        return key_tensors, value_tensors


class OffloadedCacheProcessor(CacheProcessor):
    """
    A cache processor that offloads cache tensors to conserve accelerator memory.

    This processor manages moving cache tensors between accelerator and CPU memory,
    using asynchronous prefetching to minimize performance impact. Works with both
    dynamic and static layers.
    """

    def __init__(self, cache: "Cache", **kwargs):
        raise NotImplementedError


class QuantizedCacheProcessor(CacheProcessor):
    """
    A cache processor that applies quantization to cache tensors to reduce memory usage.

    This processor quantizes cache tensors after they are stored, maintaining a residual
    length in original precision and quantizing older tokens.
    """

    def __init__(
        self,
        cache: "Cache",
        backend: str = "quanto",
        nbits: int = 4,
        axis_key: int = 0,
        axis_value: int = 0,
        q_group_size: int = 64,
        residual_length: int = 128,
        compute_dtype: ms.Type = ms.float16,
    ):
        """
        Parameters:
            backend (`str`, defaults to `"quanto"`):
                Backend to use when performing quantization, Can be one of [`quanto`, `HQQ`]
            nbits (`int`, defaults to 4):
                Number of bits, can be 2 or 4 for the `quanto` backend and one of [1, 2, 3, 4, 8] for the `HQQ` backend. Defaults to 2.
            axis_key (`int`, defaults to 0):
                Axis over which to perform grouping for the key tensors. Can be [0, -1] for `quanto` backend and [0, 1] for `HQQ` backend.
            axis_value (`int`, defaults to 0):
                Axis over which to perform grouping for the value tensors. Can be [0, -1] for `quanto` backend and [0, 1] for `HQQ` backend.
            q_group_size (`int`, defaults to 64):
                Size of the quantization group, should be a divisor of the model's hidden dimension.
                Defaults to 64.
            residual_length (`int`, defaults to 128):
                Length of the residual cache which will always be stored in original precision.
                Defaults to 128.
            compute_dtype (`ms.Type`, defaults to `ms.float16`):
                The default dtype used for computations in the model. Keys and Values will be cast to this dtype after dequantization.
        """
        raise NotImplementedError


class QuantoQuantizedCacheProcessor(QuantizedCacheProcessor):
    """
    Quantized cache processor that uses `quanto` as a backend to perform quantization.
    Current implementation supports `int2` and `int4` dtypes only.
    """

    def __init__(
        self,
        cache: "Cache",
        backend: str = "quanto",
        nbits: int = 4,
        axis_key: int = 0,
        axis_value: int = 0,
        q_group_size: int = 64,
        residual_length: int = 128,
        compute_dtype: ms.Type = ms.float16,
    ) -> None:
        """Initialize the quanto quantization processor."""
        super().__init__(cache, backend, nbits, axis_key, axis_value, q_group_size, residual_length, compute_dtype)

        raise NotImplementedError


class HQQQuantizedCacheProcessor(QuantizedCacheProcessor):
    """
    Quantized cache processor that uses `HQQ` as a backend to perform quantization.
    Current implementation supports `int2`, `int4`, `int8` dtypes.
    """

    def __init__(
        self,
        cache: "Cache",
        backend: str = "quanto",
        nbits: int = 4,
        axis_key: int = 0,
        axis_value: int = 0,
        q_group_size: int = 64,
        residual_length: int = 128,
        compute_dtype: ms.Type = ms.float16,
    ) -> None:
        """Initialize the HQQ quantization processor."""
        super().__init__(cache, backend, nbits, axis_key, axis_value, q_group_size, residual_length, compute_dtype)
        raise NotImplementedError


def apply_processors(
    fn: Callable[..., tuple[ms.Tensor, ms.Tensor]],
) -> Callable[..., tuple[ms.Tensor, ms.Tensor]]:
    @functools.wraps(fn)
    def _wrapped_update(
        self,
        key_states: ms.Tensor,
        value_states: ms.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[ms.Tensor, ms.Tensor]:
        """
        Wrapper around the update method to apply cache processors.
        """
        if self.cache_processor is not None:
            key_states, value_states = self.cache_processor.pre_update(
                self, key_states, value_states, layer_idx, cache_kwargs
            )

        key_tensors, value_tensors = fn(self, key_states, value_states, layer_idx, cache_kwargs)

        if self.cache_processor is not None:
            key_tensors, value_tensors = self.cache_processor.post_update(
                self, key_tensors, value_tensors, layer_idx, cache_kwargs
            )

        return key_tensors, value_tensors

    return _wrapped_update


class KeyValuesWrapper:
    """Helper class for Cache that simulates layer-indexed key/value lists from a layered cache.
    This allows for BC access and writing, e.g., cache.key_cache[idx] = ...
    Deprecated in favor of Cache.layers[idx].keys/values. TODO: remove in v4.56.0"""

    def __init__(self, layers, cache_type="keys"):
        self.layers = layers
        self.cache_type = cache_type

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [getattr(layer, self.cache_type) for layer in self.layers[idx]]
        return getattr(self.layers[idx], self.cache_type)

    def __setitem__(self, idx, value):
        if isinstance(idx, slice):
            for layer, val in zip(self.layers[idx], value):
                setattr(layer, self.cache_type, val)
        else:
            setattr(self.layers[idx], self.cache_type, value)

    def __len__(self):
        return len(self.layers)

    def __iter__(self):
        for layer in self.layers:
            yield getattr(layer, self.cache_type)

    def __bool__(self):
        return bool(self.layers)


class Cache:
    """
    Base container for per-layer key/value caches.

    A `Cache` behaves like a list of `CacheLayerMixin` objects, one per model layer.
    Sub-classes such as `DynamicCache`, `StaticCache`, or `SlidingWindowCache`
    simply pre-select which `CacheLayerMixin` class to use and may attach a
    `CacheProcessor` (off-loading, quantization).

    Example
    -------
    ```python
    from mindone.transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    tok   = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    inputs = tok("Hello", return_tensors="np")
    for key in inputs.keys():
        inputs[key] = ms.tensor(inputs[key])

    cache = DynamicCache()
    outputs = model(**inputs, past_key_values=cache, use_cache=True)
    ```

    Parameters:
        layer_classes (`type[CacheLayerMixin]` or `list[type[CacheLayerMixin]]`):
            A list of `CacheLayerMixin` classes to instantiate for the cache. If only a `CacheLayerMixin` class is
            provided, then it is used for all layers.
        config (`PretrainedConfig`, *optional*):
            Model configuration used to infer number of layers, head sizes, default
            device/dtype, etc.
        cache_processor (`CacheProcessor` or `str`, *optional*):
            Cache processor to apply (e.g., "offloaded", "quanto_quantized", "hqq_quantized")
            or a CacheProcessor class.
        max_batch_size (`int`, *optional*): Maximum batch size for static caches.
        max_cache_len (`int`, *optional*): Maximum sequence length. For hybrid caches, SlidingWindowLayers are
            clamped to `min(sliding_window, max_cache_len)`, StaticLayers use full `max_cache_len`.
        dtype (`ms.Type`, *optional*): Data type for cache tensors.
        tp_size (`int`, *optional*): Tensor parallel size to adjust the number of key/value heads.

    Additional keyword arguments are forwarded to the chosen layers constructor(s) and CacheProcessors. See the
    documentation of the relevant `CacheLayerMixin` class and `CacheProcessor` class for more details.
    """

    def __init__(
        self,
        layer_classes: Union[list[type[CacheLayerMixin]], type[CacheLayerMixin]],
        config: Optional[PretrainedConfig] = None,
        cache_processor: Optional[Union[str, type[CacheProcessor]]] = None,
        max_batch_size: Optional[int] = None,
        max_cache_len: Optional[int] = None,
        dtype: Optional[ms.Type] = None,
        tp_size: Optional[int] = None,
        **kwargs,
    ):
        self.layers: list[CacheLayerMixin] = []
        self.layer_classes = layer_classes

        processor_class = PROCESSOR_CLASS_MAP[cache_processor] if isinstance(cache_processor, str) else cache_processor
        kwargs.update(
            max_batch_size=max_batch_size,
            max_cache_len=max_cache_len,
            dtype=dtype,
            tp_size=tp_size,
        )
        processor_kwargs, kwargs = parse_processor_args(processor_class, kwargs)

        self.layer_init_kwargs = parse_layer_args_from_model_config(config, **kwargs)
        self.num_hidden_layers = getattr(config, "num_hidden_layers", 1)

        self.append_new_layers(self.num_hidden_layers - 1)
        self.cache_processor = processor_class(self, **processor_kwargs) if processor_class is not None else None

    def __getitem__(self, layer_idx: int) -> tuple[ms.Tensor, ms.Tensor]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self.layers):
            return self.layers[layer_idx].keys, self.layers[layer_idx].values
        else:
            raise KeyError(
                f"Cache only has {len(self.layers)} layers, attempted to access layer with index {layer_idx}"
            )

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.layers[layer_idx].keys, self.layers[layer_idx].values)

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        # Best effort BC support for old-style caches like Mambas, Falcon, HybridChunked that rely on __len__
        if getattr(self, "layers", None) is None:
            if getattr(self, "key_cache", None) is not None:
                return len(self.key_cache)
            return 0
        # Empty dynamic caches initialize an empty layer to be ready for first update
        dynamic_empty = (
            getattr(self, "layers", None) is not None
            and len(self.layers) == 1
            and isinstance(self.layers[0], DynamicLayer)
            and self.layers[0].keys is None
        )
        return len(self.layers) if not dynamic_empty else 0

    def __repr__(self):
        return f"{self.__class__.__name__}(layers={self.layers})"

    def append_new_layers(self, layer_idx: int) -> None:
        """
        Appends layers to the cache until the layer `layer_idx` is reached.
        Used for preallocation in static caches and on the fly in dynamic caches.

        Args:
            layer_idx (`int`):
                The index of the layer to append.
        """
        while len(self.layers) <= layer_idx:
            kwargs = self.layer_init_kwargs.copy()
            if self.layer_init_kwargs.get("layer_device_map", None) is not None:
                kwargs["device"] = kwargs.pop("layer_device_map")[layer_idx]

            new_layer_class = (
                self.layer_classes[len(self.layers)] if isinstance(self.layer_classes, list) else self.layer_classes
            )
            new_layer = new_layer_class(**kwargs)
            self.layers.append(new_layer)

    @apply_processors
    def update(
        self,
        key_states: ms.Tensor,
        value_states: ms.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[ms.Tensor, ms.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`ms.Tensor`):
                The new key states to cache.
            value_states (`ms.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`dict[str, Any]`, *optional*):
                Additional arguments for the cache subclass. These are specific to each subclass and allow new types of
                cache to be created.

        Return:
            A tuple containing the updated key and value states.
        """
        self.append_new_layers(layer_idx)
        return self.layers[layer_idx].update(key_states, value_states, cache_kwargs)

    def get_seq_length(self, layer_idx: int = 0, cache_position=None) -> int:
        """Returns the sequence length of the cache for the given layer. TODO: deprecate in favor of cache_position"""
        if layer_idx >= len(self.layers):
            return 0
        # Hack since QuantizedCache messes with keys shape as it becomes the residual cache
        if self.cache_processor is not None and isinstance(self.cache_processor, QuantizedCacheProcessor):
            return self.cache_processor.erased_length + self.layers[layer_idx].get_seq_length(cache_position)
        return self.layers[layer_idx].get_seq_length(cache_position)

    def get_mask_sizes(self, cache_position: ms.Tensor, layer_idx: int) -> tuple[int, int]:
        """
        Return a tuple (kv_length, kv_offset) corresponding to the length and offset that will be returned for
        the given layer at `layer_idx`.
        The masks are then prepared according to the given lengths (kv_length, kv_offset) and patterns (i.e. sliding_window, chunk_size),
        for each layer.
        """
        kv_length, kv_offset = self.layers[layer_idx].get_mask_sizes(cache_position)
        return kv_length, kv_offset

    @property
    def key_cache(self) -> KeyValuesWrapper:
        """List-like object of key cache tensors indexed by layer. Deprecated in favor of `cache.layers[idx].keys`"""
        logger.warning_once(
            "`cache.key_cache[idx]` is deprecated and will be removed in v4.56.0. Use `cache.layers[idx].keys` instead."
        )
        return KeyValuesWrapper(self.layers, "keys")

    @property
    def value_cache(self) -> KeyValuesWrapper:
        """List-like object of value cache tensors indexed by layer. Deprecated in favor of `cache.layers[idx].values`"""
        logger.warning_once(
            "`cache.value_cache[idx]` is deprecated and will be removed in v4.56.0. Use `cache.layers[idx].values` instead."
        )
        return KeyValuesWrapper(self.layers, "values")

    # Wrappers for layer operations and properties ###

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        """Returns maximum sequence length of the cache object. Dynamic caches do not have a maximum length."""
        return self.layers[layer_idx].get_max_cache_shape()

    def reset(self):
        """Recursively reset all layers tensors"""
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].reset()

    def reorder_cache(self, beam_idx: ms.Tensor):
        """Reorder the cache for beam search"""
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].reorder_cache(beam_idx)

    def crop(self, max_length: int):
        """Crop the cache to the given length"""
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].crop(max_length)

    def batch_repeat_interleave(self, repeats: int):
        """Repeat and interleave the cache"""
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices: ms.Tensor):
        """Select indices from the cache"""
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].batch_select_indices(indices)

    @property
    def max_batch_size(self) -> int:
        """Return the maximum batch size of the cache"""
        values = [layer.max_batch_size for layer in self.layers]
        if len(set(values)) > 1:
            raise ValueError(f"Max batch size is not consistent across layers: {values}")
        return values[0]

    @property
    def max_cache_len(self) -> int:
        """Return the maximum cache length of the cache"""
        values = [layer.max_cache_len for layer in self.layers]
        return max(values)

    @property
    def is_compileable(self) -> bool:
        """Return whether the cache is compileable"""
        return all(layer.is_compileable for layer in self.layers)

    @property
    def is_sliding(self) -> list[bool]:
        """Return whether the layers of the cache are sliding window"""
        return [getattr(layer, "is_sliding", False) for layer in self.layers]


class DynamicCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.

    See `Cache` for details on common methods that are implemented by all cache classes.

    Example:

        ```python
        >>> from transformers import AutoTokenizer
        >>> from mindone.transformers import AutoModelForCausalLM, DynamicCache

        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

        >>> inputs = tokenizer(text="My name is Qwen2", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> past_key_values = DynamicCache()
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        DynamicCache()
        ```
    """

    # Specialized constructor for DDP cache data, needed for BC
    def __init__(self, ddp_cache_data: Optional[Iterable[tuple[ms.Tensor, ms.Tensor]]] = None, *args, **kwargs):
        super().__init__(layer_classes=DynamicLayer, *args, **kwargs)
        # `ddp_cache_data` was originally added for compatibility with `torch.distributed` (DDP). See #36212
        # and #36373 for more information. In a nutshell, it is `map(gather_map, zip(*caches))`, i.e. each item in the
        # iterable contains the key and value states for a layer gathered across replicas by torch.distributed
        # (shape=[global batch size, num_heads, seq_len, head_dim]).
        # WARNING: `ddp_cache_data` must be the first argument in `__init__`, otherwise we'll break
        # compatibility. The name of the argument doesn't matter.
        if ddp_cache_data is not None:
            for key_states, value_states in ddp_cache_data:
                self.layers.append(DynamicLayer.from_tensors(key_states, value_states))

    def to_legacy_cache(self) -> tuple[tuple[ms.Tensor, ms.Tensor], ...]:
        """
        Converts the `Cache` instance into the its equivalent in the legacy cache format. Used for
        backward compatibility.
        """
        legacy_cache = ()
        for layer in self.layers:
            legacy_cache += ((layer.keys, layer.values),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values: tuple[tuple[ms.Tensor, ms.Tensor], ...]) -> "Cache":
        """
        Converts a cache in the legacy cache format into an equivalent `Cache`. Used for
        backward compatibility.
        """
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache


class StaticCache(Cache):
    """
    Static Cache class to be used with `mindspore.jit(model)`.

    See `Cache` for details on common methods that are implemented by all cache classes.

    Example:

        ```python
        >>> from transformers import AutoTokenizer
        >>> from mindone.transformers import AutoModelForCausalLM, StaticCache

        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

        >>> inputs = tokenizer(text="My name is Llama", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> # Leave empty space for 10 new tokens, which can be used when calling forward iteratively 10 times to generate
        >>> max_generated_length = inputs.input_ids.shape[1] + 10
        >>> past_key_values = StaticCache(config=model.config, max_batch_size=1, max_cache_len=max_generated_length, device=model.device, dtype=model.dtype)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        StaticCache()
        ```
    """

    def __init__(self, *args, **kwargs):
        super().__init__(layer_classes=StaticLayer, *args, **kwargs)


class SlidingWindowCache(Cache):
    """
    Sliding Window Cache class to be used with `mindspore.jit` for models like Mistral that support sliding window attention.
    Every time when we try to update the cache, we compute the `indices` based on `cache_position >= self.sliding_window - 1`,
    if true(which means the cache can not hold all the old key value states and new states together because of the sliding window constraint),
    we need to do a cycle shift based on `indices` to replace the oldest states by the new key value states passed in.

    The `to_shift` is only true once we are above sliding_window. Thus with `sliding_window==64`:

    indices = (slicing + to_shift[-1].sum()-1) % self.sliding_window
    tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
        55, 56, 57, 58, 59, 60, 61, 62, 63,  0])

    We overwrite the cache using these, then we always write at cache_position (clamped to `sliding_window`)

    See `Cache` for details on common methods that are implemented by all cache classes.

    Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, SlidingWindowCache

        >>> model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
        >>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

        >>> inputs = tokenizer(text="My name is Mistral", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> # Leave empty space for 10 new tokens, which can be used when calling forward iteratively 10 times to generate
        >>> max_generated_length = inputs.input_ids.shape[1] + 10
        >>> past_key_values = SlidingWindowCache(config=model.config, max_batch_size=1, max_cache_len=max_generated_length, device=model.device, dtype=model.dtype)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        SlidingWindowCache()
        ```
    """

    def __init__(self, *args, **kwargs):
        super().__init__(layer_classes=SlidingWindowLayer, *args, **kwargs)


class HybridCache(Cache):
    """
    Hybrid Cache class to be used with `mindspore.jit` for models that alternate between a local sliding window
    attention and global attention in every other layer (originally implemented for Gemma2).
    Under the hood, Hybrid Cache leverages ["SlidingWindowCache"] for sliding window attention and ["StaticCache"]
    for global attention. For more information, see the documentation of those layer types.

    See `Cache` for details on common methods that are implemented by all cache classes.

    Example:

        ```python
        >>> from transformers import AutoTokenizer
        >>> from mindone.transformers import AutoModelForCausalLM, HybridCache

        >>> model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

        >>> inputs = tokenizer(text="My name is Gemma", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> # Leave empty space for 10 new tokens, which can be used when calling forward iteratively 10 times to generate
        >>> max_generated_length = inputs.input_ids.shape[1] + 10
        >>> past_key_values = HybridCache(config=model.config, max_batch_size=1, max_cache_len=max_generated_length, dtype=model.dtype)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        HybridCache()
        ```
    """

    def __init__(self, config: PretrainedConfig, *args, **kwargs):
        if hasattr(config, "layer_types"):
            layer_classes = [LAYER_CLASS_MAP[layer_type] for layer_type in config.layer_types]
        else:
            # In this case, fall back to StaticCache
            layer_classes = [StaticLayer] * config.num_hidden_layers
        super().__init__(config=config, layer_classes=layer_classes, *args, **kwargs)


class EncoderDecoderCache(Cache):
    """
    Base, abstract class for all encoder-decoder caches. Can be used to hold combinations of self-attention and
    cross-attention caches.

    See `Cache` for details on common methods that are implemented by all cache classes.

    Example:

        ```python
        >>> from mindone.transformers import AutoProcessor, AutoModelForCausalLM, DynamicCache, EncoderDecoderCache

        >>> model = AutoModelForCausalLM.from_pretrained("openai/whisper-small")
        >>> processor = AutoProcessor.from_pretrained("openai/whisper-small")

        >>> inputs = processor(audio=YOUR-AUDIO, return_tensors="pt")

        >>> # Prepare cache classes for encoder and decoder and pass it to model's forward
        >>> self_attention_cache = DynamicCache()
        >>> cross_attention_cache = DynamicCache()
        >>> past_key_values = EncoderDecoderCache(self_attention_cache, cross_attention_cache)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        EncoderDecoderCache()
        ```

    """

    # Override @property from Cache
    is_compileable = None

    def __init__(self, self_attention_cache: Cache, cross_attention_cache: Cache):
        super().__init__(layer_classes=DynamicLayer)
        self.self_attention_cache = self_attention_cache
        self.cross_attention_cache = cross_attention_cache
        self.is_compileable = getattr(self.self_attention_cache, "is_compileable", False)

        self.is_updated = {}
        for layer_idx in range(len(cross_attention_cache)):
            self.is_updated[layer_idx] = bool(cross_attention_cache.get_seq_length(layer_idx) > 0)

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (
                self.self_attention_cache.layers[layer_idx].keys,
                self.self_attention_cache.layers[layer_idx].values,
                self.cross_attention_cache.layers[layer_idx].keys,
                self.cross_attention_cache.layers[layer_idx].values,
            )

    def __getitem__(self, layer_idx: int) -> tuple[ms.Tensor, ms.Tensor, ms.Tensor, ms.Tensor]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (
                self.self_attention_cache.layers[layer_idx].keys,
                self.self_attention_cache.layers[layer_idx].values,
                self.cross_attention_cache.layers[layer_idx].keys,
                self.cross_attention_cache.layers[layer_idx].values,
            )
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.self_attention_cache)

    def to_legacy_cache(self) -> tuple[tuple[ms.Tensor]]:
        """Converts the `EncoderDecoderCache` instance into its equivalent in the legacy cache format."""
        legacy_cache = ()
        if len(self.cross_attention_cache) > 0:
            for self_attn, cross_attn in zip(
                self.self_attention_cache.to_legacy_cache(), self.cross_attention_cache.to_legacy_cache()
            ):
                legacy_cache += (self_attn + cross_attn,)
        else:
            legacy_cache = self.self_attention_cache.to_legacy_cache()
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values: tuple[tuple[ms.Tensor, ms.Tensor], ...]) -> "EncoderDecoderCache":
        """Converts a cache in the legacy cache format into an equivalent `EncoderDecoderCache`."""
        cache = cls(
            self_attention_cache=DynamicCache(),
            cross_attention_cache=DynamicCache(),
        )
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx][:2]
                cache.self_attention_cache.update(key_states, value_states, layer_idx)
                if len(past_key_values[layer_idx]) > 2:
                    key_states, value_states = past_key_values[layer_idx][2:]
                    cache.cross_attention_cache.update(key_states, value_states, layer_idx)
                    cache.is_updated[layer_idx] = True
        return cache

    def get_seq_length(self, layer_idx: Optional[int] = 0, cache_position=None) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # check if empty list because in case of static cache it will be a tensors and we can't check `if not ms.Tensor`
        return self.self_attention_cache.get_seq_length(layer_idx, cache_position)

    def reset(self):
        if hasattr(self.self_attention_cache, "reset"):
            self.self_attention_cache.reset()
        if hasattr(self.cross_attention_cache, "reset"):
            self.cross_attention_cache.reset()
        elif not hasattr(self.self_attention_cache, "reset") and not hasattr(self.cross_attention_cache, "reset"):
            raise ValueError(
                "Neither self nor cross-attention cache have valid `.reset()` methods. `.reset()` should "
                "only be called on compatible cache classes, such as `StaticCache` or `SlidingWindowCache`. "
                f"Got {self.self_attention_cache.__str__()} for the self attention cache and "
                f"{self.cross_attention_cache.__str__()} for the cross attention cache."
            )
        for layer_idx in self.is_updated:
            self.is_updated[layer_idx] = False

    def reorder_cache(self, beam_idx: ms.Tensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        self.self_attention_cache.reorder_cache(beam_idx)
        self.cross_attention_cache.reorder_cache(beam_idx)

    def check_dynamic_cache(self, method: str):
        if not (
            isinstance(self.self_attention_cache, DynamicCache) and isinstance(self.cross_attention_cache, DynamicCache)
        ):
            raise ValueError(
                f"`{method}` is only defined for dynamic cache, got {self.self_attention_cache.__str__()} for the self "
                f"attention cache and {self.cross_attention_cache.__str__()} for the cross attention cache."
            )

    # TODO(gante, sanchit-gandhi): move following functionality into `.generate`
    def crop(self, maximum_length: int):
        """
        Crop the past key values up to a new `maximum_length` in terms of tokens. `maximum_length` can also be
        negative to remove `maximum_length` tokens. This is used in assisted decoding and contrastive search.
        """
        self.check_dynamic_cache(self.crop.__name__)
        self.self_attention_cache.crop(maximum_length)

    def batch_split(self, full_batch_size: int, split_size: int) -> "list[EncoderDecoderCache]":
        """
        Split the current instance into a list of `DynamicCache` by the batch size. This will be used by
        `_split_model_inputs()` in `generation.utils`
        """
        self.check_dynamic_cache(self.batch_split.__name__)
        self_attention_cache = self.self_attention_cache.batch_split(full_batch_size, split_size)
        cross_attention_cache = self.cross_attention_cache.batch_split(full_batch_size, split_size)

        out = []
        for self_attn, cross_attn in zip(self_attention_cache, cross_attention_cache):
            out.append(EncoderDecoderCache(self_attn, cross_attn))
        return out

    def batch_repeat_interleave(self, repeats: int):
        """Repeat the cache `repeats` times in the batch dimension. Used in contrastive search."""
        self.check_dynamic_cache(self.batch_repeat_interleave.__name__)
        self.self_attention_cache.batch_repeat_interleave(repeats)
        self.cross_attention_cache.batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices: ms.Tensor):
        """Only keep the `indices` in the batch dimension of the cache. Used in contrastive search."""
        self.check_dynamic_cache(self.batch_select_indices.__name__)
        self.self_attention_cache.batch_select_indices(indices)
        self.cross_attention_cache.batch_select_indices(indices)

    def get_max_cache_shape(self) -> int:
        """Returns the maximum sequence length (i.e. max capacity) of the cache object"""
        return self.self_attention_cache.get_max_cache_shape()

    def get_mask_sizes(self, cache_position: ms.Tensor, layer_idx: int) -> tuple[int, int]:
        return self.self_attention_cache.get_mask_sizes(cache_position, layer_idx)


class MambaCache:
    def __init__(self):
        raise NotImplementedError


class OffloadedStaticCache(StaticCache):
    def __init__(self):
        raise NotImplementedError


def parse_processor_args(processor_class: Optional[type["CacheProcessor"]], kwargs: dict) -> tuple[dict, dict]:
    """
    Parse processor arguments from kwargs based on the processor class init signature.

    Args:
        processor_class: The processor class to inspect, or None
        kwargs: Dictionary of keyword arguments

    Returns:
        tuple: (processor_kwargs, remaining_kwargs)
    """
    try:
        params = list(inspect.signature(processor_class.__init__).parameters)[2:]
    except Exception:
        return {}, kwargs

    processor_kwargs = {k: kwargs[k] for k in params if k in kwargs}
    remaining_kwargs = {k: v for k, v in kwargs.items() if k not in processor_kwargs}
    return processor_kwargs, remaining_kwargs


def parse_layer_args_from_model_config(
    config: Optional[PretrainedConfig],
    batch_size: Optional[int] = None,
    max_cache_len: Optional[int] = None,
    dtype: Optional[ms.Type] = None,
    tp_size: Optional[int] = None,
    max_batch_size: Optional[int] = None,
) -> dict:
    """
    Parse layer arguments from model configuration for cache initialization.

    Args:
        config (`Optional[PretrainedConfig]`): Model configuration containing shape/device info.
        batch_size (`Optional[int]`): Batch size for cache initialization.
        max_cache_len (`Optional[int]`): Maximum sequence length for cache.
        dtype (`Optional[ms.Type]`): Data type for cache tensors.
        tp_size (`Optional[int]`): Tensor parallel size to adjust number of key/value heads.
        max_batch_size (`Optional[int]`): Maximum batch size for cache initialization.

    Returns:
        `dict`: Dictionary containing parsed layer arguments for cache initialization.
    """
    # No model config -> must be a dynamic cache, return bare dict
    if config is None:
        return {}
    # Build the args dict for hybrid, sliding or static
    else:
        # Hybrid/Sliding caches require a config that supports sliding_window (max_cache_len already used)
        if (
            getattr(config, "layer_types", None) is not None
            and "sliding_attention" in config.layer_types
            and "full_attention" in config.layer_types
        ):
            if getattr(config, "sliding_window", None) is None:
                raise ValueError(
                    "Setting up a hybrid or sliding window KVCache requires the model config supporting "
                    "sliding window attention, please check if there is a `sliding_window` field in the model "
                    "config and it's not set to None."
                )
        # Adjust max_cache_len for sliding window layers (they can't be larger than sliding window)
        max_cache_len = max_cache_len or config.max_position_embeddings
        # Some model define a custom `head_dim` != config.hidden_size // config.num_attention_heads:
        head_dim = (
            config.head_dim
            if getattr(config, "head_dim", None) is not None
            else config.hidden_size // config.num_attention_heads
        )
        num_heads = (
            config.num_attention_heads
            if getattr(config, "num_key_value_heads", None) is None
            else config.num_key_value_heads
        )
        if tp_size is not None and tp_size > 1:
            if num_heads % tp_size != 0:
                raise ValueError(
                    f"Number of key value heads {num_heads} must be divisible by tensor parallel size {tp_size}."
                )
            # If the model is using tensor parallelism, we need to adjust the number of heads accordingly.
            num_heads //= tp_size
        layer_args = {
            "batch_size": max_batch_size if max_batch_size is not None else batch_size,
            "max_cache_len": max_cache_len,
            "dtype": dtype,
            "head_dim": head_dim,
            "num_heads": num_heads,
            "sliding_window": getattr(config, "sliding_window", None),
        }
        return {k: v for k, v in layer_args.items() if v is not None}


LAYER_CLASS_MAP: dict[str, type["CacheLayerMixin"]] = {
    "full_attention": StaticLayer,
    "sliding_attention": SlidingWindowLayer,
    "chunked_attention": ChunkedSlidingLayer,
}
PROCESSOR_CLASS_MAP: dict[str, type["CacheProcessor"]] = {
    "offloaded": OffloadedCacheProcessor,
    "quanto_quantized": QuantizedCacheProcessor,
    "hqq_quantized": HQQQuantizedCacheProcessor,
}


class CacheConfig:
    """
    Base class for cache configs
    """

    cache_implementation: None

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        """
        Constructs a CacheConfig instance from a dictionary of parameters.
        Args:
            config_dict (Dict[str, Any]): Dictionary containing configuration parameters.
            **kwargs: Additional keyword arguments to override dictionary values.
        Returns:
            CacheConfig: Instance of CacheConfig constructed from the dictionary.
        """
        config = cls(**config_dict)
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)
        return config

    # Copied from transformers.utils.quantization_config.QuantizationConfigMixin.to_json_file
    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default
                `QuantizationConfig()` is serialized to JSON file.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            config_dict = self.to_dict()
            json_string = json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

            writer.write(json_string)

    # Copied from transformers.utils.quantization_config.QuantizationConfigMixin.to_dict
    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        return copy.deepcopy(self.__dict__)

    # Copied from transformers.utils.quantization_config.QuantizationConfigMixin.__iter__
    def __iter__(self):
        """allows `dict(obj)` for situations where obj may be a dict or QuantizationConfigMixin"""
        for attr, value in copy.deepcopy(self.__dict__).items():
            yield attr, value

    # Copied from transformers.utils.quantization_config.QuantizationConfigMixin.__repr__
    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def to_json_string(self):
        """
        Serializes this instance to a JSON formatted string.
        Returns:
            str: JSON formatted string representing the configuration instance.
        """
        return json.dumps(self.__dict__, indent=2) + "\n"

    # Copied from transformers.utils.quantization_config.QuantizationConfigMixin.update
    def update(self, **kwargs):
        """
        Updates attributes of this class instance with attributes from `kwargs` if they match existing attributes,
        returning all the unused kwargs.

        Args:
            kwargs (`Dict[str, Any]`):
                Dictionary of attributes to tentatively update this class.

        Returns:
            `Dict[str, Any]`: Dictionary containing all the key-value pairs that were not used to update the instance.
        """
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                to_remove.append(key)

        # Remove all the attributes that were updated, without modifying the input dict
        unused_kwargs = {key: value for key, value in kwargs.items() if key not in to_remove}
        return unused_kwargs
