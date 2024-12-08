from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

import mindspore as ms
from mindspore import ops

logger = logging.get_logger(__name__)


def init_static_cache(config: PretrainedConfig, max_batch_size: int, max_cache_len: int, dtype=None):
    max_cache_len = config.max_position_embeddings if max_cache_len is None else max_cache_len
    # Some model define a custom `head_dim` != config.hidden_size // config.num_attention_heads
    head_dim = config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads

    dtype = dtype if dtype is not None else ms.float32
    num_key_value_heads = (
        config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
    )

    key_value_cache: Tuple[Tuple[ms.Tensor, ms.Tensor]] = ()
    cache_shape = (max_batch_size, num_key_value_heads, max_cache_len, head_dim)
    for _layer_index in range(config.num_hidden_layers):
        # Note: `mark_static_address` is used to tag the cache as an fixed data pointer, preventing cuda graph
        # breaks when updating the cache.
        new_layer_key_cache = ms.Tensor(np.zeros(cache_shape), dtype=dtype)
        new_layer_value_cache = ms.Tensor(np.zeros(cache_shape), dtype=dtype)
        key_value_cache += ((new_layer_key_cache, new_layer_value_cache),)

    return key_value_cache


# Notes: Only return the updated value, do not modifying the original `past_key_value` in-place !
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

    if cache_position.shape[0] == 1:
        k_out[:, :, cache_position] = key_states
        v_out[:, :, cache_position] = value_states
    else:
        # assert cache_position.shape[0] == k_out.shape[2]

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


@dataclass
class Cache:
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
            self.key_cache[layer_idx] = self.key_cache[layer_idx].gather(input_indices=beam_idx, axis=0)
            self.value_cache[layer_idx] = self.value_cache[layer_idx].gather(input_indices=beam_idx, axis=0)

    @property
    def seen_tokens(self):
        logger.warning_once(
            "The `seen_tokens` attribute is deprecated and will be removed in v4.41. Use the `cache_position` "
            "model input instead."
        )
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
            config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
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

        k_out[:, :, cache_position] = key_states
        v_out[:, :, cache_position] = value_states

        # update to self.key_cache?
        self.key_cache[layer_idx] = k_out
        self.value_cache[layer_idx] = v_out

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
