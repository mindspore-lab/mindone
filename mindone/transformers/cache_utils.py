"""
Adapted from https://github.com/huggingface/transformers/tree/main/src/transformers/cache_utils.py.

Cache utils.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, Optional

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

import mindspore as ms
from mindspore import mint

logger = logging.get_logger(__name__)


class CacheLayerMixin(ABC):
    """Base, abstract class for a single layer's cache."""

    is_compileable = False

    def __init__(self):
        self.keys: Optional[ms.Tensor] = None
        self.values: Optional[ms.Tensor] = None
        self.is_initialized = False

    def __repr__(self):
        return f"{self.__class__.__name__}"

    @abstractmethod
    def lazy_initialization(self, key_states: ms.Tensor):
        ...

    @abstractmethod
    def update(
        self,
        key_states: ms.Tensor,
        value_states: ms.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[ms.Tensor, ms.Tensor]:
        ...

    @abstractmethod
    def get_mask_sizes(self, cache_position: ms.Tensor) -> tuple[int, int]:
        ...

    @abstractmethod
    def get_seq_length(self) -> int:
        ...

    @abstractmethod
    def get_max_cache_shape(self) -> int:
        ...

    def offload(self):
        """Offload this layer's data to CPU device."""
        raise NotImplementedError("mindspore do not support offload/prefetch yet")

    def prefetch(self):
        """In case of layer offloading, this allows to move the data back to the layer's device ahead of time."""
        raise NotImplementedError("mindspore do not support offload/prefetch yet")

    def reset(self) -> None:
        """Resets the cache values while preserving the objects"""
        if self.is_initialized:
            self.keys.zero_()
            self.values.zero_()
        # This attribute is set on several Layers
        if hasattr(self, "cumulative_length"):
            self.cumulative_length = 0

    def reorder_cache(self, beam_idx: ms.Tensor) -> None:
        """Reorders this layer's cache for beam search."""
        if self.get_seq_length() > 0:
            self.keys = self.keys.index_select(0, beam_idx)
            self.values = self.values.index_select(0, beam_idx)


class DynamicLayer(CacheLayerMixin):
    """
    A cache layer that grows dynamically as more tokens are generated. This is the default for generative models.
    It stores the Key and Value states as tensors with shape `[batch_size, num_heads, seq_len, head_dim]`.

    See `CacheLayerMixin` for details on common methods that are implemented by all cache layers.
    """

    is_sliding = False

    # FIXME "mindspore.mint.cat" does not support operation between tensor shape like (0,) and (b, h, s, d)
    def lazy_initialization(self, key_states: ms.Tensor, value_states: ms.Tensor):
        self.keys = key_states
        self.values = value_states
        self.is_initialized = True

    def update(
        self,
        key_states: ms.Tensor,
        value_states: ms.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[ms.Tensor, ms.Tensor]:
        """
        Update the key and value caches in-place, and return the necessary keys and value states.

        Args:
            key_states (`ms.Tensor`):
                The new key states to cache.
            value_states (`ms.Tensor`):
                The new value states to cache.
            cache_kwargs (`dict[str, Any]`, *optional*):
                Additional arguments for the cache.

        Returns:
            tuple[`ms.Tensor`, `ms.Tensor`]: The key and value states.
        """
        # Lazy initialization. FIXME not really working now.
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        else:
            self.keys = mint.cat([self.keys, key_states], dim=-2)
            self.values = mint.cat([self.values, value_states], dim=-2)
        return self.keys, self.values

    def get_mask_sizes(self, cache_position: ms.Tensor) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the mask"""
        kv_offset = 0
        query_length = cache_position.shape[0]
        kv_length = self.get_seq_length() + query_length
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        """Returns the sequence length of the cached states."""
        if not self.is_initialized or self.keys.numel() == 0:
            return 0
        return self.keys.shape[-2]

    def get_max_cache_shape(self) -> int:
        """Returns the maximum sequence length of the cache object. DynamicLayer does not have a maximum length."""
        return -1

    def crop(self, max_length: int) -> None:
        """
        Crop the past key values up to a new `max_length` in terms of tokens. `max_length` can also be negative
        to remove `max_length` tokens.
        """
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)

        if self.get_seq_length() <= max_length:
            return

        self.keys = self.keys[..., :max_length, :]
        self.values = self.values[..., :max_length, :]

    def batch_repeat_interleave(self, repeats: int) -> None:
        """Repeat the cache `repeats` times in the batch dimension."""
        if self.get_seq_length() > 0:
            self.keys = self.keys.repeat_interleave(repeats, dim=0)
            self.values = self.values.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: ms.Tensor) -> None:
        """Only keep the `indices` in the batch dimension of the cache."""
        if self.get_seq_length() > 0:
            self.keys = self.keys[indices, ...]
            self.values = self.values[indices, ...]


class DynamicSlidingWindowLayer(DynamicLayer):
    """
    A cache layer that grows dynamically as more tokens are generated, up until the sliding window size.
    It stores the key and value states as tensors of shape `[batch_size, num_heads, min(seq_len, sliding_window), head_dim]`.
    """

    is_sliding = True

    def __init__(self, sliding_window: int):
        super().__init__()
        self.sliding_window = sliding_window
        self.cumulative_length = 0

    def update(
        self,
        key_states: ms.Tensor,
        value_states: ms.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[ms.Tensor, ms.Tensor]:
        """
        Update the key and value caches in-place, and return the necessary keys and value states.

        Args:
            key_states (`ms.Tensor`): The new key states to cache.
            value_states (`ms.Tensor`): The new value states to cache.
            cache_kwargs (`dict[str, Any]`, *optional*): Additional arguments for the cache.

        Returns:
            tuple[`ms.Tensor`, `ms.Tensor`]: The key and value states.
        """
        self.cumulative_length += key_states.shape[-2]

        if not self.is_initialized:
            # Lazy initialization
            # FIXME not really working now as upstream repo.
            # mint.cat does not support tensor([]) input
            full_key_states = key_states
            full_value_states = value_states
            self.is_initialized = True
        else:
            # Compute the full states
            full_key_states = mint.cat([self.keys, key_states], dim=-2)
            full_value_states = mint.cat([self.values, value_states], dim=-2)
        # Only cache the last `self.sliding_window - 1` tokens (or all of them if lower than that)
        self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]

        # Return the full states
        return full_key_states, full_value_states

    def get_mask_sizes(self, cache_position: ms.Tensor) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the attention mask"""
        query_length = cache_position.shape[0]
        is_full = self.cumulative_length >= self.sliding_window

        kv_offset = max(self.cumulative_length - self.sliding_window + 1, 0)
        if is_full:
            kv_length = self.sliding_window - 1 + query_length
        else:
            kv_length = self.cumulative_length + query_length

        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        """Returns the sequence length of the cached states."""
        return self.cumulative_length

    def get_max_cache_shape(self) -> int:
        """Return the maximum cache shape of the cache"""
        return self.sliding_window

    def crop(self, max_length: int) -> None:
        """
        Crop the past key values up to a new `max_length` in terms of tokens. `max_length` can also be
        negative to remove `max_length` tokens.
        """
        if self.get_seq_length() >= self.sliding_window:
            raise ValueError(
                "Cannot `crop` a `DynamicSlidingWindowLayer` after it has seen more tokens than its"
                "sliding window (otherwise some states are lost)"
            )
        super().crop(max_length)
        self.cumulative_length = self.keys.shape[-2]


class StaticLayer(CacheLayerMixin):
    """
    A static cache layer that stores the key and value states as static tensors of shape `[batch_size, num_heads, max_cache_len), head_dim]`.
    It lazily allocates its full backing tensors, and then mutates them in-place. Built for `mindspore.jit` support.

    Args:
        max_cache_len (`int`):
            Maximum number of tokens that can be stored, used for tensor preallocation."""

    is_compileable = True
    is_sliding = False

    def __init__(self, max_cache_len: int):
        super().__init__()
        self.max_cache_len = max_cache_len

    def lazy_initialization(self, key_states: ms.Tensor):
        """
        Lazy initialization of the keys and values tensors. This allows to get all properties (dtype, device,
        num_heads in case of TP etc...) at runtime directly, which is extremely practical as it avoids moving
        devices, dtypes etc later on for each `update` (which could break the static dynamo addresses as well).

        If this is unwanted, one can call `early_initialization(...)` on the Cache directly, which will call this
        function ahead-of-time (this is required for `torch.export` for example). Note that for `compile`, as we
        internally don't compile the prefill, this is guaranteed to have been called already when compiling.
        If compiling the prefill as well, e.g. calling `model.compile(...)` before `generate` with a static cache,
        it is still supported in general, but without guarantees depending on the compilation options (e.g. cuda graphs,
        i.e. `mode="reduce-overhead"` is known to fail). But it will in general work correctly, and prefill should
        not be compiled anyway for performances!
        """
        self.max_batch_size, self.num_heads, _, self.head_dim = key_states.shape
        self.dtype = key_states.dtype

        self.keys = mint.zeros(
            (self.max_batch_size, self.num_heads, self.max_cache_len, self.head_dim),
            dtype=self.dtype,
        )
        self.values = mint.zeros(
            (self.max_batch_size, self.num_heads, self.max_cache_len, self.head_dim),
            dtype=self.dtype,
        )

        self.is_initialized = True

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
        # Lazy initialization
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        # Some old models give None for `cache_position` or even omit passing `cache_kwargs` when used as cross-attention,
        # in which case we should copy the whole Layer (key_states.shape[-2] == self.max_cache_len)
        cache_position = cache_kwargs.get("cache_position") if cache_kwargs is not None else None
        cache_position = cache_position if cache_position is not None else mint.arange(key_states.shape[-2])

        # Update the cache
        try:
            self.keys.index_copy_(2, cache_position, key_states)
            self.values.index_copy_(2, cache_position, value_states)
        except Exception:  # MindSpore does not support index_copy_
            # Fallback for devices like MPS where index_copy_ might not be supported.
            self.keys[:, :, cache_position] = key_states
            self.values[:, :, cache_position] = value_states
        return self.keys, self.values

    def get_mask_sizes(self, cache_position: ms.Tensor) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the attention mask"""
        kv_offset = 0
        kv_length = self.max_cache_len
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        """Returns the sequence length of the cached states."""
        # Occupied cache == any slot in the 3rd dim (sequence length) holds a non-zero value. To save on compute, let's
        # limit the check to the first batch member and head dimension.
        return (self.keys[0, 0].any(dim=-1)).sum() if self.is_initialized else 0

    def get_max_cache_shape(self) -> int:
        """Return the maximum cache shape of the cache"""
        return self.max_cache_len


class StaticSlidingWindowLayer(StaticLayer):
    """
    A static cache layer that stores the key and value states as static tensors of shape
    `[batch_size, num_heads, min(max_cache_len, sliding_window), head_dim]`. It lazily allocates its full backing
    tensors, and then mutates them in-place. Built for `torch.compile` support.

    Args:
        max_cache_len (`int`):
            Maximum number of tokens that can be stored, used for tensor preallocation.
        sliding_window (`int`):
            The size of the sliding window.
    """

    is_sliding = True

    def __init__(self, max_cache_len: int, sliding_window: int):
        effective_max_cache_len = min(sliding_window, max_cache_len)
        super().__init__(max_cache_len=effective_max_cache_len)
        self.cumulative_length = 0

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

        # Lazy initialization
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        # Some old models give None for `cache_position` or even omit passing `cache_kwargs` when used as cross-attention,
        # in which case we should copy the whole Layer (key_states.shape[-2] == self.max_cache_len)
        cache_position = cache_kwargs.get("cache_position") if cache_kwargs is not None else None
        cache_position = cache_position if cache_position is not None else mint.arange(key_states.shape[-2])

        cumulative_length = self.cumulative_length
        is_full = cumulative_length >= self.max_cache_len
        # Update it now that we saved the value above
        self.cumulative_length += key_states.shape[-2]

        if is_full:
            # In general, we should use a much simpler `cat` here as well, independently of the states size. However,
            # dynamo is currently bugged when doing it - see https://github.com/pytorch/pytorch/issues/159855 for more details
            if key_states.shape[-2] == 1:
                # Roll all values to the left by 1 position
                new_keys = self.keys.roll(-1, dims=-2)
                new_values = self.values.roll(-1, dims=-2)
                # Overwrite the last position with new states
                # (note: very important to use a tensor to index here, see https://github.com/pytorch/pytorch/issues/159855)
                index = ms.tensor([-1], dtype=ms.int32)
                new_keys[:, :, index] = key_states
                new_values[:, :, index] = value_states

                # Copy back into `self` (do not just assign again) in order to keep the static dynamo address
                self.keys.copy_(new_keys)
                self.values.copy_(new_values)
                # Very important to return the `self` tensors here, as they have the static dynamo address
                return self.keys, self.values
            # Already full but using more than 1 new token (e.g. prefill caching, chat continuation, etc...)
            else:
                full_key_states = mint.cat((self.keys[:, :, 1:, :], key_states), dim=-2)
                full_value_states = mint.cat((self.values[:, :, 1:, :], value_states), dim=-2)
        # Not yet full, but becoming full on this update
        elif cumulative_length + key_states.shape[2] > self.max_cache_len:
            # Fast prefill path, no need to cat() in this case, as the cache is currently empty
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
            except Exception:
                self.keys[:, :, cache_position] = key_states
                self.values[:, :, cache_position] = value_states

            # Very important to return the `self` tensors here, as they have the static dynamo address
            return self.keys, self.values

        # We only cache the last `sliding_window` tokens
        self.keys.copy_(full_key_states[:, :, -self.max_cache_len :, :])
        self.values.copy_(full_value_states[:, :, -self.max_cache_len :, :])
        # we should return the whole states instead of `self.keys/values` here, as otherwise we lose some context
        return full_key_states, full_value_states

    def get_mask_sizes(self, cache_position: ms.Tensor) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the attention mask"""
        query_length = cache_position.shape[0]
        sliding_window = self.max_cache_len
        is_full = self.cumulative_length >= self.max_cache_len

        kv_offset = max(self.cumulative_length - sliding_window + 1, 0)
        # The cache is already full
        if is_full:
            kv_length = sliding_window + query_length - 1
        # Not yet full, but becoming full on this update
        elif self.cumulative_length + query_length > sliding_window:
            kv_length = self.cumulative_length + query_length
        # Here the Cache is still smaller than the local size, but we return the local size as it's static
        else:
            kv_length = sliding_window

        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        """Returns the sequence length of the cached states."""
        return self.cumulative_length


class QuantizedLayer(DynamicLayer):
    """
    A quantized layer similar to what is described in the
      [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache paper](https://huggingface.co/papers/2402.02750).
    It allows the model to generate longer sequence length without allocating too much memory for the key and value caches by
    applying quantization.

    The cache has two types of storage, one for original precision and one for the quantized cache. A `residual length`
    is set as a maximum capacity for the original precision cache. When the length goes beyond maximum capacity, the original
    precision cache is discarded and moved into the quantized cache. The quantization is done per-channel with a set `q_group_size`
    for both Keys and Values, in contrast to what was described in the paper.
    """

    def __init__(
        self,
        nbits: int = 4,
        axis_key: int = 0,
        axis_value: int = 0,
        q_group_size: int = 64,
        residual_length: int = 128,
    ):
        raise NotImplementedError(f"Not support {self.__class__.__name__} in mindspore yet.")


class QuantoQuantizedLayer(QuantizedLayer):
    def __init__(
        self,
        nbits: int = 4,
        axis_key: int = 0,
        axis_value: int = 0,
        q_group_size: int = 64,
        residual_length: int = 128,
    ):
        raise NotImplementedError(f"Not support {self.__class__.__name__} in mindspore yet.")


class Cache:
    """
    A `Cache` is mostly a list of `CacheLayerMixin` objects, one per model layer. It serves as a container for
    the Cache of each layer.

    Args:
        layers (`Optional`, *optional*):
            A list of pre-created `CacheLayerMixin`. If omitted (`None`), then `layer_class_to_replicate` will
            be used.
        layer_class_to_replicate (`type[CacheLayerMixin]`, *optional*):
            Only used if `layers` is omitted (`None`), in which case it will be used as the base class for each layer,
            and the layers will be added lazily as soon as `update` is called with a `layer_idx` greater than the current
            list of layers.
        offloading (`bool`, *optional*, defaults to `False`):
            Whether to perform offloading of the layers to `cpu`, to save GPU memory.
            Not support yet.
        offload_only_non_sliding (`bool`, *optional*, defaults to `True`):
            If `offloading` is `True`, this further decides if only the non-sliding layers will be offloaded (because
            usually the sliding layers are small in size, so there is no need to offload them, and skipping it is faster).
            Not support yet.
    """

    def __init__(
        self,
        layers: Optional[list[CacheLayerMixin]] = None,
        layer_class_to_replicate: Optional[type[CacheLayerMixin]] = None,
        offloading: bool = False,  # not support yet
        offload_only_non_sliding: bool = True,  # not support yet
    ):
        if layers is not None and layer_class_to_replicate is not None:
            raise ValueError(
                "You can construct a Cache either from a list `layers` of all the predefined `CacheLayer`, or from a "
                "`layer_class_to_replicate`, in which case the Cache will append a new layer corresponding to "
                "`layer_class_to_replicate` for each new call to `update` with an idx not already in the Cache."
            )
        if layers is None and layer_class_to_replicate is None:
            raise ValueError(
                "You should provide exactly one of `layers` or `layer_class_to_replicate` to initialize a Cache."
            )
        self.layers = layers if layers is not None else []
        self.layer_class_to_replicate = layer_class_to_replicate
        self.offloading = offloading
        if self.offloading:
            raise NotImplementedError(
                "mindspore do not support offload/prefetch yet, please set to `offloading` to `False`"
            )

    def __repr__(self):
        return f"{self.__class__.__name__}(layers={self.layers})"

    def prefetch(self, layer_idx: int, only_non_sliding: bool = True):
        """
        Prefetch a given layer on its device. If `only_non_sliding` is True, it will try to prefetch only the layers
        which are non-sliding. If the `layer_idx` is outside the range, this will circle back to the first layers.
        Note that we use a non-default stream for this, to avoid blocking.
        """
        raise NotImplementedError("mindspore do not support offload/prefetch yet")

    def offload(self, layer_idx: int, only_non_sliding: bool = True):
        """
        Offload a given `layer_idx`. If `only_non_sliding` is True, it will offload `layer_idx` only if it is a
        non-sliding layer. Note that we do it on the default stream, so that we ensure all earlier
        computation in the layer's `update` methods are finished.
        """
        raise NotImplementedError("mindspore do not support offload/prefetch yet")

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
        # In this case, the `layers` were not provided, and we must append as much as `layer_idx`
        if self.layer_class_to_replicate is not None:
            while len(self.layers) <= layer_idx:
                self.layers.append(self.layer_class_to_replicate())

        keys, values = self.layers[layer_idx].update(key_states, value_states, cache_kwargs)

        return keys, values

    def early_initialization(self, batch_size: int, num_heads: int, head_dim: int, dtype: ms.Type):
        """
        Initialize all the layers in advance (it's otherwise lazily initialized on the first `update` call).
        This is useful for our `export` recipes, as `export` needs everything in advance.
        """
        # Note that the initialization needs all dimensions (except -2), as well as device and dtype, so we use
        # this fake tensor approach. It has size 0 on the -2 dimension, so it does not allocate any data (it only
        # creates an empty tensor with correct shape, dtype and device), which is very efficient and practical
        fake_keys_tensor = mint.zeros((batch_size, num_heads, 0, head_dim), dtype=dtype)
        # Init all layers
        for layer in self.layers:
            layer.lazy_initialization(fake_keys_tensor)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Returns the sequence length of the cache for the given layer."""
        if layer_idx >= len(self.layers):
            return 0
        return self.layers[layer_idx].get_seq_length()

    def get_mask_sizes(self, cache_position: ms.Tensor, layer_idx: int) -> tuple[int, int]:
        """
        Return a tuple (kv_length, kv_offset) corresponding to the length and offset that will be returned for
        the given layer at `layer_idx`.
        The masks are then prepared according to the given lengths (kv_length, kv_offset) and patterns (i.e. sliding_window, chunk_size),
        for each layer.
        """
        # For DynamicCache, where the layers are created at runtime -> if it was not yet created, the size is
        # simply the shape of `cache_position`
        if layer_idx >= len(self.layers):
            return cache_position.shape[0], 0
        return self.layers[layer_idx].get_mask_sizes(cache_position)

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        """Returns maximum sequence length of the cache object. Dynamic caches do not have a maximum length."""
        # For DynamicCache, where the layers are created at runtime -> if it was not yet created, return -1
        # as DynamicLayer does
        if layer_idx >= len(self.layers):
            return -1
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
        # For DynamicCache dispatching the layers lazily (otherwise, all([]) is True)
        if len(self.layers) == 0:
            return False
        return all(layer.is_compileable for layer in self.layers)

    @property
    def is_initialized(self) -> bool:
        """Return whether the cache data is initialized"""
        return len(self.layers) > 0 and all(layer.is_initialized for layer in self.layers)

    @property
    def is_sliding(self) -> list[bool]:
        """Return whether the layers of the cache are sliding window"""
        return [getattr(layer, "is_sliding", False) for layer in self.layers]

    def __getitem__(self, layer_idx: int) -> tuple[ms.Tensor, ms.Tensor]:
        """
        Support for backwards-compatible `past_key_values` indexing, e.g. `past_key_values[0][0].shape[2]` to get the
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
        Support for backwards-compatible `past_key_values` iteration, e.g. `for x in past_key_values:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.layers[layer_idx].keys, self.layers[layer_idx].values)

    def __len__(self):
        """
        This value corresponds to the number of layers in the model.
        """
        # Note: for DynamicCache, layers are initialized lazily, so this will not be accurate before the first
        # forward through all the layers
        return len(self.layers)


class DynamicCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.
    It stores the key and value states as a list of `CacheLayer`, one for each layer. The expected shape for each tensor
    in the `CacheLayer`s is `[batch_size, num_heads, seq_len, head_dim]`.
    If a config is passed, it will additionally check for sliding or hybrid cache structure, greatly reducing the
    memory requirement of the cached tensors to `[batch_size, num_heads, min(seq_len, sliding_window), head_dim]`.

    See `Cache` for details on common methods that are implemented by all cache classes.

    Args:
        ddp_cache_data (`Iterable[tuple[ms.Tensor, ms.Tensor]]`, *optional*):
            It was originally added for compatibility with `torch.distributed` (DDP). In a nutshell, it is
            `map(gather_map, zip(*caches))`, i.e. each item in the iterable contains the key and value states
            for a layer gathered across replicas by torch.distributed (shape=[global batch size, num_heads, seq_len, head_dim]).
            Note: it needs to be the 1st arg as well to work correctly
        config (`PretrainedConfig`, *optional*):
            The config of the model for which this Cache will be used. If passed, it will be used to check for sliding
            or hybrid layer structure, greatly reducing the memory requirement of the cached tensors to
            `[batch_size, num_heads, min(seq_len, sliding_window), head_dim]`.
        offloading (`bool`, *optional*, defaults to `False`):
            Whether to perform offloading of the layers to `cpu`, to save GPU memory.
            Not support yet.
        offload_only_non_sliding (`bool`, *optional*, defaults to `False`):
            If `offloading` is `True`, this further decides if only the non-sliding layers will be offloaded (because
            usually the sliding layers are small in size, so there is no need to offload them, and skipping it is faster).
            Not support yet.

    Example:

    ```python
    >>> from transformers import AutoTokenizer
    >>> from mindone.transformers import AutoModelForCausalLM, DynamicCache

    >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
    >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

    >>> inputs = tokenizer(text="My name is Qwen2", return_tensors="np")
    >>> inputs = {k: ms.tensor(v) for k, v in inputs.items()}

    >>> # Prepare a cache class and pass it to model's forward
    >>> past_key_values = DynamicCache(config=model.config)
    >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
    >>> outputs.past_key_values # access cache filled with key/values from generation
    DynamicCache()
        ```
    """

    def __init__(
        self,
        ddp_cache_data: Optional[Iterable[tuple[ms.Tensor, ms.Tensor]]] = None,
        config: Optional[PretrainedConfig] = None,
        offloading: bool = False,  # Not support yet
        offload_only_non_sliding: bool = False,  # Not support yet
    ):
        layers = []
        # If a config is passed, use it to infer the layer types and initialize accordingly
        if config is not None:
            decoder_config = config.get_text_config(decoder=True)
            sliding_window = getattr(decoder_config, "sliding_window", None) or getattr(
                decoder_config, "attention_chunk_size", None
            )
            layer_types = getattr(decoder_config, "layer_types", None)
            if layer_types is None:
                layer_types = [
                    "sliding_attention" if sliding_window is not None else "full_attention"
                    for _ in range(decoder_config.num_hidden_layers)
                ]
            # Some models have shared layers thus no cache is needed for them (e.g. Gemma3n)
            if hasattr(decoder_config, "num_kv_shared_layers"):
                layer_types = layer_types[: -decoder_config.num_kv_shared_layers]

            for layer_type in layer_types:
                # From a cache point of view, both sliding and chunked are the same in how they should behave and how many
                # states they should return - only the mask changes to make them different at the end!
                if layer_type in ("sliding_attention", "chunked_attention"):
                    layers.append(DynamicSlidingWindowLayer(sliding_window=sliding_window))
                else:
                    layers.append(DynamicLayer())

        # In this case, use the passed data to already fill in the Cache
        if ddp_cache_data is not None:
            # Init all the layers with the data
            for layer_idx, (key_states, value_states) in enumerate(ddp_cache_data):
                # If the config was not passed above, initialize a DynamicLayer for each entry of the ddp_data
                if config is None:
                    layers.append(DynamicLayer())
                # Update the layer with the data
                _, _ = layers[layer_idx].update(key_states, value_states)

        # If neither of config nor ddp_data was passed, then simply lazy init a full cache of DynamicLayer
        if len(layers) == 0:
            super().__init__(
                layer_class_to_replicate=DynamicLayer,
                offloading=offloading,
                offload_only_non_sliding=offload_only_non_sliding,
            )
        else:
            super().__init__(layers=layers, offloading=offloading, offload_only_non_sliding=offload_only_non_sliding)

    def to_legacy_cache(self) -> tuple[tuple[ms.Tensor, ms.Tensor]]:
        """
        Converts the `Cache` instance into the its equivalent in the legacy cache format. Used for
        backward compatibility.
        """
        legacy_cache = ()
        for layer in self.layers:
            legacy_cache += ((layer.keys, layer.values),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values: tuple[tuple[ms.Tensor, ms.Tensor]]) -> "DynamicCache":
        """
        Converts a cache in the legacy cache format into an equivalent `Cache`. Used for
        backward compatibility.
        """
        cache = cls()
        if past_key_values is None:
            logger.warning_once("past_key_values should not be None in from_legacy_cache()")
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache


class StaticCache(Cache):
    """
    Static Cache class to be used with `mindspore.jit(model)`.

    See `Cache` for details on common methods that are implemented by all cache classes.

    Args:
        config (`PretrainedConfig`):
            The config of the model for which this Cache will be used. It will be used to check for sliding
            or hybrid layer structure, and initialize each layer accordingly.
        max_cache_len (`int`):
            The maximum number of tokens that this Cache should hold.
        offloading (`bool`, *optional*, defaults to `False`):
            Whether to perform offloading of the layers to `cpu`, to save GPU memory.
        offload_only_non_sliding (`bool`, *optional*, defaults to `True`):
            If `offloading` is `True`, this further decides if only the non-sliding layers will be offloaded (because
            usually the sliding layers are small in size, so there is no need to offload them, and skipping it is faster).

    Example:

    ```python
    >>> from transformers import AutoTokenizer
    >>> from mindone.transformers import AutoModelForCausalLM, StaticCache

    >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

    >>> inputs = tokenizer(text="My name is Llama", return_tensors="np")
    >>> inputs = {k: ms.tensor(v) for k, v in inputs.items()}

    >>> # Prepare a cache class and pass it to model's forward
    >>> # Leave empty space for 10 new tokens, which can be used when calling forward iteratively 10 times to generate
    >>> max_generated_length = inputs.input_ids.shape[1] + 10
    >>> past_key_values = StaticCache(config=model.config, max_cache_len=max_generated_length)
    >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
    >>> outputs.past_key_values # access cache filled with key/values from generation
    StaticCache()
    ```
    """

    # Pass-in kwargs as well to avoid crashing for BC (it used more arguments before)
    def __init__(
        self,
        config: PretrainedConfig,
        max_cache_len: int,
        offloading: bool = False,
        offload_only_non_sliding: bool = True,
        **kwargs,
    ):
        config = config.get_text_config(decoder=True)
        layer_types = getattr(config, "layer_types", None)
        # If `layer_types` is not explicitly provided, infer if the model is fully sliding
        if layer_types is None:
            if getattr(config, "sliding_window", None) is not None:
                layer_types = ["sliding_attention" for _ in range(config.num_hidden_layers)]
            elif getattr(config, "attention_chunk_size", None) is not None:
                layer_types = ["chunked_attention" for _ in range(config.num_hidden_layers)]
            else:
                layer_types = ["full_attention" for _ in range(config.num_hidden_layers)]
        # Some models have shared layers thus no cache is needed for them (e.g. Gemma3n)
        if hasattr(config, "num_kv_shared_layers"):
            layer_types = layer_types[: -config.num_kv_shared_layers]

        layers = []
        for layer_type in layer_types:
            if layer_type == "sliding_attention":
                layer = StaticSlidingWindowLayer(max_cache_len=max_cache_len, sliding_window=config.sliding_window)
            elif layer_type == "chunked_attention":
                # From a cache point of view, both sliding and chunked are the same in how they should behave and how many
                # states they should return - only the mask changes to make them different at the end!
                layer = StaticSlidingWindowLayer(
                    max_cache_len=max_cache_len, sliding_window=config.attention_chunk_size
                )
            else:
                layer = StaticLayer(max_cache_len=max_cache_len)
            layers.append(layer)

        super().__init__(layers=layers, offloading=offloading, offload_only_non_sliding=offload_only_non_sliding)


class QuantizedCache(Cache):
    """
    A quantizer cache similar to what is described in the
    [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache paper](https://huggingface.co/papers/2402.02750).
    It allows the model to generate longer sequence length without allocating too much memory for keys and values
    by applying quantization.
    The cache has two types of storage, one for original precision and one for the
    quantized cache. A `residual length` is set as a maximum capacity for the original precision cache. When the
    length goes beyond maximum capacity, the original precision cache is discarded and moved into the quantized cache.
    The quantization is done per-channel with a set `q_group_size` for both keys and values, in contrast to what was
    described in the paper.

    See `Cache` for details on common methods that are implemented by all cache classes.

    Args:
        backend (`str`):
            The quantization backend to use. One of `("quanto", "hqq").
        config (`PretrainedConfig`):
            The config of the model for which this Cache will be used.
        nbits (`int`, *optional*, defaults to 4):
            The number of bits for quantization.
        axis_key (`int`, *optional*, defaults to 0):
            The axis on which to quantize the keys.
        axis_value (`int`, *optional*, defaults to 0):
            The axis on which to quantize the values.
        q_group_size (`int`, *optional*, defaults to 64):
            Quantization is done per-channel according to a set `q_group_size` for both keys and values.
        residual_length (`int`, *optional*, defaults to 128):
            Maximum capacity for the original precision cache
    """

    def __init__(
        self,
        backend: str,
        config: PretrainedConfig,
        nbits: int = 4,
        axis_key: int = 0,
        axis_value: int = 0,
        q_group_size: int = 64,
        residual_length: int = 128,
    ):
        raise NotImplementedError


class EncoderDecoderCache(Cache):
    """
    Base, abstract class for all encoder-decoder caches. Can be used to hold combinations of self-attention and
    cross-attention caches.

    See `Cache` for details on common methods that are implemented by all cache classes.

    Args:
        caches (`Iterable`):
            Usually an iterable of length 2, containing 2 `Cache` objects, the first one for self-attention, the
            second one for cross-attention. Can optionally also be an iterable of length 1, containing a
            `tuple[tuple[ms.Tensor]]` (usually used for compatibility with torch dp and ddp).

    Example:

    ```python
    >>> from mindone.transformers import AutoProcessor, AutoModelForCausalLM, DynamicCache, EncoderDecoderCache

    >>> model = AutoModelForCausalLM.from_pretrained("openai/whisper-small")
    >>> processor = AutoProcessor.from_pretrained("openai/whisper-small")

    >>> inputs = processor(audio=YOUR-AUDIO, return_tensors="np")
    >>> inputs = {k: ms.tensor(v) for k, v in inputs.items()}

    >>> # Prepare cache classes for encoder and decoder and pass it to model's forward
    >>> self_attention_cache = DynamicCache(config=self.config)
    >>> cross_attention_cache = DynamicCache(config=self.config)
    >>> past_key_values = EncoderDecoderCache(self_attention_cache, cross_attention_cache)
    >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
    >>> outputs.past_key_values # access cache filled with key/values from generation
    EncoderDecoderCache()
    ```
    """

    def __init__(self, *caches) -> None:
        # For dp and ddp support, if only one argument is passed, it should be an iterable of tuples of tensors
        if len(caches) == 1:
            self.self_attention_cache = DynamicCache()
            self.cross_attention_cache = DynamicCache()
            # Populate cache from the iterable
            for layer_idx, key_value_states in enumerate(caches[0]):
                key_states, value_states = key_value_states[:2]
                self.self_attention_cache.update(key_states, value_states, layer_idx)
                if len(key_value_states) > 2:
                    key_states, value_states = key_value_states[2:]
                    self.cross_attention_cache.update(key_states, value_states, layer_idx)
        # Otherwise, we should get two arguments, a self-attention cache and a cross-attention cache
        elif len(caches) == 2:
            if not isinstance(caches[0], Cache) or not isinstance(caches[1], Cache):
                raise TypeError(f"One of the two arguments is not a Cache: {type(caches[0]) = }, {type(caches[1]) = }")
            self.self_attention_cache = caches[0]
            self.cross_attention_cache = caches[1]
        # Error case
        else:
            raise ValueError(f"Expected 1 or 2 arguments, got {len(caches)}")

        self.is_updated = {}
        for layer_idx in range(len(self.cross_attention_cache)):
            self.is_updated[layer_idx] = bool(self.cross_attention_cache.get_seq_length(layer_idx) > 0)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(self_attention_cache={self.self_attention_cache}, cross_attention_cache="
            f"{self.cross_attention_cache})"
        )

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_values` iteration, e.g. `for x in past_key_values:` to iterate over
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
        Support for backwards-compatible `past_key_values` indexing, e.g. `past_key_values[0][0].shape[2]` to get the
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
        Support for backwards-compatible `past_key_values` length, e.g. `len(past_key_values)`. This value corresponds
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
        cache = cls(DynamicCache(), DynamicCache())
        if past_key_values is None:
            logger.warning_once("past_key_values should not be None in from_legacy_cache()")
        else:
            for layer_idx, key_value_states in enumerate(past_key_values):
                key_states, value_states = key_value_states[:2]
                cache.self_attention_cache.update(key_states, value_states, layer_idx)
                if len(key_value_states) > 2:
                    key_states, value_states = key_value_states[2:]
                    cache.cross_attention_cache.update(key_states, value_states, layer_idx)
                    cache.is_updated[layer_idx] = True
        return cache

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # check if empty list because in case of static cache it will be a tensors and we can't check `if not ms.Tensor`
        return self.self_attention_cache.get_seq_length(layer_idx)

    def reset(self):
        self.self_attention_cache.reset()
        self.cross_attention_cache.reset()
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
        negative to remove `maximum_length` tokens. This is used in assisted decoding and contrastive search (on the Hub).
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
        """Repeat the cache `repeats` times in the batch dimension. Used in contrastive search (on the Hub)."""
        self.check_dynamic_cache(self.batch_repeat_interleave.__name__)
        self.self_attention_cache.batch_repeat_interleave(repeats)
        self.cross_attention_cache.batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices: ms.Tensor):
        """Only keep the `indices` in the batch dimension of the cache. Used in contrastive search (on the Hub)."""
        self.check_dynamic_cache(self.batch_select_indices.__name__)
        self.self_attention_cache.batch_select_indices(indices)
        self.cross_attention_cache.batch_select_indices(indices)

    def get_max_cache_shape(self) -> int:
        """Returns the maximum sequence length (i.e. max capacity) of the cache object"""
        return self.self_attention_cache.get_max_cache_shape()

    def get_mask_sizes(self, cache_position: ms.Tensor, layer_idx: int) -> tuple[int, int]:
        return self.self_attention_cache.get_mask_sizes(cache_position, layer_idx)

    @property
    def is_sliding(self):
        return self.self_attention_cache.is_sliding

    @property
    def is_compileable(self) -> bool:
        return self.self_attention_cache.is_compileable


# transfomrers v4.57.1 - Deprecated classes


class SlidingWindowLayer(StaticSlidingWindowLayer):
    def __init__(self, max_cache_len: int, sliding_window: int):
        logger.warning_once(
            "`SlidingWindowLayer` is deprecated and will be removed in version v4.59 "
            "Use `StaticSlidingWindowLayer` instead, which is a better name for it."
        )
        super().__init__(max_cache_len, sliding_window)


class ChunkedSlidingLayer(StaticSlidingWindowLayer):
    def __init__(self, max_cache_len: int, sliding_window: int):
        logger.warning_once(
            "`ChunkedSlidingLayer` is deprecated and will be removed in version v4.59 "
            "Use `StaticSlidingWindowLayer` instead, which has the exact same functionalities."
        )
        super().__init__(max_cache_len, sliding_window)


class OffloadedCache(DynamicCache):
    def __init__(self) -> None:
        logger.warning_once(
            "`OffloadedCache` is deprecated and will be removed in version v4.59 "
            "Use `DynamicCache(offloading=True)` instead"
        )
        super().__init__(offloading=True)
        raise NotImplementedError


class OffloadedStaticCache(StaticCache):
    def __init__(self, config: PretrainedConfig, max_cache_len: int, *args, **kwargs):
        logger.warning_once(
            "`OffloadedStaticCache` is deprecated and will be removed in version v4.59 "
            "Use `StaticCache(..., offloading=True)` instead"
        )
        super().__init__(config=config, max_cache_len=max_cache_len, offloading=True)
        raise NotImplementedError


class SlidingWindowCache(StaticCache):
    def __init__(self, config: PretrainedConfig, max_cache_len: int, *args, **kwargs):
        logger.warning_once(
            "`SlidingWindowCache` is deprecated and will be removed in version v4.59 "
            "Use `StaticCache(...)` instead which will correctly infer the type of each layer."
        )
        super().__init__(config=config, max_cache_len=max_cache_len)


class HybridCache(StaticCache):
    def __init__(self, config: PretrainedConfig, max_cache_len: int, *args, **kwargs):
        logger.warning_once(
            "`HybridCache` is deprecated and will be removed in version v4.59 "
            "Use `StaticCache(...)` instead which will correctly infer the type of each layer."
        )
        super().__init__(config=config, max_cache_len=max_cache_len)


class HybridChunkedCache(StaticCache):
    def __init__(self, config: PretrainedConfig, max_cache_len: int, *args, **kwargs):
        logger.warning_once(
            "`HybridChunkedCache` is deprecated and will be removed in version v4.59 "
            "Use `StaticCache(...)` instead which will correctly infer the type of each layer."
        )
        super().__init__(config=config, max_cache_len=max_cache_len)


class OffloadedHybridCache(StaticCache):
    def __init__(self, config: PretrainedConfig, max_cache_len: int, *args, **kwargs):
        logger.warning_once(
            "`OffloadedHybridCache` is deprecated and will be removed in version v4.59 "
            "Use `StaticCache(..., offload=True)` instead which will correctly infer the type of each layer."
        )
        super().__init__(config=config, max_cache_len=max_cache_len, offloading=True)
        raise NotImplementedError


class QuantoQuantizedCache(QuantizedCache):
    def __init__(
        self,
        config: PretrainedConfig,
        nbits: int = 4,
        axis_key: int = 0,
        axis_value: int = 0,
        q_group_size: int = 64,
        residual_length: int = 128,
    ):
        logger.warning_once(
            "`QuantoQuantizedCache` is deprecated and will be removed in version v4.59 "
            "Use `QuantizedCache(backend='quanto', ...)` instead."
        )
        super().__init__("quanto", config, nbits, axis_key, axis_value, q_group_size, residual_length)
        raise NotImplementedError


class HQQQuantizedCache(QuantizedCache):
    def __init__(
        self,
        config: PretrainedConfig,
        nbits: int = 4,
        axis_key: int = 0,
        axis_value: int = 0,
        q_group_size: int = 64,
        residual_length: int = 128,
    ):
        logger.warning_once(
            "`HQQQuantizedCache` is deprecated and will be removed in version v4.59 "
            "Use `QuantizedCache(backend='hqq', ...)` instead."
        )
        super().__init__("hqq", config, nbits, axis_key, axis_value, q_group_size, residual_length)
        raise NotImplementedError


class SinkCache(Cache):
    """
    It is now a `custom_generate` repository on the Hub: https://huggingface.co/transformers-community/sink_cache.
    See [these docs](https://huggingface.co/docs/transformers/generation_strategies#custom-decoding-methods) for
    general `custom_generate`usage.
    """

    # TODO (joao, manuel): Remove this class in v4.59.0
    def __init__(self, **kwargs) -> None:
        raise NotImplementedError(
            "`SinkCache` has been moved as a `custom_generate` repository on the Hub: "
            "https://huggingface.co/transformers-community/sink_cache. See the repository for usage examples."
        )


# TODO: should be deprecated now by v4.57.1
class MambaCache:
    """
    Importing `MambaCache` from `transformers.cache_utils` is deprecated and will be removed
    in a future version. Please import it from `transformers` or `transformers.models.mamba.cache_mamba` instead.

    Cache for mamba model which does not have attention mechanism and key value states.

    Arguments:
        config (`PretrainedConfig):
            The configuration file defining the shape-related attributes required to initialize the static cache.
        max_batch_size (`int`):
            The maximum batch size with which the model will be used. Note that a new instance must be instantiated if a smaller batch size is used.
        dtype (`ms.Type`, *optional*, defaults to `ms.float16`):
            The default `dtype` to use when initializing the layer.

    Example:

        ```python
        >>> from transformers import AutoTokenizer, MambaForCausalLM, MambaCache

        >>> model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")

        >>> inputs = tokenizer(text="My name is Mamba", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> past_key_values = MambaCache(config=model.config, max_batch_size=1, dtype=model.dtype)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values
        MambaCache()
        ```
    """

    is_compileable = True

    # TODO (joao): add layer_device_map arg and update code in `generate` accordingly
    def __init__(
        self,
        config,
        max_batch_size: int,
        dtype: ms.Type = ms.float16,
    ):
        self.max_batch_size = max_batch_size
        self._dtype = dtype
        self.intermediate_size = config.intermediate_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel

        self.conv_states: list[ms.Tensor] = []
        self.ssm_states: list[ms.Tensor] = []
        for _ in range(config.num_hidden_layers):
            conv_state: ms.Tensor = mint.zeros(
                (self.max_batch_size, self.intermediate_size, self.conv_kernel_size),
                dtype=self._dtype,
            )
            ssm_state: ms.Tensor = mint.zeros(
                (self.max_batch_size, self.intermediate_size, self.ssm_state_size),
                dtype=self._dtype,
            )

            self.conv_states.append(conv_state)
            self.ssm_states.append(ssm_state)

    def update_conv_state(self, layer_idx: int, new_conv_state: ms.Tensor, cache_position: ms.Tensor) -> ms.Tensor:
        conv_state = self.conv_states[layer_idx]
        cache_position = cache_position.clamp(0, self.conv_kernel_size - 1)

        conv_state = conv_state.roll(shifts=-1, dims=-1)
        conv_state[:, :, cache_position] = new_conv_state.to(dtype=conv_state.dtype)
        self.conv_states[layer_idx].zero_()
        self.conv_states[layer_idx] += conv_state
        return self.conv_states[layer_idx]

    def update_ssm_state(self, layer_idx: int, new_ssm_state: ms.Tensor):
        self.ssm_states[layer_idx] = new_ssm_state
        return self.ssm_states[layer_idx]

    def reset(self):
        for layer_idx in range(len(self.conv_states)):
            # In-place ops prevent breaking the static address
            self.conv_states[layer_idx].zero_()
            self.ssm_states[layer_idx].zero_()
