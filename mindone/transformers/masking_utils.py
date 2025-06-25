# coding=utf-8
# Copyright 2025 HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Callable, Optional, Union

from transformers.configuration_utils import PretrainedConfig

import mindspore as ms
from mindspore import mint

from .cache_utils import Cache
from .modeling_attn_mask_utils import dtype_to_min
from .utils.generic import GeneralInterface

BlockMask = ms.Tensor


def and_masks(*mask_functions: list[Callable]) -> Callable:
    """Returns a mask function that is the intersection of provided mask functions"""
    if not all(callable(arg) for arg in mask_functions):
        raise RuntimeError(f"All inputs should be callable mask_functions: {mask_functions}")

    def and_mask(batch_idx, head_idx, q_idx, kv_idx):
        result = q_idx.new_ones((), dtype=ms.bool_)
        for mask in mask_functions:
            result = result & mask(batch_idx, head_idx, q_idx, kv_idx)
        return result

    return and_mask


def or_masks(*mask_functions: list[Callable]) -> Callable:
    """Returns a mask function that is the union of provided mask functions"""
    if not all(callable(arg) for arg in mask_functions):
        raise RuntimeError(f"All inputs should be callable mask_functions: {mask_functions}")

    def or_mask(batch_idx, head_idx, q_idx, kv_idx):
        result = q_idx.new_zeros((), dtype=ms.bool_)
        for mask in mask_functions:
            result = result | mask(batch_idx, head_idx, q_idx, kv_idx)
        return result

    return or_mask


def causal_mask_function(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    """
    This creates a basic lower-diagonal causal mask.
    """
    return kv_idx <= q_idx


def _vmap_for_bhqkv(mask_function: Callable, bh_indices: bool = True) -> Callable:
    """
    Used to vmap our mask_functions over the q_idx and kv_idx dimensions of the inputs. Optionally, vmap over
    the batch and head indices as well if `bh_indices=True`.
    Using vmap here allows us to keep the performance of vectorized ops, while having a single set of primitive
    functions between attention interfaces (i.e. between flex and sdpa/eager, FA2 being a bit different).

    Args:
        mask_function (`Callable`):
            The mask_function to vmap.
        bh_indices (`bool`, optional):
            Whether to vmap over the batch and head indices as well, or only q and kv indices.

    Returns:
        Callable: The vmapped function.
    """
    # We vmap the function 2 times, broadcasting the [q_idx, kv_idx] dimensions
    dimensions = [(None, None, None, 0), (None, None, 0, None)]
    if bh_indices:
        # We extend broadcasting over the [batch_idx, head_idx] dimensions
        dimensions.extend([(None, 0, None, None), (0, None, None, None)])

    for dims in dimensions:
        mask_function = ms.vmap(mask_function, in_axes=dims, out_axes=0)
    return mask_function


def prepare_padding_mask(
    attention_mask: Optional[ms.Tensor], kv_length: int, kv_offset: int, _slice: bool = True
) -> Optional[ms.Tensor]:
    """
    From the 2D attention mask, prepare the correct padding mask to use by potentially padding it, and slicing
    according to the `kv_offset` if `_slice` is `True`.
    """
    local_padding_mask = attention_mask
    if attention_mask is not None:
        # Pad it if necesary
        if (padding_length := kv_length + kv_offset - attention_mask.shape[-1]) > 0:
            local_padding_mask = mint.nn.functional.pad(attention_mask, (0, padding_length))
        # For flex, we should not slice them, only use an offset
        if _slice:
            # Equivalent to: `local_padding_mask = attention_mask[:, kv_offset : kv_offset + kv_length]`,
            # but without data-dependent slicing (i.e. torch.compile friendly)
            mask_indices = mint.arange(kv_length)
            mask_indices += kv_offset
            local_padding_mask = local_padding_mask[:, mask_indices]
    return local_padding_mask


def sdpa_mask_older_torch(
    batch_size: int,
    cache_position: ms.Tensor,
    kv_length: int,
    kv_offset: int = 0,
    mask_function: Callable = causal_mask_function,
    attention_mask: Optional[ms.Tensor] = None,
    local_size: Optional[int] = None,
    allow_is_causal_skip: bool = True,
    allow_torch_fix: bool = True,
    **kwargs,
) -> Optional[ms.Tensor]:
    """
    NOTE: This function is only used when torch version is torch<2.5 - see `sdpa_mask_recent_torch` otherwise.

    Create a 4D boolean mask of shape `(batch_size, 1, query_length, kv_length)` where a value of True indicates that
    the element should take part in the attention computation, and False that it should not.
    If `allow_torch_fix=True` (the default), rows corresponding to query tokens that do not attend
    to any other tokens (due to padding) will be fully attended to instead, in order to avoid `nan` propagation (this does
    not change the final result).

    Args:
        batch_size (`int`):
            The batch size of the input sequence.
        cache_position (`torch.Tensor`):
            A tensor of shape (query_length,) indicating the current indices of the input sequence elements.
        kv_length (`int`):
            The size that the key and value states will have during the attention computation.
        kv_offset (`int`, optional):
            An optional offset to indicate at which first position the key and values states will refer to.
        mask_function (`Callable`):
            The mask factory function describing the mask pattern.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length)
        local_size (`int`, optional):
            The size of the local attention, if we do not use full attention. This is used only if `allow_is_causal_skip=True`
            to try to skip mask creation if possible.
        allow_is_causal_skip (`bool`, optional):
            Whether to allow to return `None` for the mask under conditions where we can use the `is_causal` argument in
            `torch.sdpa` instead. Default to `True`.
        allow_torch_fix (`bool`, optional):
            Whether to update the mask in case a query is not attending to any tokens, to solve a bug in torch's older
            versions. We need an arg to skip it when using eager. By default `True`.
    """
    q_length = cache_position.shape[0]
    # Potentially pad the 2D mask, and slice it correctly
    padding_mask = prepare_padding_mask(attention_mask, kv_length, kv_offset)

    # Under specific conditions, we can avoid materializing the mask, instead relying on the `is_causal` argument
    if allow_is_causal_skip and _ignore_causal_mask_sdpa(padding_mask, q_length, kv_length, kv_offset, local_size):
        return None

    # Similar to `kv_arange = torch.arange(start=kv_offset, end=kv_offset + kv_length, device=cache_position.device)`
    # but without data-dependent slicing (i.e. torch.compile friendly)
    kv_arange = mint.arange(kv_length)
    kv_arange += kv_offset

    # This creates the 4D mask easily. Note that we do not include vmap over the batch_idx dimension as well,
    # as vmap cannot handle slicing a tensor from scalar tensor (it internally calls `.item()` which vmap does not allow
    # However, in more recent version of Pytorch, a trick was introduced to handle it - which is the reason we have
    # `sdpa_mask_recent_torch`, as it allows more general `mask_function`
    causal_mask = _vmap_for_bhqkv(mask_function, bh_indices=False)(None, None, cache_position, kv_arange)
    causal_mask = causal_mask[None, None, :, :].broadcast_to((batch_size, -1, -1, -1))
    if padding_mask is not None:
        causal_mask = causal_mask * padding_mask[:, None, None, :]

    # # Due to a bug in versions of torch<2.5, we need to update the mask in case a query is not attending to any
    # # tokens (due to padding). See details in https://github.com/pytorch/pytorch/issues/110213
    # if not _is_torch_greater_or_equal_than_2_5 and allow_torch_fix:
    #     causal_mask |= torch.all(~causal_mask, dim=-1, keepdim=True)
    return causal_mask


def _ignore_causal_mask_sdpa(
    padding_mask: Optional[ms.Tensor],
    query_length: int,
    kv_length: int,
    kv_offset: int,
    local_attention_size: Optional[int] = None,
) -> bool:
    """
    Detects whether the causal mask can be ignored in case PyTorch's SDPA is used, rather relying on SDPA's `is_causal` argument.

    In case no token is masked in the 2D `padding_mask` argument, if `query_length == 1` or
    `key_value_length == query_length`, we rather rely on SDPA `is_causal` argument to use causal/non-causal masks,
    allowing to dispatch to the flash attention kernel (that can otherwise not be used if a custom `attn_mask` is
    passed).
    """
    is_tracing = False
    if padding_mask is not None and padding_mask.shape[-1] > kv_length:
        mask_indices = mint.arange(kv_length)
        mask_indices += kv_offset
        padding_mask = padding_mask[:, mask_indices]

    # When using `torch.export` or `torch.onnx.dynamo_export`, we must pass an example input, and `is_causal` behavior is
    # hard-coded to the forward. If a user exports a model with query_length > 1, the exported model will hard-code `is_causal=True`
    # which is in general wrong (see https://github.com/pytorch/pytorch/issues/108108). Thus, we only set
    # `ignore_causal_mask = True` if we are not tracing
    if (
        not is_tracing
        # only cases when lower and upper diags are the same, see https://github.com/pytorch/pytorch/issues/108108
        and (query_length == 1 or kv_length == query_length)
        # in this case we need to add special patterns to the mask so cannot be skipped otherwise
        and (local_attention_size is None or kv_length < local_attention_size)
        # In this case, we need to add padding to the mask, so cannot be skipped otherwise
        and (padding_mask is None or padding_mask.all())
    ):
        return True

    return False


# We use the version with newer torch whenever possible, as it is more general and can handle arbitrary mask functions
# (especially mask_function indexing a tensor, such as the padding mask function)
sdpa_mask = sdpa_mask_older_torch


def eager_mask(
    batch_size: int,
    cache_position: ms.Tensor,
    kv_length: int,
    kv_offset: int = 0,
    mask_function: Callable = causal_mask_function,
    attention_mask: Optional[ms.Tensor] = None,
    dtype: ms.Type = ms.float32,
    **kwargs,
) -> ms.Tensor:
    """
    Create a 4D float mask of shape `(batch_size, 1, query_length, kv_length)` where a value of 0 indicates that
    the element should take part in the attention computation, and -inf (minimum value for the given `dtype`) that
    it should not.

    Args:
        batch_size (`int`):
            The batch size of the input sequence.
        cache_position (`torch.Tensor`):
            A tensor of shape (query_length,) indicating the current indices of the input sequence elements.
        kv_length (`int`):
            The size that the key and value states will have during the attention computation.
        kv_offset (`int`, optional):
            An optional offset to indicate at which first position the key and values states will refer to.
        mask_function (`Callable`):
            The mask factory function describing the mask pattern.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length)
        dtype (`torch.dtype`, optional):
            The dtype to use for the mask. By default, `torch.float32`.
    """
    # The masks for eager attention are simply boolean mask from sdpa, casted to 0 and -inf
    _ = kwargs.pop("allow_is_causal_skip", None)
    mask = sdpa_mask(
        batch_size=batch_size,
        cache_position=cache_position,
        kv_length=kv_length,
        kv_offset=kv_offset,
        mask_function=mask_function,
        attention_mask=attention_mask,
        allow_is_causal_skip=False,
        allow_torch_fix=False,
        **kwargs,
    )
    min_dtype = dtype_to_min(dtype)
    # we need 0s where the tokens should be taken into account, and -inf otherwise (mask is already of boolean type)
    mask = mint.where(mask, ms.tensor(0.0, dtype=dtype), min_dtype)
    return mask


class AttentionMaskInterface(GeneralInterface):
    # Class instance object, so that a call to `register` can be reflected into all other files correctly, even if
    # a new instance is created (in order to locally override a given function)
    _global_mapping = {
        "eager": eager_mask,
        "flash_attention_2": eager_mask,
    }


# Global AttentionMaskInterface shared by all models which do not need to overwrite any of the existing ones
ALL_MASK_ATTENTION_FUNCTIONS: AttentionMaskInterface = AttentionMaskInterface()


def _preprocess_mask_arguments(
    config: PretrainedConfig,
    input_embeds: ms.Tensor,
    attention_mask: Optional[Union[ms.Tensor, BlockMask]],
    cache_position: ms.Tensor,
    past_key_values: Optional[Cache],
    layer_idx: Optional[int],
) -> tuple[bool, Optional[Union[ms.Tensor, BlockMask]], int, int]:
    """
    Perform some common pre-processing of the mask arguments we get from the modeling code. Mostly determine the
    key-value length and offsets, and if we should early exit or not.

    Args:
        config (`PretrainedConfig`):
            The model config.
        input_embeds (`torch.Tensor`):
            The input embeddings of shape (batch_size, query_length, hidden_dim). This is used only to infer the
            batch size, query length and dtype.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length).
            It can also be an already prepared 4D mask, in which case it is returned as-is.
        cache_position (`torch.Tensor`):
            A tensor of shape (query_length,) indicating the current indices of the input sequence elements.
        past_key_values (`Cache`, optional):
            The past key values, if we use a cache.
        layer_idx (`int`, optional):
            If `past_key_values` is not None, this is the layer index of the cache from which to get the key-value
            length and offset. Indeed, for hybrid caches, different layers may return different lengths.

    Returns:
        early_exit (`bool`):
            Whether we should early exit mask creation, and return the mask as-is.
        attention_mask (`torch.Tensor` or `BlockMask` or `None`):
            The attention mask to either return immediately, or to use in downstream mask creation.
        kv_length (`int`):
            The size that the key and value states will have during the attention computation.
        kv_offset (`int`):
            An offset to indicate at which first position the key and values states will refer to.
    """
    # If the mask is already 4D, simply return as-is (it was already prepared, or it is custom)
    if isinstance(attention_mask, (ms.Tensor, BlockMask)) and len(attention_mask.shape) == 4:
        return True, attention_mask, None, None

    # For TGI/vLLM backends, or other custom attention without equivalent mask creation: we don't need a mask!
    # Note: it's not ideal to check the `_global_mapping` attribute instead of the object itself, however otherwise
    # full graph dynamo tracing (i.e. torch.export or compile with `fullgraph=True`) will fail on Python<3.11
    # with `torch._dynamo.exc.Unsupported: 'inline in skipfiles:Mapping.__contains__ | __contains__, skipped
    # according trace_rules.lookup SKIP_DIRS'` -- can be removed when we require Python>=3.11
    if config._attn_implementation not in ALL_MASK_ATTENTION_FUNCTIONS._global_mapping:
        return True, None, None, None

    # Move the mask to correct device, and potentially switch dtype for efficiency
    if attention_mask is not None and attention_mask.ndim == 2:
        attention_mask = attention_mask.to(dtype=ms.bool_)

    # If using a cache, it can give all informations about mask sizes based on seen tokens
    if past_key_values is not None:
        kv_length, kv_offset = past_key_values.get_mask_sizes(cache_position, layer_idx)
    # Otherwise, the sizes are simply the input sizes
    else:
        kv_length, kv_offset = input_embeds.shape[1], 0

    return False, attention_mask, kv_length, kv_offset


def create_causal_mask(
    config: PretrainedConfig,
    input_embeds: ms.Tensor,
    attention_mask: Optional[ms.Tensor],
    cache_position: ms.Tensor,
    past_key_values: Optional[Cache],
    or_mask_function: Optional[Callable] = None,
    and_mask_function: Optional[Callable] = None,
) -> Optional[Union[ms.Tensor, BlockMask]]:
    """
    Create a standard causal mask based on the attention implementation used (stored in the config). If `past_key_values`
    has an HybridCache structure, this function will return the mask corresponding to one of the "full_attention" layers (to align
    to what is needed in the `modeling_xxx.py` files).

    Args:
        config (`PretrainedConfig`):
            The model config.
        input_embeds (`torch.Tensor`):
            The input embeddings of shape (batch_size, query_length, hidden_dim). This is used only to infer the
            batch size, query length and dtype.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length).
            It can also be an already prepared 4D mask, in which case it is returned as-is.
        cache_position (`torch.Tensor`):
            A tensor of shape (query_length,) indicating the current indices of the input sequence elements.
        past_key_values (`Cache`, optional):
            The past key values, if we use a cache.
        or_mask_function (`Callable`, optional):
            An optional mask function to combine with the causal mask function (by doing the union of both). This is
            useful to easily overlay another mask on top of the causal one, for example for image tokens handling.
        and_mask_function (`Callable`, optional):
            An optional mask function to combine with the causal mask function (by doing the intersection of both). This is
            useful to easily overlay another mask on top of the causal one, for example for image tokens handling.
    """
    # If we have an HybridCache structure, here we want to create the mask for the full layers
    if hasattr(past_key_values, "is_sliding") and False in past_key_values.is_sliding:
        layer_idx = past_key_values.is_sliding.index(False)
    else:
        layer_idx = 0

    early_exit, attention_mask, kv_length, kv_offset = _preprocess_mask_arguments(
        config, input_embeds, attention_mask, cache_position, past_key_values, layer_idx
    )
    if early_exit:
        return attention_mask

    batch_size, dtype = input_embeds.shape[0], input_embeds.dtype
    mask_factory_function = causal_mask_function
    mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[config._attn_implementation]

    # Do not allow skip if we are compiling (this is to match BC)
    # TODO: cyril -> probably revisit and remove this, but a lot of tests rely on it
    allow_is_causal_skip = not past_key_values.is_compileable if past_key_values is not None else True

    # Allow slight deviations from causal mask
    if or_mask_function is not None:
        mask_factory_function = or_masks(mask_factory_function, or_mask_function)
        allow_is_causal_skip = False
    if and_mask_function is not None:
        mask_factory_function = and_masks(mask_factory_function, and_mask_function)
        allow_is_causal_skip = False

    # We now create the mask
    causal_mask = mask_interface(
        batch_size=batch_size,
        cache_position=cache_position,
        kv_length=kv_length,
        kv_offset=kv_offset,
        mask_function=mask_factory_function,
        attention_mask=attention_mask,
        allow_is_causal_skip=allow_is_causal_skip,  # additional kwarg for sdpa
        dtype=dtype,  # Additional kwarg for eager
        config=config,  # Pass the config as well, in case someone wants to easily have their own mask_interface
    )
    return causal_mask


LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING = {
    "full_attention": create_causal_mask,
}
