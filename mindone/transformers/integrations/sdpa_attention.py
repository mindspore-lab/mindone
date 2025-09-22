"""Adapted from https://github.com/huggingface/transformers/tree/main/src/transformers/integrations/sdpa_attention.py."""
from math import sqrt
from typing import Optional

import mindspore as ms
from mindspore import mint, nn, ops

from ..utils import logging

logger = logging.get_logger(__name__)


def repeat_kv(hidden_states: ms.Tensor, n_rep: int) -> ms.Tensor:
    """
    This is the equivalent of mint.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape  # BNSD format
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand((batch, num_key_value_heads, n_rep, slen, head_dim))
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False
) -> ms.Tensor:
    L, S = query.shape[-2], key.shape[-2]
    scale_factor = 1 / sqrt(query.shape[-1]) if scale is None else scale
    attn_bias = mint.zeros((L, S), dtype=query.dtype)
    if is_causal:
        if attn_mask is not None:
            if attn_mask.dtype == ms.bool_:
                attn_mask = mint.logical_and(
                    attn_mask, mint.ones((query.shape[-2], key.shape[-2]), dtype=ms.bool_).tril(diagonal=0)
                )
            else:
                attn_mask = attn_mask + mint.triu(mint.full((L, S), float("-inf"), dtype=attn_mask.dtype), diagonal=1)
        else:
            temp_mask = mint.ones((L, S), dtype=ms.bool_).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias = attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == ms.bool_:
            attn_bias = attn_bias.masked_fill(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.shape[-3] // key.shape[-3], -3)
        value = value.repeat_interleave(query.shape[-3] // value.shape[-3], -3)

    attn_weight = mint.matmul(query, key.swapaxes(-2, -1)) * scale_factor
    attn_weight += attn_bias

    # Identify rows where all elements are -inf (caused by causal mask or padding)
    row_is_all_inf = mint.isinf(attn_weight).all(axis=-1, keep_dims=True)

    # Before softmax: replace full -inf rows with zeros to avoid NaN
    attn_weight = mint.where(row_is_all_inf, mint.zeros_like(attn_weight), attn_weight)
    attn_weight = mint.softmax(attn_weight, dim=-1)

    # After softmax: set full -inf rows to zeros
    attn_weight = mint.where(row_is_all_inf, mint.zeros_like(attn_weight), attn_weight)

    attn_weight = ops.dropout(attn_weight, dropout_p, training=True)
    return mint.matmul(attn_weight, value)


def sdpa_attention_forward(
    module: nn.Cell,
    query: ms.Tensor,
    key: ms.Tensor,
    value: ms.Tensor,
    attention_mask: Optional[ms.Tensor] = None,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> tuple[ms.Tensor, None]:
    if kwargs.get("output_attentions", False) or kwargs.get("head_mask", None) is not None:
        logger.warning_once(
            "`sdpa` attention does not support `output_attentions=True` or `head_mask`."
            " Please set your attention to `eager` if you want any of these features."
        )

    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    # Note that it is important to check first for the shape, otherwise compile will fail with `argument 'is_causal' must be bool, not SymBool`
    if is_causal is None:
        # The last condition is for encoder (decoder) models which specify this by passing their own `is_causal` flag
        # This is mainly due to those models having mixed implementations for encoder, decoder, and encoder-decoder attns
        is_causal = query.shape[2] > 1 and attention_mask is None and getattr(module, "is_causal", True)

    attn_output = scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
    )
    attn_output = mint.transpose(attn_output, 1, 2).contiguous()

    return attn_output, None
