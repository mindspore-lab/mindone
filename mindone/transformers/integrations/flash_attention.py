"""Adapted from https://github.com/huggingface/transformers/tree/main/src/transformers/integrations/flash_attention.py."""

from typing import Optional

from transformers.utils import logging

import mindspore as ms
from mindspore import mint, nn, ops

logger = logging.get_logger(__name__)


def flash_attention_forward(
    module: nn.Cell,
    query: ms.Tensor,
    key: ms.Tensor,
    value: ms.Tensor,
    attention_mask: Optional[ms.Tensor],
    dropout: float = 0.0,
    scaling: float = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> tuple[ms.Tensor, None]:
    """
    Flash attention forward function. This function is a wrapper for `flash_attention_score` in MindSpore. It is used
    to calculate the attention score of the query and key, and then apply the attention score to the value.
    Args:
        module (`ms.Cell``):
            The attention module to be applied to the attention score.
        query (`ms.Tensor`):
            The query tensor of shape `(batch_size, num_head, seq_length, head_dim)`.
        key (`ms.Tensor`):
            The key tensor of shape `(batch_size, num_head, seq_length, head_dim)`.
        value (`ms.Tensor`):
            The value tensor of shape `(batch_size, num_head, seq_length, head_dim)`.
        attention_mask (`ms.Tensor`):
            The attention mask tensor of bool or uint8. For each element, 0/False indicates discard and 1/True
            indicates retention.The shape is `(batch_size, num_head, seq_length, head_dim)`. Default to `None`, which
            means no attention mask is applied.
        scaling (`float`, *required*):
            The scaling factor for the attention score.
        sliding_window (`int`):
            The sliding window size of self-attention. Default to `None`.
        softcap (`float`, *optional*):
            Softcap for the attention logits, used e.g. in gemma2. Default to `None`.

    """
    if kwargs.get("output_attentions", False) or kwargs.get("head_mask", None) is not None:
        logger.warning_once(
            "`flash_attention_2` does not support `output_attentions=True` or `head_mask`."
            " Please set your attention to `eager` if you want any of these features."
        )

    if attention_mask is not None:
        raise RuntimeError("FlashAttention's variable-length attention is not available.")
    if (
        kwargs.get("position_ids", None) is not None
        and query.shape[0] == 1
        and (
            kwargs.get("max_length_q", None) is not None
            or (query.shape[2] != 1 and not (mint.diff(kwargs["position_ids"], dim=-1) >= 0).all())
        )
    ):
        raise RuntimeError("FlashAttention's variable-length attention is not available.")
    if all(
        kwarg is not None
        for kwarg in (
            kwargs.get("cu_seq_lens_q", None),
            kwargs.get("cu_seq_lens_k", None),
            kwargs.get("max_length_q", None),
            kwargs.get("max_length_k", None),
        )
    ):
        raise RuntimeError("FlashAttention's variable-length attention is not available.")
    if sliding_window is not None:
        raise NotImplementedError("Sliding window is not supported in Mindspore yet. Please set `sliding_window=None`.")
    if softcap is not None:
        raise NotImplementedError("Softcap is not supported in Mindspore yet. Please set `softcap=None`.")

    if module.is_causal:
        attention_mask = mint.tril(mint.ones((query.shape[-2], key.shape[-2]), dtype=ms.bool_), diagonal=0)
    # For `attn_mask` of ops.flash_attention_score, False indicates retention and True indicates discard, Which is
    # opposite to PyTorch
    if attention_mask is not None:
        attention_mask = mint.logical_not(attention_mask) if attention_mask.dtype == ms.bool_ else attention_mask.bool()

    # flash_attention only supports [float16, bfloat16]
    origin_dtype = query.dtype
    if origin_dtype not in (ms.float16, ms.bfloat16):
        query = query.to(ms.float16)
        key = key.to(ms.float16)
        value = value.to(ms.float16)

    if scaling is None:
        # `flash_attention_score` does not support `None`
        # and the value can't set in jit mode, thus must be set in advance
        raise ValueError("`scaling` must be provided.")

    attn_output = ops.flash_attention_score(
        query,
        key,
        value,
        head_num=query.shape[1],
        attn_mask=attention_mask,
        keep_prob=1.0 - dropout,
        scalar_value=scaling,
        input_layout="BNSD",
    )
    attn_output = attn_output.to(origin_dtype)
    attn_output = mint.transpose(attn_output, 1, 2).contiguous()

    return attn_output, None
