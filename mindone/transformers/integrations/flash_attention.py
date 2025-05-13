from typing import Optional, Tuple

import mindspore as ms
from mindspore import nn, mint, ops
from transformers.utils import logging

logger = logging.get_logger(__name__)


def flash_attention_forward(
    module: Optional[nn.Cell], # aligned with torch
    query: ms.Tensor,
    key: ms.Tensor,
    value: ms.Tensor,
    attention_mask: Optional[ms.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None, # aligned with torch
    softcap: Optional[float] = None,  # a
    **kwargs,
) -> Tuple[ms.Tensor, None]:
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
        sliding_window (`int`):
            The sliding window size of self-attention. Default to `None`.
        softcap (`float`, *optional*):
            Softcap for the attention logits, used e.g. in gemma2. Default to `None`.

    """

    if sliding_window is not None:
        raise NotImplementedError(
            "Sliding window is not supported in Mindspore yet. Please set `sliding_window=None`."
        )
    if softcap is not None:
        raise NotImplementedError(
            "Softcap is not supported in Mindspore yet. Please set `softcap=None`."
        )

    # This is before the transpose
    num_head = query.shape[1]
    seq_len = query.shape[2]  # BNSD, N: num_head, S: seq_len, D: head_dim

    # BNSD -> BSND
    query = query.swapaxes(1, 2)
    key = key.swapaxes(1, 2)
    value = value.swapaxes(1, 2)
    input_layout = "BSND"

    # For `attn_mask` of ops.flash_attention_score, False indicates retention and True indicates discard, Which is
    # opposite to PyTorch
    seq_len_key = key.shape[2]
    if attention_mask is not None:
        attention_mask = mint.logical_not(attention_mask) if attention_mask.dtype == ms.bool_ else attention_mask.bool()

    # flash_attention only supports [float16, bfloat16]
    origin_dtype = query.dtype
    if origin_dtype not in (ms.float16, ms.bfloat16):
        query = query.to(ms.float16)
        key = key.to(ms.float16)
        value = value.to(ms.float16)

    attn_output = ops.flash_attention_score(
        query,
        key,
        value,
        head_num=num_head,
        attn_mask=attention_mask,
        keep_prob=1.0 - dropout,
        scalar_value=scaling,
        input_layout=input_layout,
    )
    attn_output = attn_output.to(origin_dtype)

    return attn_output, None
