# Copyright 2024 The Fairseq Authors and the HuggingFace Inc. team. All rights reserved.
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

from typing import Optional, TypedDict

from transformers.utils import logging

import mindspore as ms
from mindspore import mint, ops

logger = logging.get_logger(__name__)


def is_flash_attn_available():
    """Determine whether flash-attention can be used or not."""

    if ms.get_context("device_target") == "Ascend":
        return True

    return False


class FlashAttentionKwargs(TypedDict, total=False):
    """
    Keyword arguments for Flash Attention with Compile.

    Attributes:
        cu_seq_lens_q (`torch.LongTensor`, *optional*)
            Gets cumulative sequence length for query state.
        cu_seq_lens_k (`torch.LongTensor`, *optional*)
            Gets cumulative sequence length for key state.
        max_length_q (`int`, *optional*):
            Maximum sequence length for query state.
        max_length_k (`int`, *optional*):
            Maximum sequence length for key state.
    """

    cu_seq_lens_q: Optional[ms.Tensor]
    cu_seq_lens_k: Optional[ms.Tensor]
    max_length_q: Optional[int]
    max_length_k: Optional[int]


def flash_attn_func(
    query: ms.Tensor,
    key: ms.Tensor,
    value: ms.Tensor,
    attention_mask: Optional[ms.Tensor],
    dropout: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: Optional[bool] = False,
    sliding_window: Optional[int] = None,  # aligned with torch
    softcap: Optional[float] = None,  # a
    **kwargs,
):
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
    if causal and query.shape[-2] > 1:
        attention_mask = mint.tril(mint.ones((query.shape[-2], key.shape[-2]), dtype=ms.bool_))

    if sliding_window is not None:
        raise NotImplementedError("Sliding window is not supported in Mindspore yet. Please set `sliding_window=None`.")
    if softcap is not None:
        raise NotImplementedError("Softcap is not supported in Mindspore yet. Please set `softcap=None`.")

    # This is before the transpose
    num_head = query.shape[1]

    # BNSD -> BSND
    query = query.swapaxes(1, 2)
    key = key.swapaxes(1, 2)
    value = value.swapaxes(1, 2)
    input_layout = "BSND"

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

    attn_output = ops.flash_attention_score(
        query,
        key,
        value,
        head_num=num_head,
        attn_mask=attention_mask,
        keep_prob=1.0 - dropout,
        scalar_value=softmax_scale,
        input_layout=input_layout,
    )
    attn_output = attn_output.to(origin_dtype)

    return attn_output


def _flash_attention_forward(
    query_states: ms.tensor,
    key_states: ms.tensor,
    value_states: ms.tensor,
    attention_mask: ms.tensor,
    query_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    position_ids: Optional[ms.tensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: bool = None,
    cu_seq_lens_q: Optional[ms.tensor] = None,
    cu_seq_lens_k: Optional[ms.tensor] = None,
    max_length_q: Optional[int] = None,
    max_length_k: Optional[int] = None,
    target_dtype=None,
    **kwargs,
):
    """
    Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
    first unpad the input, then computes the attention scores and pad the final attention scores.

    Args:
        query_states (`ms.tensor`):
            Input query states to be passed to Flash Attention API
        key_states (`ms.tensor`):
            Input key states to be passed to Flash Attention API
        value_states (`ms.tensor`):
            Input value states to be passed to Flash Attention API
        attention_mask (`ms.tensor`):
            The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
            position of padding tokens and 1 for the position of non-padding tokens.
        dropout (`float`):
            Attention dropout
        softmax_scale (`float`, *optional*):
            The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        use_top_left_mask (`bool`, defaults to `False`):
            flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignment,
            that was made default for flash_attn>=2.1. This attribute is used to handle this difference.
        softcap (`float`, *optional*):
            Softcap for the attention logits, used e.g. in gemma2.
        deterministic (`bool`, *optional*):
            Determines if the deterministic option introduced in flash_attn>=2.4.1 is enabled.
    """
    if not use_top_left_mask:
        causal = is_causal
    else:
        causal = is_causal and query_length != 1

    if attention_mask is not None:
        raise RuntimeError("FlashAttention's variable-length attention is not available.")

    elif position_ids is not None and (
        max_length_q is not None or (query_length != 1 and not (mint.diff(position_ids, dim=-1) >= 0).all())
    ):
        raise RuntimeError("FlashAttention's variable-length attention is not available.")

    else:
        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout,
            softmax_scale,
            causal,
            sliding_window,
            softcap,
            **kwargs,
        )

    return attn_output
