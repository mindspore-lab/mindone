# Copyright 2024 The Fairseq Authors and the HuggingFace Inc. team. All rights reserved.
#
# This code is adapted from https://github.com/huggingface/transformers
# with modifications to run transformers on mindspore.
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
        cu_seq_lens_q (`ms.Tensor`, *optional*)
            Gets cumulative sequence length for query state.
        cu_seq_lens_k (`ms.Tensor`, *optional*)
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


def _is_packed_sequence(position_ids, batch_size):
    """
    # NOTE `position_ids` here is for allowing padding free approach -> enable varlen flash attn. not for real computing.
    # it is user's responsibility to take care of flattening `position_ids` if that's needed by the model. and curren†ly
    # varlen flash attn are not avaliable yet.

    Check the position ids whether packed sequences are indicated or not
        1. Position ids exist
        2. Flattened sequences only are supported
        3. Compile-friendly `not (torch.diff(position_ids, dim=-1) >= 0).all()`, i.e. we have multiple increasing sequences
    """
    if position_ids is None:
        return False

    increasing_position_sequences = mint.arange(position_ids.shape[1]) + position_ids.min()
    return batch_size == 1 and (increasing_position_sequences - position_ids).abs().sum().bool()


def _flash_attention_forward(
    query_states: ms.Tensor,
    key_states: ms.Tensor,
    value_states: ms.Tensor,
    attention_mask: Optional[ms.Tensor],
    query_length: int = None,
    is_causal: bool = None,
    dropout: float = 0.0,
    position_ids: Optional[ms.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: Optional[bool] = None,
    cu_seq_lens_q: Optional[ms.Tensor] = None,
    cu_seq_lens_k: Optional[ms.Tensor] = None,
    max_length_q: Optional[int] = None,
    max_length_k: Optional[int] = None,
    target_dtype: Optional[ms.Type] = None,
    implementation: Optional[str] = None,
    input_layout: str = "BSND",
    **kwargs,
):
    """
    Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
    first unpad the input, then computes the attention scores and pad the final attention scores.

    (Optional) kwargs are described further in `_process_flash_attention_kwargs` and `FlashAttentionKwargs`.

    Args:
        query_states (`ms.Tensor`):
            Input query states to be passed to Flash Attention API
        key_states (`ms.Tensor`):
            Input key states to be passed to Flash Attention API
        value_states (`ms.Tensor`):
            Input value states to be passed to Flash Attention API
        attention_mask (`ms.Tensor`, *optional*):
            The attention mask tensor of bool or uint8. For each element, 0/False indicates discard and 1/True
            indicates retention.The shape is `(batch_size, num_head, seq_length_q, seq_length_k)`,
            `(batch_size, 1, seq_length_q, seq_length_k)` or `(seq_length_q, seq_length_k)`.
            Default to `None`, which means no attention mask is applied.
        is_causal (`bool`, *optional*):
            `True` indicates to use causal mask, `None` or `False` not to use. Default to `None`.
        implementation (`str`, *optional*):
            The attention implementation to use. If None, will default to the one based on the environment.
        input_layout (`str`, *optional*):
            Input query, key and value input layout, supports `BSND` or `BNSD`. Default to `BSND`.
    """
    # if ( # NOTE: this is commented to support padded position_ids inputs
    #     kwargs.get("position_ids", None) is not None
    #     and query.shape[0] == 1
    #     and (
    #         kwargs.get("max_length_q", None) is not None
    #         or (query.shape[1] != 1 and not (mint.diff(kwargs["position_ids"], dim=-1) >= 0).all())
    #     )
    # ):
    #     raise RuntimeError("FlashAttention's variable-length attention is not available.")

    # NOTE: `max_length_q`, `max_length_k`, `cu_seq_lens_q`, `cu_seq_lens_k` are originally for variable-length flash attn,
    # not available yet.
    if max_length_q is not None or max_length_k is not None:
        raise RuntimeError("FlashAttention's variable-length attention is not available.")
    if cu_seq_lens_q is not None or cu_seq_lens_k is not None:
        raise ValueError(
            "`_flash_attention_forward` does not support `cu_seq_lens_q` or `cu_seq_lens_k` yet,"
            "please use `ops.flash_attention_score` with input_layout `TND` instead."
        )

    if sliding_window is not None:
        raise NotImplementedError("Sliding window is not supported yet. Please set `sliding_window=None`.")
    if use_top_left_mask:
        raise NotImplementedError(
            "Top left mask is not supported yet. Please set `use_top_left_mask=False`."
            "it's an outdated args for BC in upstream repo. will be deprecated in future version"
        )
    if softcap is not None:
        raise NotImplementedError("Softcap is not supported yet. Please set `softcap=None`.")
    if deterministic:
        raise NotImplementedError(
            "`deterministic` option is not supported yet. Please set `deterministic=None`. "
            "originally introduced in flash_attn>=2.4.1"
        )
    if input_layout == "TND":
        raise ValueError(
            "`_flash_attention_forward` does not support input_layout `TND` yet, "
            "please use `ops.flash_attention_score` instead."
        )
    if softmax_scale is None:
        # `flash_attention_score` does not support `None`
        # and the value can't be set in jit mode, thus must be set in advance
        raise ValueError("`softmax_scale` must be provided.")

    num_head = query_states.shape[2] if input_layout == "BSND" else query_states.shape[1]

    # For `attn_mask` of ops.flash_attention_score, False indicates retention and True indicates discard, Which is
    # opposite to PyTorch
    if attention_mask is not None:
        attention_mask = mint.logical_not(attention_mask) if attention_mask.dtype == ms.bool_ else attention_mask.bool()

    # Apply causal mask
    seq_len_q = query_states.shape[1] if input_layout == "BSND" else query_states.shape[2]
    seq_len_k = key_states.shape[1] if input_layout == "BSND" else key_states.shape[2]
    if is_causal and seq_len_q > 1:
        if attention_mask is None:  # create a new mask
            attention_mask = mint.triu(mint.ones((seq_len_q, seq_len_k), dtype=ms.bool_), diagonal=1)
        else:  # integrate causal mask and input attention mask
            causal_mask = mint.triu(mint.ones_like(attention_mask, dtype=ms.bool_), diagonal=1)
            attention_mask = attention_mask | causal_mask

    # flash_attention only supports [float16, bfloat16]
    origin_dtype = query_states.dtype
    if target_dtype in (ms.float16, ms.bfloat16):
        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)
    elif origin_dtype not in (ms.float16, ms.bfloat16):
        query_states = query_states.to(ms.float16)
        key_states = key_states.to(ms.float16)
        value_states = value_states.to(ms.float16)

    out = ops.flash_attention_score(
        query_states,
        key_states,
        value_states,
        head_num=num_head,
        attn_mask=attention_mask,
        keep_prob=1.0 - dropout,
        scalar_value=softmax_scale,
        input_layout=input_layout,
    )
    out = out.to(origin_dtype)

    return out
