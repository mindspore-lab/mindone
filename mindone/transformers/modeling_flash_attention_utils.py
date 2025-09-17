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
from mindspore import ops

logger = logging.get_logger(__name__)


def is_flash_attn_available():
    """Determine whether flash-attention can be used or not."""

    if ms.get_context("device_target") == "Ascend":
        return True

    return False


_flash_supports_window = None


def flash_attn_supports_top_left_mask():
    raise NotImplementedError("flash_attn_supports_top_left_mask is not supported yet.")


class FlashAttentionKwargs(TypedDict, total=False):
    cumulative_seqlens_q: Optional[ms.Tensor]
    cumulative_seqlens_k: Optional[ms.Tensor]


def _flash_attention_forward(
    query_states: ms.Tensor,
    key_states: ms.Tensor,
    value_states: ms.Tensor,
    attention_mask: Optional[ms.Tensor],
    query_length: int,
    scaling: float = None,
    input_layout: str = "BSND",
    dropout: float = 0.0,
    position_ids: Optional[ms.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    **kwargs,
):
    # flash_attention only supports [float16, bfloat16]
    origin_dtype = query_states.dtype
    if origin_dtype not in (ms.float16, ms.bfloat16):
        query_states = query_states.to(ms.float16)
        key_states = key_states.to(ms.float16)
        value_states = value_states.to(ms.float16)

    attn_output = ops.flash_attention_score(
        query_states,
        key_states,
        value_states,
        head_num=query_length,
        attn_mask=attention_mask,
        keep_prob=1.0 - dropout,
        scalar_value=scaling,
        input_layout=input_layout,
    )
    attn_output = attn_output.to(origin_dtype)

    return attn_output
