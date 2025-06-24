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


def _flash_attention_forward(
    query: ms.Tensor,
    key: ms.Tensor,
    value: ms.Tensor,
    num_head,
    attention_mask,
    dropout,
    scaling,
    input_layout,
):
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

    return attn_output


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
