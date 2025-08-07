# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import re
import math

import mindspore
from mindspore import nn

from ..utils import logging


logger = logging.get_logger(__name__)

FP4_VALUES = [
    +0.0,
    +0.5,
    +1.0,
    +1.5,
    +2.0,
    +3.0,
    +4.0,
    +6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


# Copied from https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/weights.py#L68
def convert_moe_packed_tensors(
    blocks,
    scales,
    *,
    dtype: mindspore.Type = mindspore.bfloat16,
    rows_per_chunk: int = 32768 * 1024,
) -> mindspore.Tensor:
    scales = scales.to(mindspore.int32) - 127

    assert blocks.shape[:-1] == scales.shape, f"{blocks.shape=} does not match {scales.shape=}"

    lut = mindspore.tensor(FP4_VALUES, dtype=dtype)

    *prefix_shape, G, B = blocks.shape
    rows_total = math.prod(prefix_shape) * G

    blocks = blocks.reshape(rows_total, B)
    scales = scales.reshape(rows_total, 1)

    out = mindspore.mint.empty(rows_total, B * 2, dtype=dtype)

    for r0 in range(0, rows_total, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, rows_total)

        blk = blocks[r0:r1]
        exp = scales[r0:r1]

        # nibble indices -> int64
        # idx_lo = (blk & 0x0F).to(mindspore.int64)
        # idx_hi = (blk >> 4).to(mindspore.int64)
        idx_lo = mindspore.tensor(blk.numpy() & 0x0F, mindspore.int64)
        idx_hi = mindspore.tensor(blk.numpy() >> 4, mindspore.int64)

        sub = out[r0:r1]
        sub[:, 0::2] = lut[idx_lo]
        sub[:, 1::2] = lut[idx_hi]

        out[r0:r1] = mindspore.ops.ldexp(sub, exp)
        del idx_lo, idx_hi, blk, exp

    return out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)


class Mxfp4GptOssExperts(nn.Cell):
    def __init__(self, config):
        super().__init__()

        self.num_experts = config.num_local_experts
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size

        self.gate_up_proj_blocks = mindspore.Parameter(
            mindspore.mint.zeros((self.num_experts, 2 * self.intermediate_size, self.hidden_size // 32, 16), dtype=mindspore.uint8),
            requires_grad=False,
        )
        self.gate_up_proj_scales = mindspore.Parameter(
            mindspore.mint.zeros((self.num_experts, 2 * self.intermediate_size, self.hidden_size // 32), dtype=mindspore.uint8),
            requires_grad=False,
        )
        # self.gate_up_proj_bias = mindspore.Parameter(
        #     mindspore.mint.zeros((self.num_experts, 2 * self.intermediate_size), dtype=mindspore.float32), requires_grad=False
        # )

        self.down_proj_blocks = mindspore.Parameter(
            mindspore.mint.zeros((self.num_experts, self.hidden_size, self.intermediate_size // 32, 16), dtype=mindspore.uint8),
            requires_grad=False,
        )
        self.down_proj_scales = mindspore.Parameter(
            mindspore.mint.zeros((self.num_experts, self.hidden_size, self.intermediate_size // 32), dtype=mindspore.uint8),
            requires_grad=False,
        )
        # self.down_proj_bias = mindspore.Parameter(
        #     mindspore.mint.zeros((self.num_experts, self.hidden_size), dtype=mindspore.float32), requires_grad=False
        # )
        self.alpha = 1.702

        self.gate_up_proj_precision_config = None
        self.down_proj_precision_config = None

        self.is_dequantized = False

    def construct(self, hidden_states: mindspore.Tensor, routing_data, gather_idx, scatter_idx) -> mindspore.Tensor:
        raise NotImplementedError

    # FIXME: Temporary for support MXFP4 inference
    def dequantize(self):

        if self.is_dequantized:
            return

        for proj in ["gate_up_proj", "down_proj"]:
            blocks_attr = f"{proj}_blocks"
            scales_attr = f"{proj}_scales"
            dequantized = convert_moe_packed_tensors(getattr(self, blocks_attr), getattr(self, scales_attr))
            dequantized = dequantized.transpose(1, 2).contiguous()
            setattr(self, proj, mindspore.Parameter(dequantized))
            delattr(self, blocks_attr)
            delattr(self, scales_attr)

        self.is_dequantized = True
