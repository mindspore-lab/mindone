# Copyright 2025 StepFun Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# ==============================================================================

import math

from stepvideo.mindspore_adapter.scaled_dot_product_attn import scaled_dot_product_attention

import mindspore as ms
from mindspore import nn, ops

from mindone.transformers.mindspore_adapter.attention import DTYPE_FP16_MIN


class FlashSelfAttention(nn.Cell):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("mindspore fa do not support custom-flaot mask.")


# refer to:
# https://huggingface.co/stepfun-ai/Step-Audio-Chat/commit/aa82b184aa5ec627ef94545daa7a661711e83596#d2h-542184
# https://huggingface.co/stepfun-ai/Step-Audio-TTS-3B/blob/main/modeling_step1.py


class StepAttention(nn.Cell):
    def construct(self, q, k, v, cu_seqlens=None, max_seq_len=None):
        # b s h d
        _mask = self.build_alibi_cache(k.shape[1], q.shape[2], q.dtype)[:, :, -q.shape[1] :, :]

        # b s h d -> b h s d
        q = q.swapaxes(1, 2)
        k = k.swapaxes(1, 2)
        v = v.swapaxes(1, 2)

        attn_output = scaled_dot_product_attention(q, k, v, attn_mask=_mask)

        # b h s d -> b s h d
        attn_output = attn_output.swapaxes(1, 2)

        return attn_output

    def build_alibi_cache(self, block_size, n_heads, dtype):
        # get slopes
        n = 2 ** math.floor(math.log2(n_heads))  # nearest 2**n to n_heads
        m0 = 2.0 ** (-8.0 / n)
        # 2^(-8/n), 2^(-8*2/n), 2^(-8*3/n), ...
        slopes = ops.pow(m0, ops.arange(1, n + 1))
        if n < n_heads:
            m1 = 2.0 ** (-4.0 / n)
            # 2^(-8/(2n)), 2^(-8*3/(2n)), 2^(-8*5/(2n)), ...
            mm = ops.pow(m1, ops.arange(1, 1 + 2 * (n_heads - n), 2))
            slopes = ops.cat([slopes, mm])

        tril = ops.tril(ops.ones((1, 1, block_size, block_size), dtype=ms.bool_)).to(ms.int32)

        bias_rows = ops.arange(block_size).view(1, -1)
        bias_cols = ops.arange(block_size).view(-1, 1)
        bias = -ops.sqrt(bias_cols - bias_rows)
        bias = bias.view(1, block_size, block_size) * slopes.view(-1, 1, 1)
        bias = bias.masked_fill(tril == 0, DTYPE_FP16_MIN)

        return bias.type(dtype)
