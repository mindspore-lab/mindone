# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging

import mindspore as ms
from mindspore import Tensor, nn, ops

from mindone.models.modules.flash_attention import FLASH_IS_AVAILABLE, MSFlashAttention

XFORMERS_ENABLED = FLASH_IS_AVAILABLE

logger = logging.getLogger("dinov2")


class Attention(nn.Cell):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        FA_dtype=ms.bfloat16,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Dense(dim, dim, has_bias=proj_bias)
        self.proj_drop = nn.Dropout(p=proj_drop)

        if XFORMERS_ENABLED:
            self.flash_attention = MSFlashAttention(
                head_dim=head_dim,
                head_num=self.num_heads,
                attention_dropout=attn_drop,
                input_layout="BNSD",
                dtype=FA_dtype,
            )

    def construct(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute((2, 0, 3, 1, 4))

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.swapaxes(-2, -1)

        attn = attn.softmax(axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).swapaxes(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# In inference, do not use attn_bias=cam_emb yet
class MemEffAttention(Attention):
    def construct(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_ENABLED:
            # if attn_bias is not None:
            #     raise AssertionError("xFormers is required for using nested tensors")
            return super().construct(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = ops.unbind(qkv, 2)  # each 'b n h d'
        q = q.swapaxes(1, 2)
        k = k.swapaxes(1, 2)
        v = v.swapaxes(1, 2)
        # 'b n h d' -> (b, h=num_head, n, d) == BNSD

        x = self.flash_attention(q, k, v)  # orginally memory_efficient_attention(q, k ,v, attn_bias=attn_bias)
        x = x.swapaxes(1, 2)  # b h n d -> b n h d
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
