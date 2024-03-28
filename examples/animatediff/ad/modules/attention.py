# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import logging
import math

import numpy as np

import mindspore as ms
import mindspore.numpy as msnp
from mindspore import nn, ops
from mindspore.common.initializer import initializer

from mindone.utils.version_control import (
    MS_VERSION,
    check_valid_flash_attention,
    choose_flash_attention_dtype,
    is_old_ms_version,
)

logger = logging.getLogger()

FLASH_IS_AVAILABLE = check_valid_flash_attention()
FA_MS23_UPDATE = False
if FLASH_IS_AVAILABLE:
    try:
        from mindspore.nn.layer.flash_attention import FlashAttention
    except Exception:
        # for ms2.3 >= 20240219, FA API changed
        from mindspore.ops.operations.nn_ops import FlashAttentionScore

        FA_MS23_UPDATE = True
        print("D--: get MS2.3 PoC API! ")

    logger.info("Flash attention is available.")


def exists(val):
    return val is not None


def uniq(arr):
    return {el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    if isinstance(d, (ms.Tensor, int, float)):
        return d
    return d()


class GEGLU(nn.Cell):
    def __init__(self, dim_in, dim_out, dtype=ms.float32):
        super().__init__()
        self.proj = nn.Dense(dim_in, dim_out * 2).to_float(dtype)
        self.split = ops.Split(-1, 2)
        self.gelu = ops.GeLU()

    def construct(self, x):
        x, gate = self.split(self.proj(x))

        return x * self.gelu(gate)


class FeedForward(nn.Cell):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=1.0, dtype=ms.float32):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(nn.Dense(dim, inner_dim).to_float(dtype), nn.GELU().to_float(dtype))
            if not glu
            else GEGLU(dim, inner_dim, dtype=dtype)
        )
        self.net = nn.SequentialCell(
            project_in,
            nn.Dropout(dropout) if is_old_ms_version() else nn.Dropout(p=1 - dropout),
            nn.Dense(inner_dim, dim_out).to_float(dtype),
        )

    def construct(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    weight = initializer("zeros", module.weight.shape)
    bias_weight = initializer("zeros", module.bias.shape)
    module.weight.set_data(weight)
    module.bias.set_data(bias_weight)
    return module


def Normalize(in_channels):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True).to_float(ms.float32)


class LinearAttention(nn.Cell):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, has_bias=False, pad_mode="pad")
        self.to_out = nn.Conv2d(hidden_dim, dim, 1, has_bias=True, pad_mode="pad")


class CrossAttention(nn.Cell):
    """
    Flash attention doesnot work well (leading to noisy images) for SD1.5-based models on 910B up to MS2.2.1-20231122 version,
    due to the attention head dimension is 40, num heads=5. Require test on future versions
    """

    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=1.0,
        dtype=ms.float32,
        enable_flash_attention=False,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads

        self.transpose = ops.Transpose()
        self.to_q = nn.Dense(query_dim, inner_dim, has_bias=False).to_float(dtype)
        self.to_k = nn.Dense(context_dim, inner_dim, has_bias=False).to_float(dtype)
        self.to_v = nn.Dense(context_dim, inner_dim, has_bias=False).to_float(dtype)
        self.to_out = nn.SequentialCell(
            nn.Dense(inner_dim, query_dim).to_float(dtype),
            nn.Dropout(dropout) if is_old_ms_version() else nn.Dropout(p=1 - dropout),
        )
        self.head_dim = dim_head
        self.attention = Attention(dim_head)

        self.enable_flash_attention = (
            enable_flash_attention and FLASH_IS_AVAILABLE and (ms.context.get_context("device_target") == "Ascend")
        )

        if self.enable_flash_attention:
            if not FA_MS23_UPDATE:
                self.flash_attention = FlashAttention(head_dim=dim_head, head_num=heads, high_precision=True)
            else:
                # TODO: for MS2.3 PoC, how to adapt high_precision=True?
                # Q: (b s n*d) -> (b n s d))  #  s - seq_len, n - num_head, d - head dim
                self.flash_attention = FlashAttentionScore(
                    scale_value=1.0 / math.sqrt(dim_head),  # required if we didn't scale q or k before FA
                    head_num=heads,
                    input_layout="BNSD",  # BSH or BNSD
                )
            # TODO: need to change mask type for MS2.3 PoC version?
            self.fa_mask_dtype = choose_flash_attention_dtype()  # ms.uint8 or ms.float16 depending on version
            # logger.info("Flash attention is enabled.")
        else:
            self.flash_attention = None

        # TODO: due to FA supports well for head dimensions: 64, 80, 96, 120, 128 and 256
        self.FA_max_head_dim = 256
        if MS_VERSION >= "2.2" and MS_VERSION < "2.3":
            self.FA_pad_head_dim = 160
        elif MS_VERSION >= "2.3":
            self.FA_pad_head_dim = 40
            # if self.enable_flash_attention:
            # logger.warning("Will head_dim 40 to 64 for Flash Attention for MS2.3-dev. This can be removed in later MS version after check.")
        else:
            self.FA_pad_head_dim = -1

    @staticmethod
    def _rearange_in(x, h):
        # (b, n, h*d) -> (b*h, n, d)
        b, n, d = x.shape
        d = d // h

        x = ops.reshape(x, (b, n, h, d))
        x = ops.transpose(x, (0, 2, 1, 3))
        x = ops.reshape(x, (b * h, n, d))
        return x

    @staticmethod
    def _rearange_out(x, h):
        # (b*h, n, d) -> (b, n, h*d)
        b, n, d = x.shape
        b = b // h

        x = ops.reshape(x, (b, h, n, d))
        x = ops.transpose(x, (0, 2, 1, 3))
        x = ops.reshape(x, (b, n, h * d))
        return x

    def construct(self, x, context=None, mask=None):
        x_dtype = x.dtype
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q_b, q_n, _ = q.shape  # (b s n*d)
        k_b, k_n, _ = k.shape
        v_b, v_n, _ = v.shape

        head_dim = q.shape[-1] // self.heads

        if (
            self.enable_flash_attention and q_n % 16 == 0 and k_n % 16 == 0 and head_dim <= self.FA_max_head_dim
        ):  # FIXME: now restrict head_dim to 128 to avoid 160 bug. revert to 256 once FA bug is fixed.
            # reshape qkv shape ((b s n*d) -> (b n s d))and mask dtype for FA input format
            q = q.view(q_b, q_n, h, -1).transpose(0, 2, 1, 3)
            k = k.view(k_b, k_n, h, -1).transpose(0, 2, 1, 3)
            v = v.view(v_b, v_n, h, -1).transpose(0, 2, 1, 3)

            if head_dim == self.FA_pad_head_dim:
                # pad to 2**n * 64
                padding_size = 64 * 2 ** math.ceil(math.log(head_dim / 64, 2)) - head_dim
                q = msnp.pad(q, ((0, 0), (0, 0), (0, 0), (0, padding_size)), constant_value=0)
                k = msnp.pad(k, ((0, 0), (0, 0), (0, 0), (0, padding_size)), constant_value=0)
                v = msnp.pad(v, ((0, 0), (0, 0), (0, 0), (0, padding_size)), constant_value=0)

            if not FA_MS23_UPDATE:
                if mask is None:
                    mask = ops.zeros((q_b, q_n, q_n), self.fa_mask_dtype)
                out = self.flash_attention(
                    q.to(ms.float16), k.to(ms.float16), v.to(ms.float16), mask.to(self.fa_mask_dtype)
                )
            else:
                _, _, _, out = self.flash_attention(
                    q.to(ms.float16), k.to(ms.float16), v.to(ms.float16), None, None, None, None, None
                )

            if head_dim == self.FA_pad_head_dim:
                out = ops.slice(out, [0, 0, 0, 0], [q_b, h, q_n, head_dim])

            b, h, n, d = out.shape
            # reshape FA output to original attn input format, (b n s d) -> (b s n*d)
            out = out.transpose(0, 2, 1, 3).view(b, n, -1)
        else:
            # (b, n, h*d) -> (b*h, n, d)
            q = self._rearange_in(q, h)
            k = self._rearange_in(k, h)
            v = self._rearange_in(v, h)

            out = self.attention(q, k, v, mask)
            # (b*h, n, d) -> (b, n, h*d)
            out = self._rearange_out(out, h)

        return self.to_out(out).to(x_dtype)


class Attention(nn.Cell):
    def __init__(self, dim_head):
        super().__init__()
        self.softmax = ops.Softmax(axis=-1)
        self.transpose = ops.Transpose()
        self.scale = dim_head**-0.5

    def construct(self, q, k, v, mask):
        sim = ops.matmul(q, self.transpose(k, (0, 2, 1))) * self.scale

        if exists(mask):
            mask = self.reshape(mask, (mask.shape[0], -1))
            if sim.dtype == ms.float16:
                finfo_type = np.float16
            else:
                finfo_type = np.float32
            max_neg_value = -np.finfo(finfo_type).max
            mask = mask.repeat(self.heads, axis=0)
            mask = ops.expand_dims(mask, axis=1)
            sim.masked_fill(mask, max_neg_value)

        # TODO: testing use fp16 instead
        # use fp32 for exponential inside
        # attn = self.softmax(sim.astype(ms.float32)).astype(v.dtype)
        attn = self.softmax(sim)

        out = ops.matmul(attn, v)

        return out


class BasicTransformerBlock(nn.Cell):
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=1.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        dtype=ms.float32,
        enable_flash_attention=False,
        unet_chunk_size=2,
    ):
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            dtype=dtype,
            enable_flash_attention=enable_flash_attention,
        )  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff, dtype=dtype)
        self.attn2 = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            dtype=dtype,
            enable_flash_attention=enable_flash_attention,
        )  # is self-attn if context is none
        self.norm1 = nn.LayerNorm([dim], epsilon=1e-05).to_float(dtype)
        self.norm2 = nn.LayerNorm([dim], epsilon=1e-05).to_float(dtype)
        self.norm3 = nn.LayerNorm([dim], epsilon=1e-05).to_float(dtype)
        self.checkpoint = checkpoint

    def construct(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Cell):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=1.0,
        context_dim=None,
        use_checkpoint=True,
        use_linear=False,
        dtype=ms.float32,
        enable_flash_attention=False,
        unet_chunk_size=2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.dtype = dtype
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        if not use_linear:
            self.proj_in = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0, has_bias=True, pad_mode="pad"
            ).to_float(dtype)
        else:
            self.proj_in = nn.Dense(in_channels, inner_dim).to_float(dtype)

        self.transformer_blocks = nn.CellList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim,
                    checkpoint=use_checkpoint,
                    dtype=self.dtype,
                    enable_flash_attention=enable_flash_attention,
                    unet_chunk_size=unet_chunk_size,
                )
                for d in range(depth)
            ]
        )

        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv2d(
                    inner_dim, in_channels, kernel_size=1, stride=1, padding=0, has_bias=True, pad_mode="pad"
                ).to_float(self.dtype)
            )
        else:
            self.proj_out = zero_module(nn.Dense(in_channels, inner_dim).to_float(dtype))

        self.use_linear = use_linear
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()

    def construct(self, x, emb=None, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = self.reshape(x, (b, c, h * w))  # (b, c, h*w)
        x = self.transpose(x, (0, 2, 1))  # (b, h*w, c)
        if self.use_linear:
            x = self.proj_in(x)
        for block in self.transformer_blocks:
            x = block(x, context=context)
        if self.use_linear:
            x = self.proj_out(x)
        x = self.reshape(x, (b, h, w, c))
        x = self.transpose(x, (0, 3, 1, 2))
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in
