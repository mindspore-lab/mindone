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
from ldm.util import is_old_ms_version
from packaging import version

import mindspore as ms
import mindspore.numpy as msnp
from mindspore import nn, ops
from mindspore.common.initializer import initializer

try:
    from mindspore.nn.layer.flash_attention import FlashAttention

    FLASH_IS_AVAILABLE = True
    print("flash attention is available.")
except ImportError:
    FLASH_IS_AVAILABLE = False
    print("flash attention is unavailable.")

logger = logging.getLogger()


def exists(val):
    return val is not None


def uniq(arr):
    return {el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    # return d() if isfunction(d) else d
    # TODO: this may lead to error in mindspore 2.1. use isinstance, and if return, return
    if isinstance(d, (ms.Tensor, int, float)):
        return d
    return d()


class GEGLU(nn.Cell):
    def __init__(self, dim_in, dim_out, dtype=ms.float32):
        super().__init__()
        self.proj = nn.Dense(dim_in, dim_out * 2).to_float(dtype)
        self.split = ops.Split(-1, 2)
        self.gelu = ops.GeLU()
        # self.gelu = nn.GELU(approximate=False)

    def construct(self, x):
        x, gate = self.split(self.proj(x))

        return x * self.gelu(gate)


class FeedForward(nn.Cell):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=1.0, dtype=ms.float32):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.SequentialCell(nn.Dense(dim, inner_dim).to_float(dtype), nn.GELU().to_float(dtype))
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
        upcast=False,
        fa_max_head_dim=256,
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

        self.attention = Attention(dim_head, upcast=upcast)
        self.fa_max_head_dim = fa_max_head_dim

        self.enable_flash_attention = (
            enable_flash_attention and FLASH_IS_AVAILABLE and (ms.context.get_context("device_target") == "Ascend")
        )
        if self.enable_flash_attention:
            # TODO: how high_precision affect the training or inference quality
            if version.parse(ms.__version__) <= version.parse("2.2.0"):
                self.flash_attention = FlashAttention(head_dim=dim_head, high_precision=True)
                self.fa_mask_dtype = ms.float16  # choose_flash_attention_dtype()
            else:
                self.flash_attention = FlashAttention(head_dim=dim_head, head_num=heads, high_precision=True)
                self.fa_mask_dtype = ms.uint8  # choose_flash_attention_dtype()
            # logger.info("Flash attention is enabled.")
        else:
            self.flash_attention = None

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

        q_b, q_n, _ = q.shape  # (b n h*d)
        k_b, k_n, _ = k.shape
        v_b, v_n, _ = v.shape

        head_dim = q.shape[-1] // self.heads

        if (
            self.enable_flash_attention and q_n % 16 == 0 and k_n % 16 == 0 and head_dim <= self.fa_max_head_dim
        ):  # restrict head_dim to avoid UB oom. Reduce fa_max_head_dim value in case of OOM.
            # reshape qkv shape ((b n h*d) -> (b h n d))and mask dtype for FA input format
            q = q.view(q_b, q_n, h, -1).transpose(0, 2, 1, 3)
            k = k.view(k_b, k_n, h, -1).transpose(0, 2, 1, 3)
            v = v.view(v_b, v_n, h, -1).transpose(0, 2, 1, 3)
            if mask is None:
                mask = ops.zeros((q_b, q_n, q_n), self.fa_mask_dtype)
            # FIXME: a trick to pad sdv1.5 head dimensions from 160 to 256
            if head_dim == 160:
                # pad to 2**n * 64
                padding_size = 64 * 2 ** math.ceil(math.log(head_dim / 64, 2)) - head_dim
                q = msnp.pad(q, ((0, 0), (0, 0), (0, 0), (0, padding_size)), constant_value=0)
                k = msnp.pad(k, ((0, 0), (0, 0), (0, 0), (0, padding_size)), constant_value=0)
                v = msnp.pad(v, ((0, 0), (0, 0), (0, 0), (0, padding_size)), constant_value=0)

            out = self.flash_attention(
                q.to(ms.float16), k.to(ms.float16), v.to(ms.float16), mask.to(self.fa_mask_dtype)
            )
            if head_dim == 160:
                out = ops.slice(out, [0, 0, 0, 0], [q_b, h, q_n, head_dim])
            b, h, n, d = out.shape
            # reshape FA output to original attn input format, (b h n d) -> (b n h*d)
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


class CrossFrameAttention(CrossAttention):
    def __init__(self, unet_chunk_size=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unet_chunk_size = unet_chunk_size

    def construct(self, x, context=None, mask=None):
        x_dtype = x.dtype
        h = self.heads

        q = self.to_q(x)
        is_cross_attention = context is not None
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        def rearange_in(x):
            # (b, n, h*d) -> (b*h, n, d)
            h = self.heads
            b, n, d = x.shape
            d = d // h

            x = self.reshape(x, (b, n, h, d))
            x = self.transpose(x, (0, 2, 1, 3))
            x = self.reshape(x, (b * h, n, d))
            return x

        def rearange_frame(x, f):
            b, n, d = x.shape
            b = b // f
            x = self.reshape(x, (b, f, n, d))
            return x

        def rearange_frame_back(x):
            b, f, n, d = x.shape
            x = self.reshape(x, (b * f, n, d))
            return x

        if not is_cross_attention:
            video_length = k.shape[0] // self.unet_chunk_size
            former_frame_index = [0] * video_length
            k = rearange_frame(k, video_length)
            k = k[:, former_frame_index]
            k = rearange_frame_back(k)
            v = rearange_frame(v, video_length)
            v = v[:, former_frame_index]
            v = rearange_frame_back(v)

        q_b, q_n, _ = q.shape  # (b n h*d)
        k_b, k_n, _ = k.shape
        v_b, v_n, _ = v.shape

        head_dim = q.shape[-1] // self.heads

        if (
            self.enable_flash_attention and q_n % 16 == 0 and k_n % 16 == 0 and head_dim <= self.fa_max_head_dim
        ):  # restrict head_dim to avoid UB oom. Reduce fa_max_head_dim value in case of OOM.
            # reshape qkv shape ((b n h*d) -> (b h n d))and mask dtype for FA input format
            q = q.view(q_b, q_n, h, -1).transpose(0, 2, 1, 3)
            k = k.view(k_b, k_n, h, -1).transpose(0, 2, 1, 3)
            v = v.view(v_b, v_n, h, -1).transpose(0, 2, 1, 3)
            if mask is None:
                mask = ops.zeros((q_b, q_n, q_n), self.fa_mask_dtype)

            out = self.flash_attention(
                q.to(ms.float16), k.to(ms.float16), v.to(ms.float16), mask.to(self.fa_mask_dtype)
            )

            b, h, n, d = out.shape
            # reshape FA output to original attn input format, (b h n d) -> (b n h*d)
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
    def __init__(self, dim_head, upcast=False):
        super().__init__()
        self.softmax = ops.Softmax(axis=-1)
        self.transpose = ops.Transpose()
        self.scale = dim_head**-0.5
        self.upcast = upcast

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

        if self.upcast:
            # use fp32 for exponential inside
            attn = self.softmax(sim.astype(ms.float32)).astype(v.dtype)
        else:
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
        cross_frame_attention=False,
        unet_chunk_size=2,
        upcast_attn=False,
        fa_max_head_dim=256,
    ):
        super().__init__()
        if cross_frame_attention:
            self.attn1 = CrossFrameAttention(
                unet_chunk_size=unet_chunk_size,
                query_dim=dim,
                heads=n_heads,
                dim_head=d_head,
                dropout=dropout,
                dtype=dtype,
                enable_flash_attention=enable_flash_attention,
                upcast=upcast_attn,
                fa_max_head_dim=fa_max_head_dim,
            )  # is a self-attention
        else:
            self.attn1 = CrossAttention(
                query_dim=dim,
                heads=n_heads,
                dim_head=d_head,
                dropout=dropout,
                dtype=dtype,
                enable_flash_attention=enable_flash_attention,
                upcast=upcast_attn,
                fa_max_head_dim=fa_max_head_dim,
            )
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff, dtype=dtype)
        if cross_frame_attention:
            self.attn2 = CrossFrameAttention(
                unet_chunk_size=unet_chunk_size,
                query_dim=dim,
                context_dim=context_dim,
                heads=n_heads,
                dim_head=d_head,
                dropout=dropout,
                dtype=dtype,
                enable_flash_attention=enable_flash_attention,
                upcast=upcast_attn,
                fa_max_head_dim=fa_max_head_dim,
            )  # is self-attn if context is none
        else:
            self.attn2 = CrossAttention(
                query_dim=dim,
                context_dim=context_dim,
                heads=n_heads,
                dim_head=d_head,
                dropout=dropout,
                dtype=dtype,
                enable_flash_attention=enable_flash_attention,
                upcast=upcast_attn,
                fa_max_head_dim=fa_max_head_dim,
            )  # is self-attn if context is none
        self.norm1 = (
            nn.LayerNorm([dim], epsilon=1e-05).to_float(ms.float32)
            if upcast_attn
            else nn.LayerNorm([dim], epsilon=1e-05).to_float(dtype)
        )
        self.norm2 = (
            nn.LayerNorm([dim], epsilon=1e-05).to_float(ms.float32)
            if upcast_attn
            else nn.LayerNorm([dim], epsilon=1e-05).to_float(dtype)
        )
        self.norm3 = (
            nn.LayerNorm([dim], epsilon=1e-05).to_float(ms.float32)
            if upcast_attn
            else nn.LayerNorm([dim], epsilon=1e-05).to_float(dtype)
        )
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
        cross_frame_attention=False,
        unet_chunk_size=2,
        upcast_attn=False,
        fa_max_head_dim=256,
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
                    cross_frame_attention=cross_frame_attention,
                    unet_chunk_size=unet_chunk_size,
                    upcast_attn=upcast_attn,
                    fa_max_head_dim=fa_max_head_dim,
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
