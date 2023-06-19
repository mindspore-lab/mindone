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
import numpy as np
from inspect import isfunction
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import initializer
from ldm.util import is_old_ms_version 


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    #return d() if isfunction(d) else d # TODO: this may lead to error in mindspore 2.1. use isinstance, and if return, return
    if isinstance(d, (ms.Tensor, int, float)):
        return d
    return d()


# def max_neg_value(t):
#     return -torch.finfo(t.dtype).max


class GEGLU(nn.Cell):
    def __init__(self, dim_in, dim_out, dtype=ms.float32):
        super().__init__()
        self.proj = nn.Dense(dim_in, dim_out * 2).to_float(dtype)
        self.split = split = ops.Split(-1, 2)
        self.gelu = ops.GeLU()

    def construct(self, x):
        x, gate = self.split(self.proj(x))
        
        return x * self.gelu(gate)


class FeedForward(nn.Cell):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=1.0, dtype=ms.float32):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Dense(dim, inner_dim).to_float(dtype),
            nn.GELU().to_float(dtype)
        ) if not glu else GEGLU(dim, inner_dim, dtype=dtype)
        self.net = nn.SequentialCell(
            project_in,
            nn.Dropout(dropout) if is_old_ms_version() else nn.Dropout(p=1-dropout),
            nn.Dense(inner_dim, dim_out).to_float(dtype)
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
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, has_bias = False, pad_mode="pad")
        self.to_out = nn.Conv2d(hidden_dim, dim, 1, has_bias = True, pad_mode="pad")


class CrossAttention(nn.Cell):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=1.0, dtype=ms.float32):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.reshape = ops.Reshape()
        self.softmax = ops.Softmax(axis=-1)
        self.transpose = ops.Transpose()
        self.to_q = nn.Dense(query_dim, inner_dim, has_bias=False).to_float(dtype)
        self.to_k = nn.Dense(context_dim, inner_dim, has_bias=False).to_float(dtype)
        self.to_v = nn.Dense(context_dim, inner_dim, has_bias=False).to_float(dtype)
        self.to_out = nn.SequentialCell(
            nn.Dense(inner_dim, query_dim).to_float(dtype),
            nn.Dropout(dropout) if is_old_ms_version() else nn.Dropout(p=1-dropout)
            )


    def construct(self, x, context=None, mask=None):
        q = self.to_q(x)
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
            x = self.reshape(x, (b*h, n, d))
            return x

        q = rearange_in(q)
        k = rearange_in(k)
        v = rearange_in(v)

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

        attn = self.softmax(sim)
        out = ops.matmul(attn, v)
        
        def rearange_out(x):
            # (b*h, n, d) -> (b, n, h*d)
            h = self.heads
            b, n, d = x.shape
            b = b // h

            x = self.reshape(x, (b, h, n, d))
            x = self.transpose(x, (0, 2, 1, 3))
            x = self.reshape(x, (b, n, h*d))
            return x

        out = rearange_out(out)
        return self.to_out(out)


class BasicTransformerBlock(nn.Cell):
    def __init__(self, dim, n_heads, d_head, dropout=1.0, context_dim=None, gated_ff=True, checkpoint=True, dtype=ms.float32):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout, dtype=dtype)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff, dtype=dtype)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout, dtype=dtype)  # is self-attn if context is none
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
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=1.0, context_dim=None, use_checkpoint=True, use_linear=False, dtype=ms.float32):
        super().__init__()
        self.in_channels = in_channels
        self.dtype=dtype
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0,
                                     has_bias=True,
                                     pad_mode='pad').to_float(dtype)
        else:
            self.proj_in = nn.Dense(in_channels,
                                    inner_dim).to_float(dtype)


        self.transformer_blocks = nn.CellList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim, 
                                   checkpoint=use_checkpoint, dtype=self.dtype)
                for d in range(depth)]
        )

        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                                  in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0,
                                                  has_bias=True,
                                                  pad_mode='pad').to_float(self.dtype))
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
        x = self.reshape(x, (b, c, h*w))    # (b, c, h*w)
        x = self.transpose(x, (0, 2, 1))    # (b, h*w, c)
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
