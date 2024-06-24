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
from functools import partial

import numpy as np

import mindspore as ms
import mindspore.numpy as msnp
from mindspore import nn, ops
from mindspore.common.initializer import XavierUniform, initializer

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


class RelativePosition(nn.Cell):
    """ https://github.com/evelinehong/Transformer_Relative_Position_PyTorch/blob/master/relative_position.py """

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        # self.embeddings_table = ms.Parameter(ms.Tensor(max_relative_position * 2 + 1, num_units))
        # nn.init.xavier_uniform_(self.embeddings_table)
        self.embeddings_table = ms.Parameter(initializer(XavierUniform(), shape=(max_relative_position * 2 + 1, num_units), dtype=ms.float32))

    def forward(self, length_q, length_k):
        device = self.embeddings_table.device
        range_vec_q = ops.arange(length_q, device=device)
        range_vec_k = ops.arange(length_k, device=device)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = ops.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = final_mat.long()
        embeddings = self.embeddings_table[final_mat]
        return embeddings
    

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
        relative_position=False,
        temporal_length=None,
        video_length=None,
        image_cross_attention=False,
        image_cross_attention_scale=1.0,
        image_cross_attention_scale_learnable=False,
        text_context_len=77,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads

        # self.transpose = ops.Transpose()
        self.to_q = nn.Dense(query_dim, inner_dim, has_bias=False).to_float(dtype)
        self.to_k = nn.Dense(context_dim, inner_dim, has_bias=False).to_float(dtype)
        self.to_v = nn.Dense(context_dim, inner_dim, has_bias=False).to_float(dtype)
        self.to_out = nn.SequentialCell(
            nn.Dense(inner_dim, query_dim).to_float(dtype),
            nn.Dropout(dropout) if is_old_ms_version() else nn.Dropout(p=1 - dropout),
        )

        self.video_length = video_length
        self.image_cross_attention = image_cross_attention
        self.image_cross_attention_scale = image_cross_attention_scale
        self.image_cross_attention_scale_learnable = image_cross_attention_scale_learnable
        self.text_context_len = text_context_len
        if self.image_cross_attention:
            self.to_k_ip = nn.Dense(context_dim, inner_dim, has_bias=False).to_float(dtype)
            self.to_v_ip = nn.Dense(context_dim, inner_dim, has_bias=False).to_float(dtype)
            if image_cross_attention_scale_learnable:
                self.alpha = ms.Parameter(ms.Tensor(0.))

        self.relative_position = relative_position
        if self.relative_position:
            assert(temporal_length is not None)
            self.relative_position_k = RelativePosition(num_units=dim_head, max_relative_position=temporal_length)
            self.relative_position_v = RelativePosition(num_units=dim_head, max_relative_position=temporal_length)
            self.attention = Attention(dim_head,
                                       heads,
                                       self.relative_position,
                                       self.relative_position_k,
                                       self.relative_position_v,
                                       image_cross_attention_scale=image_cross_attention_scale,
                                       image_cross_attention_scale_learnable=image_cross_attention_scale_learnable,
                                       alpha=self.alpha if image_cross_attention_scale_learnable else None,
                                       )
        # else:
        #     ## only used for spatial attention, while NOT for temporal attention
        #     if XFORMERS_IS_AVAILBLE and temporal_length is None:
        #         self.forward = self.efficient_forward
        else:
            self.attention = Attention(dim_head,
                                       heads,
                                       image_cross_attention_scale=image_cross_attention_scale,
                                       image_cross_attention_scale_learnable=image_cross_attention_scale_learnable,
                                       alpha=self.alpha if image_cross_attention_scale_learnable else None,
                                       )

            
        self.head_dim = dim_head
        # self.attention = Attention(dim_head)

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
        spatial_self_attn = (context is None)
        k_ip, v_ip, out_ip = None, None, None

        h = self.heads
        q = self.to_q(x)
        context = default(context, x)

        # k = self.to_k(context)
        # v = self.to_v(context)
        if self.image_cross_attention and not spatial_self_attn:
            context, context_image = context[:, :self.text_context_len, :], context[:, self.text_context_len:, :]
            k = self.to_k(context)
            v = self.to_v(context)
            k_ip = self.to_k_ip(context_image)
            v_ip = self.to_v_ip(context_image)
        else:
            if not spatial_self_attn:
                context = context[:,:self.text_context_len,:]
            k = self.to_k(context)
            v = self.to_v(context)

        q_b, q_n, _ = q.shape  # (b s n*d)
        k_b, k_n, _ = k.shape
        v_b, v_n, _ = v.shape

        head_dim = q.shape[-1] // self.heads

        if (
            self.enable_flash_attention and q_n % 16 == 0 and k_n % 16 == 0 and head_dim <= self.FA_max_head_dim
        ):  # FIXME: now restrict head_dim to 128 to avoid 160 bug. revert to 256 once FA bug is fixed.
            raise NotImplementedError
            """
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
            """
        else:
            # (b, n, h*d) -> (b*h, n, d)
            q = self._rearange_in(q, h)
            k = self._rearange_in(k, h)
            v = self._rearange_in(v, h)

            out = self.attention(q, k, v, k_ip, v_ip, out_ip, mask)
            # (b*h, n, d) -> (b, n, h*d)
            # out = self._rearange_out(out, h)  # Done in class Attention

        return self.to_out(out).to(x_dtype)


class Attention(nn.Cell):
    def __init__(self,
                 dim_head,
                 heads=8,
                 relative_position=False,
                 relative_position_k=None,
                 relative_position_v=None,
                 image_cross_attention_scale=1.0,
                 image_cross_attention_scale_learnable=False,
                 alpha=None,
                 ):
        super().__init__()
        # self.softmax = ops.Softmax(axis=-1)
        # self.transpose = ops.Transpose()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.relative_position = relative_position
        self.relative_position_k = relative_position_k
        self.relative_position_v = relative_position_v
        self.image_cross_attention_scale_learnable = image_cross_attention_scale_learnable
        self.image_cross_attention_scale = image_cross_attention_scale    
        self.alpha = alpha
        if self.image_cross_attention_scale_learnable and self.alpha is None:
            raise ValueError

    def construct(self, q, k, v, k_ip, v_ip, out_ip, mask):
        sim = ops.matmul(q, ops.transpose(k, (0, 2, 1))) * self.scale
        # sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        if self.relative_position:
            len_q, len_k, len_v = q.shape[1], k.shape[1], v.shape[1]
            k2 = self.relative_position_k(len_q, len_k)
            sim2 = ops.matmul(q, ops.transpose(k2, (0, 2, 1))) * self.scale # TODO check 
            sim += sim2
        del k

        if exists(mask):
            mask = ops.reshape(mask, (mask.shape[0], -1))
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
        attn = ops.softmax(sim, axis=-1)
        out = ops.matmul(attn, v)

        if self.relative_position:
            v2 = self.relative_position_v(len_q, len_v)
            out2 = einsum('b t s, t s d -> b t d', sim, v2) # FIXME: 核对与torch.einsum的写法是否等效
            out += out2
        # out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        out = self._rearrange_out(out, self.heads)
        # out = ops.reshape(out, (-1, self.heads, out.shape[1], out.shape[2]))  # (b, h, n, d)
        # out = ops.transpose(out, (0, 2, 1, 3))  # (b, n, h, d)
        # out = ops.reshape(out, (out.shape[0], out.shape[1], -1))  # (b, n, (h d))

        ## for image cross-attention
        if k_ip is not None:
            k_ip = self._rearrange_in(k_ip, self.heads)
            v_ip = self._rearrange_in(v_ip, self.heads)
            sim_ip = ops.matmul(q, ops.transpose(k_ip, (0, 2, 1))) * self.scale
            # k_ip, v_ip = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (k_ip, v_ip))
            # sim_ip =  torch.einsum('b i d, b j d -> b i j', q, k_ip) * self.scale
            del k_ip
            sim_ip = sim_ip.softmax(axis=-1)
            out_ip = ops.matmul(sim_ip, v_ip)
            out_ip = self._rearrange_out(out_ip, self.heads)
            # out_ip = torch.einsum('b i j, b j d -> b i d', sim_ip, v_ip)
            # out_ip = rearrange(out_ip, '(b h) n d -> b n (h d)', h=h)


        if out_ip is not None:
            if self.image_cross_attention_scale_learnable:
                out = out + self.image_cross_attention_scale * out_ip * (ops.tanh(self.alpha)+1)
            else:
                out = out + self.image_cross_attention_scale * out_ip
        
        # return self.to_out(out)
        return out

    def _rearrange_in(self, x: ms.Tensor, h: int):
        """b n (h d) -> (b h) n d"""
        x = ops.reshape(x, (x.shape[0], x.shape[1], h, -1))  # (b, n, h, d)
        x = ops.transpose(x, (0, 2, 1, 3))  # (b, h, n, d)
        x = ops.reshape(x, (-1, x.shape[2], x.shape[3]))  # ((b h), n, d)
        return x
    
    def _rearrange_out(self, x: ms.Tensor, h: int):
        """(b h) n d -> b n (h d)"""
        x = ops.reshape(x, (-1, h, x.shape[1], x.shape[2]))  # (b, h, n, d)
        x = ops.transpose(x, (0, 2, 1, 3))  # (b, n, h, d)
        x = ops.reshape(x, (x.shape[0], x.shape[1], -1))  # (b, n, (h d))
        return x


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
        disable_self_attn=False,
        attention_cls=None,
        video_length=None,
        image_cross_attention=False,
        image_cross_attention_scale=1.0,
        image_cross_attention_scale_learnable=False,
        text_context_len=77,
        dtype=ms.float32,
        enable_flash_attention=False,
        unet_chunk_size=2,
    ):
        super().__init__()
        attn_cls = CrossAttention if attention_cls is None else attention_cls
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None,
            dtype=dtype,
            enable_flash_attention=enable_flash_attention,
        )  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff, dtype=dtype)
        self.attn2 = attn_cls(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            video_length=video_length,
            image_cross_attention=image_cross_attention,
            image_cross_attention_scale=image_cross_attention_scale,
            image_cross_attention_scale_learnable=image_cross_attention_scale_learnable,
            text_context_len=text_context_len,
            dtype=dtype,
            enable_flash_attention=enable_flash_attention,
        )  # is self-attn if context is none
        self.image_cross_attention = image_cross_attention

        self.norm1 = nn.LayerNorm([dim], epsilon=1e-05).to_float(dtype)
        self.norm2 = nn.LayerNorm([dim], epsilon=1e-05).to_float(dtype)
        self.norm3 = nn.LayerNorm([dim], epsilon=1e-05).to_float(dtype)
        self.checkpoint = checkpoint

    def construct(self, x, context=None):
        # import pdb;pdb.set_trace()
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
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
        disable_self_attn=False,
        use_linear=False,
        video_length=None,
        image_cross_attention=False,
        image_cross_attention_scale_learnable=False,
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

        attention_cls = None
        self.transformer_blocks = nn.CellList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim,
                    disable_self_attn=disable_self_attn,
                    checkpoint=use_checkpoint,
                    attention_cls=attention_cls,
                    video_length=video_length,
                    image_cross_attention=image_cross_attention,
                    image_cross_attention_scale_learnable=image_cross_attention_scale_learnable,
                    dtype=self.dtype,
                    enable_flash_attention=enable_flash_attention,
                    unet_chunk_size=unet_chunk_size,
                )
                for d in range(depth)
            ]
        )

        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0, has_bias=True, pad_mode="pad").to_float(self.dtype)
            )
        else:
            self.proj_out = zero_module(nn.Dense(in_channels, inner_dim).to_float(dtype))

        self.use_linear = use_linear
        # self.reshape = ops.Reshape()
        # self.transpose = ops.Transpose()

    def construct(self, x, context=None, **kwargs):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = ops.reshape(x, (b, c, h * w))  # (b, c, h*w)
        x = ops.transpose(x, (0, 2, 1))  # (b, h*w, c)
        if self.use_linear:
            x = self.proj_in(x)
        for block in self.transformer_blocks:
            x = block(x, context=context)
        if self.use_linear:
            x = self.proj_out(x)
        x = ops.reshape(x, (b, h, w, c))  # (b, h, w, c)
        x = ops.transpose(x, (0, 3, 1, 2))  # (b, c, h, w)
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


class TemporalTransformer(nn.Cell):
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
        disable_self_attn=False,
        use_linear=False,
        use_checkpoint=True,
        only_self_att=True,
        multiply_zero=False,  # from MS
        causal_attention=False,
        causal_block_size=1,
        relative_position=False,
        temporal_length=None,
        dtype=ms.float32,
    ):
        super().__init__()
        self.multiply_zero = multiply_zero
        self.only_self_att = only_self_att
        self.relative_position = relative_position
        self.causal_attention = causal_attention
        self.causal_block_size = causal_block_size
        self.dtype = dtype
        if self.only_self_att:
            context_dim = None
        if not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        if not use_linear:
            self.proj_in = nn.Conv1d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0, has_bias=True).to_float(self.dtype)
        else:
            self.proj_in = nn.Dense(in_channels, inner_dim).to_float(self.dtype)

        if relative_position:
            assert(temporal_length is not None)
            attention_cls = partial(CrossAttention, relative_position=True, temporal_length=temporal_length)
        else:
            attention_cls = partial(CrossAttention, temporal_length=temporal_length)
        if self.causal_attention:
            assert(temporal_length is not None)
            self.mask = ops.tril(ops.ones([1, temporal_length, temporal_length]))

        self.transformer_blocks = nn.CellList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    attention_cls=attention_cls,
                    checkpoint=use_checkpoint,
                    dtype=self.dtype,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv1d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0, has_bias=True).to_float(
                    self.dtype
                )
            )
        else:
            self.proj_out = zero_module(nn.Dense(in_channels, inner_dim).to_float(self.dtype))
            # if self.use_adaptor:
            #     self.adaptor_out = nn.Dense(frames, frames).to_float(self.dtype)  # todo: what frames
        self.use_linear = use_linear

    def construct(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if self.only_self_att:
            context = None
        if not isinstance(context, list):
            context = [context]
        b, c, t, h, w = x.shape
        x_in = x
        x = self.norm(x)


        # b c t h w -> (b h w) c t
        # x = rearrange(x, 'b c t h w -> (b h w) c t').contiguous()
        x = ops.transpose(x, (0, 3, 4, 1, 2))
        x = ops.reshape(x, (-1, x.shape[3], x.shape[4]))
        if not self.use_linear:
            x = self.proj_in(x)
        # bhw c t -> bhw t c
        # x = rearrange(x, 'bhw c t -> bhw t c').contiguous()
        x = ops.transpose(x, (0, 2, 1))
        if self.use_linear:
            x = self.proj_in(x)

        # if not self.use_linear:
        #     # b c t h w -> b h w c t -> (b h w) c t
        #     x = ops.transpose(x, (0, 3, 4, 1, 2))
        #     x = ops.reshape(x, (-1, x.shape[3], x.shape[4]))
        #     x = self.proj_in(x)
        # # [16384, 16, 320]
        # if self.use_linear:
        #     # (b t) c h w -> b t c h w -> b h w t c -> b (h w) t c
        #     x = ops.reshape(x, (x.shape[0] // t, t, x.shape[1], x.shape[2], x.shape[3]))
        #     # x = ops.reshape(x, (x.shape[0] // self.frames, self.frames, x.shape[1], x.shape[2], x.shape[3]))
        #     x = ops.transpose(x, (0, 3, 4, 1, 2))
        #     x = ops.reshape(x, (x.shape[0], -1, x.shape[3], x.shape[4]))  # todo: what frames
        #     x = self.proj_in(x)

            """NotImplemented
            temp_mask = None
            if self.causal_attention:
                # slice the from mask map
                temp_mask = self.mask[:,:t,:t].to(x.device)

            if temp_mask is not None:
                mask = temp_mask.to(x.device)
                mask = repeat(mask, 'l i j -> (l bhw) i j', bhw=b*h*w)
            else:
                mask = None
            """
        if self.only_self_att:
            # x = ops.transpose(x, (0, 2, 1))
            for i, block in enumerate(self.transformer_blocks):
                x = block(x)
            # (b hw) f c -> b hw f c
            x = ops.reshape(x, (b, x.shape[0] // b, x.shape[1], x.shape[2]))
        else:
            # (b hw) c f -> b hw f c
            x = ops.reshape(x, (b, x.shape[0] // b, x.shape[1], x.shape[2]))
            x = ops.transpose(x, (0, 1, 3, 2))
            for i, block in enumerate(self.transformer_blocks):
                # context[i] = repeat(context[i], '(b f) l con -> b (f r) l con', r=(h*w)//self.frames, f=self.frames).contiguous()
                # (b f) l con -> b f l con
                context[i] = ops.reshape(
                    context[i],
                    (context[i].shape[0] // self.frames, self.frames, context[i].shape[1], context[i].shape[2]),
                )  # todo: wtf frames
                # calculate each batch one by one (since number in shape could not greater then 65,535 for some package)
                for j in range(b):
                    context_i_j = ops.tile(context[i][j], ((h * w) // self.frames, 1, 1))  # todo: wtf frames
                    x[j] = block(x[j], context=context_i_j)

        if self.use_linear:
            x = self.proj_out(x)
            # b (h w) t c -> b h w t c -> b c t h w
            x = ops.reshape(x, (x.shape[0], h, w, x.shape[2], x.shape[3]))
            # x = ops.transpose(x, (0, 3, 4, 1, 2))
            x = ops.transpose(x, (0, 4, 3, 1, 2))
        if not self.use_linear:
            # b hw t c -> (b hw) t c -> (b hw) c t
            x = ops.reshape(x, (-1, x.shape[2], x.shape[3]))
            x = ops.transpose(x, (0, 2, 1))
            x = self.proj_out(x)
            # (b h w) c t -> b h w c t -> b c t h w
            x = ops.reshape(x, (b, h, w, x.shape[1], x.shape[2]))
            x = ops.transpose(x, (0, 3, 4, 1, 2))

        if self.multiply_zero:
            x = 0.0 * x + x_in
        else:
            x = x + x_in
        return x


class GroupNorm(nn.GroupNorm):
    # GroupNorm in calculated in FP32
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__(num_groups=num_groups, num_channels=num_channels, eps=eps, affine=affine)

    def construct(self, x):
        x_shape = x.shape
        dtype = x.dtype
        if x.ndim >= 3:
            x = x.view(x_shape[0], x_shape[1], x_shape[2], -1)
        y = super().construct(x.to(ms.float32)).to(dtype)
        return y.view(x_shape)
