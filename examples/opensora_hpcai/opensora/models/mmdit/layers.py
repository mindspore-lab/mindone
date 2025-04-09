# Modified from Flux
#
# Copyright 2024 Black Forest Labs

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import Literal, Optional

import numpy as np

from mindspore import PYNATIVE_MODE, Parameter, Tensor
from mindspore import dtype as mstype
from mindspore import get_context, mint, nn, ops, tensor
from mindspore.communication import get_group_size
from mindspore.ops.operations.nn_ops import FlashAttentionScore

from ...acceleration import AlltoAll2 as AlltoAll
from ...acceleration import get_sequence_parallel_group
from .math import apply_rope, apply_rotary_pos_emb, liger_rope, rope


class EmbedND(nn.Cell):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def construct(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = mint.cat([rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)], dim=-3)
        return emb.unsqueeze(1)


class LigerEmbedND(nn.Cell):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def construct(self, ids: Tensor) -> tuple[Tensor, Tensor]:
        n_axes = ids.shape[-1]
        cos_list = []
        sin_list = []
        for i in range(n_axes):
            cos, sin = liger_rope(ids[..., i], self.axes_dim[i], self.theta)
            cos_list.append(cos)
            sin_list.append(sin)
        cos_emb = mint.cat(cos_list, dim=-1).tile((1, 1, 2)).contiguous()
        sin_emb = mint.cat(sin_list, dim=-1).tile((1, 1, 2)).contiguous()

        return mint.cat((cos_emb, sin_emb))


class SinusoidalEmbedding(nn.Cell):
    def __init__(self, frequency_embedding_size: int, max_period: int = 10000, time_factor: float = 1000.0):
        super().__init__()
        half = frequency_embedding_size // 2
        self._freqs = tensor(
            np.expand_dims(np.exp(-np.log(max_period) * np.arange(start=0, stop=half, dtype=np.float32) / half), axis=0)
        )
        self._dim = frequency_embedding_size
        self._time_factor = time_factor

    def construct(self, x: Tensor) -> Tensor:
        x = x * self._time_factor
        args = x[:, None].to(self._freqs.dtype) * self._freqs
        embedding = mint.cat([mint.cos(args), mint.sin(args)], dim=-1)
        if self._dim % 2:
            embedding = mint.cat([embedding, mint.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(x.dtype)


class MLPEmbedder(nn.Cell):
    def __init__(self, in_dim: int, hidden_dim: int, dtype: mstype.Type = mstype.float32):
        super().__init__()
        self.in_layer = nn.Dense(in_dim, hidden_dim, has_bias=True, dtype=dtype)
        self.silu = nn.SiLU()
        self.out_layer = nn.Dense(hidden_dim, hidden_dim, has_bias=True, dtype=dtype)

    def construct(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(nn.Cell):
    def __init__(self, dim: int, eps: float = 1e-6, dtype: mstype.Type = mstype.float32):
        super().__init__()
        self.scale = Parameter(tensor(np.ones(dim), dtype=dtype))
        self.variance_epsilon = eps
        self.dtype = dtype

    def construct(self, x: Tensor) -> Tensor:
        if self.dtype == mstype.float16:  # for faster graph building
            return ops.rms_norm(x.to(mstype.float32), self.scale.to(mstype.float32), epsilon=self.variance_epsilon)[
                0
            ].to(mstype.float16)
        return ops.rms_norm(x, self.scale, epsilon=self.variance_epsilon)[0]


class QKNorm(nn.Cell):
    def __init__(self, dim: int, dtype: mstype.Type = mstype.float32):
        super().__init__()
        self.query_norm = RMSNorm(dim, dtype=dtype)
        self.key_norm = RMSNorm(dim, dtype=dtype)

    def construct(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        return self.query_norm(q), self.key_norm(k)


class Attention(nn.Cell):
    """
    Vanilla attention just for tests
    """

    def __init__(self, head_dim: int):
        super().__init__()
        self._scale = head_dim**-0.5

    def construct(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        b, h, _, d = q.shape
        q = mint.reshape(q, (-1, q.shape[2], q.shape[3]))
        k = mint.reshape(k, (-1, k.shape[2], k.shape[3]))
        v = mint.reshape(v, (-1, v.shape[2], v.shape[3]))

        sim = mint.matmul(q, k.transpose(0, 2, 1)) * self._scale

        # (b h n_q n_k)
        attn = mint.softmax(sim, dim=-1)
        out = mint.matmul(attn, v)

        return mint.reshape(out, (b, h, -1, d))


class SelfAttention(nn.Cell):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        fused_qkv: bool = True,
        use_liger_rope: bool = False,
        dtype: mstype.Type = mstype.float32,
    ):
        super().__init__()
        self._use_lr = use_liger_rope
        self.num_heads = num_heads
        self.fused_qkv = fused_qkv
        head_dim = dim // num_heads

        self.flash_attention = FlashAttentionScore(num_heads, scale_value=head_dim**-0.5, input_layout="BNSD")

        if fused_qkv:
            self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias, dtype=dtype)
        else:
            self.q_proj = nn.Dense(dim, dim, has_bias=qkv_bias, dtype=dtype)
            self.k_proj = nn.Dense(dim, dim, has_bias=qkv_bias, dtype=dtype)
            self.v_proj = nn.Dense(dim, dim, has_bias=qkv_bias, dtype=dtype)
        self.norm = QKNorm(head_dim, dtype=dtype)
        self.proj = nn.Dense(dim, dim, dtype=dtype)

    def construct(self, x: Tensor, pe: Tensor) -> Tensor:
        if self.fused_qkv:
            qkv = self.qkv(x)
            # B L (K H D) -> K B H L D
            qkv = qkv.reshape(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = mint.split(qkv, split_size_or_sections=1)
            q, k, v = mint.squeeze(q, dim=0), mint.squeeze(k, dim=0), mint.squeeze(v, dim=0)
        else:
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
            q, k, v = (  # B L (H D) -> B L H D
                q.reshape(q.shape[0], q.shape[1], self.num_heads, -1),
                k.reshape(k.shape[0], k.shape[1], self.num_heads, -1),
                v.reshape(v.shape[0], v.shape[1], self.num_heads, -1),
            )
        q, k = self.norm(q, k, v)
        if not self.fused_qkv:
            q, k, v = q.swapdims(1, 2), k.swapdims(1, 2), v.swapdims(1, 2)  # B L H D -> B H L D

        if self._use_lr:
            cos, sin = mint.chunk(pe, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        else:
            q, k = apply_rope(q, k, pe)
        x = self.flash_attention(q, k, v, None, None, None, None)[-1]
        x = x.swapdims(1, 2).reshape(x.shape[0], x.shape[2], -1)  # B H L D -> B L (H D)

        x = self.proj(x)
        return x


class SingleModulation(nn.Cell):
    def __init__(self, dim: int, dtype: mstype.Type = mstype.float32):
        super().__init__()
        self.multiplier = 3
        self.lin = nn.Dense(dim, self.multiplier * dim, has_bias=True, dtype=dtype)
        self.silu = nn.SiLU()

    def construct(self, vec: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        out = self.lin(self.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)
        return out


class DoubleModulation(nn.Cell):
    def __init__(self, dim: int, dtype: mstype.Type = mstype.float32):
        super().__init__()
        self.multiplier = 6
        self.lin = nn.Dense(dim, self.multiplier * dim, has_bias=True, dtype=dtype)
        self.silu = nn.SiLU()

    def construct(self, vec: Tensor) -> tuple[tuple[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]]:
        out = self.lin(self.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)
        return out[:3], out[3:]


class DoubleStreamBlock(nn.Cell):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool = False,
        fused_qkv: bool = True,
        use_liger_rope: bool = False,
        attn_type: Literal["eager", "flash_attention"] = "flash_attention",
        dtype: mstype.Type = mstype.float32,
    ):
        super().__init__()
        self._use_lr = use_liger_rope
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        if (sp_group := get_sequence_parallel_group()) is not None:
            self.sp_group_size = get_group_size(sp_group)
            self.alltoall = (
                AlltoAll(1, 2, group=sp_group)
                if get_context("mode") == PYNATIVE_MODE
                else ops.AlltoAll(self.sp_group_size, 1, 2, group=sp_group)
            )
            num_heads = num_heads // self.sp_group_size
        else:
            self.sp_group_size = None
            self.alltoall = nn.Identity()

        self.attention, self.flash_attention = None, None
        if attn_type == "flash_attention":
            self.flash_attention = FlashAttentionScore(
                num_heads, scale_value=self.head_dim**-0.5, input_layout="BNSD"
            )
        else:
            self.attention = Attention(self.head_dim)

        # image stream
        self.img_mod = DoubleModulation(hidden_size, dtype=dtype)
        self.img_norm1 = mint.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype)
        self.img_attn = SelfAttention(
            dim=hidden_size, num_heads=self.num_heads, qkv_bias=qkv_bias, fused_qkv=fused_qkv, dtype=dtype
        )

        self.img_norm2 = mint.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype)
        self.img_mlp = nn.SequentialCell(
            nn.Dense(hidden_size, mlp_hidden_dim, has_bias=True, dtype=dtype),
            nn.GELU(),
            nn.Dense(mlp_hidden_dim, hidden_size, has_bias=True, dtype=dtype),
        )

        # text stream
        self.txt_mod = DoubleModulation(hidden_size, dtype=dtype)
        self.txt_norm1 = mint.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype)
        self.txt_attn = SelfAttention(
            dim=hidden_size,
            num_heads=self.num_heads,
            qkv_bias=qkv_bias,
            fused_qkv=fused_qkv,
            use_liger_rope=use_liger_rope,
            dtype=dtype,
        )

        self.txt_norm2 = mint.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype)
        self.txt_mlp = nn.SequentialCell(
            nn.Dense(hidden_size, mlp_hidden_dim, has_bias=True, dtype=dtype),
            nn.GELU(),
            nn.Dense(mlp_hidden_dim, hidden_size, has_bias=True, dtype=dtype),
        )

    def construct(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, **kwargs) -> tuple[Tensor, Tensor]:
        # process img and txt separately while both are influenced by text vec

        # vec will interact with image latent and text context
        img_mod1, img_mod2 = self.img_mod(vec)  # get shift, scale, gate for each mod
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1[1]) * img_modulated + img_mod1[0]

        if self.img_attn.fused_qkv:
            img_qkv = self.img_attn.qkv(img_modulated)
            img_qkv = img_qkv.reshape(img_qkv.shape[0], img_qkv.shape[1], 3, self.num_heads, -1).permute(
                2, 0, 3, 1, 4
            )  # B L (K H D) -> K B H L D
            img_q, img_k, img_v = mint.split(img_qkv, split_size_or_sections=1)
            img_q, img_k, img_v = mint.squeeze(img_q, dim=0), mint.squeeze(img_k, dim=0), mint.squeeze(img_v, dim=0)
        else:
            img_q, img_k, img_v = (
                self.img_attn.q_proj(img_modulated),
                self.img_attn.k_proj(img_modulated),
                self.img_attn.v_proj(img_modulated),
            )
            img_q, img_k, img_v = (  # B L (H D) -> B L H D
                img_q.reshape(img_q.shape[0], img_q.shape[1], self.num_heads, -1),
                img_k.reshape(img_k.shape[0], img_k.shape[1], self.num_heads, -1),
                img_v.reshape(img_v.shape[0], img_v.shape[1], self.num_heads, -1),
            )

        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)  # RMSNorm for QK Norm as in SD3 paper
        if not self.img_attn.fused_qkv:
            img_q, img_k, img_v = img_q.swapdims(1, 2), img_k.swapdims(1, 2), img_v.swapdims(1, 2)  # B L H D -> B H L D

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1[1]) * txt_modulated + txt_mod1[0]
        if self.txt_attn.fused_qkv:
            txt_qkv = self.txt_attn.qkv(txt_modulated)
            # B L (K H D) -> K B H L D
            txt_qkv = txt_qkv.reshape(txt_qkv.shape[0], txt_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            txt_q, txt_k, txt_v = mint.split(txt_qkv, split_size_or_sections=1)
            txt_q, txt_k, txt_v = mint.squeeze(txt_q, dim=0), mint.squeeze(txt_k, dim=0), mint.squeeze(txt_v, dim=0)
        else:
            txt_q, txt_k, txt_v = (
                self.txt_attn.q_proj(txt_modulated),
                self.txt_attn.k_proj(txt_modulated),
                self.txt_attn.v_proj(txt_modulated),
            )
            txt_q, txt_k, txt_v = (  # B L (H D) -> B L H D
                txt_q.reshape(txt_q.shape[0], txt_q.shape[1], self.num_heads, -1),
                txt_k.reshape(txt_k.shape[0], txt_k.shape[1], self.num_heads, -1),
                txt_v.reshape(txt_v.shape[0], txt_v.shape[1], self.num_heads, -1),
            )
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)
        if not self.txt_attn.fused_qkv:
            txt_q, txt_k, txt_v = txt_q.swapdims(1, 2), txt_k.swapdims(1, 2), txt_v.swapdims(1, 2)  # B L H D -> B H L D

        # run actual attention, image and text attention are calculated together by concat different attn heads
        q = mint.cat((txt_q, img_q), dim=2)
        k = mint.cat((txt_k, img_k), dim=2)
        v = mint.cat((txt_v, img_v), dim=2)

        q, k, v = self.alltoall(q), self.alltoall(k), self.alltoall(v)

        if self._use_lr:
            cos, sin = pe[: pe.shape[0] // 2], pe[pe.shape[0] // 2 :]
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        else:
            q, k = apply_rope(q, k, pe)

        if self.flash_attention is not None:
            attn1 = self.flash_attention(q, k, v, None, None, None, None)[-1]
        else:
            attn1 = self.attention(q, k, v)
        attn1 = attn1.swapdims(1, 2).reshape(attn1.shape[0], attn1.shape[2], -1)  # B H L D -> B L (H D)

        attn1 = self.alltoall(attn1)

        txt_attn, img_attn = attn1[:, : txt_q.shape[2]], attn1[:, txt_q.shape[2] :]

        # calculate the img blocks
        img = img + img_mod1[2] * self.img_attn.proj(img_attn)
        img = img + img_mod2[2] * self.img_mlp((1 + img_mod2[1]) * self.img_norm2(img) + img_mod2[0])

        # calculate the txt blocks
        txt = txt + txt_mod1[2] * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2[2] * self.txt_mlp((1 + txt_mod2[1]) * self.txt_norm2(txt) + txt_mod2[0])
        return img, txt


class SingleStreamBlock(nn.Cell):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: Optional[float] = None,
        fused_qkv: bool = True,
        use_liger_rope: bool = False,
        attn_type: Literal["eager", "flash_attention"] = "flash_attention",
        dtype: mstype.Type = mstype.float32,
    ):
        super().__init__()
        self._use_lr = use_liger_rope
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        self.fused_qkv = fused_qkv

        if (sp_group := get_sequence_parallel_group()) is not None:
            self.sp_group_size = get_group_size(sp_group)
            self.alltoall = (
                AlltoAll(1, 2, group=sp_group)
                if get_context("mode") == PYNATIVE_MODE
                else ops.AlltoAll(self.sp_group_size, 1, 2, group=sp_group)
            )
            num_heads = num_heads // self.sp_group_size
        else:
            self.sp_group_size = None
            self.alltoall = nn.Identity()

        self.attention, self.flash_attention = None, None
        if attn_type == "flash_attention":
            self.flash_attention = FlashAttentionScore(
                num_heads, scale_value=self.head_dim**-0.5, input_layout="BNSD"
            )
        else:
            self.attention = Attention(self.head_dim)

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if fused_qkv:
            # qkv and mlp_in
            self.linear1 = nn.Dense(hidden_size, hidden_size * 3 + self.mlp_hidden_dim, dtype=dtype)
        else:
            self.q_proj = nn.Dense(hidden_size, hidden_size, dtype=dtype)
            self.k_proj = nn.Dense(hidden_size, hidden_size, dtype=dtype)
            self.v_mlp = nn.Dense(hidden_size, hidden_size + self.mlp_hidden_dim, dtype=dtype)

        # proj and mlp_out
        self.linear2 = nn.Dense(hidden_size + self.mlp_hidden_dim, hidden_size, dtype=dtype)

        self.norm = QKNorm(self.head_dim, dtype=dtype)

        self.hidden_size = hidden_size
        self.pre_norm = mint.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype)

        self.mlp_act = nn.GELU()
        self.modulation = SingleModulation(hidden_size, dtype=dtype)

    def construct(self, x: Tensor, vec: Tensor, pe: Tensor, **kwargs) -> Tensor:
        mod = self.modulation(vec)
        x_mod = (1 + mod[1]) * self.pre_norm(x) + mod[0]
        if self.fused_qkv:
            qkv, mlp = mint.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
            # B L (K H D) -> K B H L D
            qkv = qkv.reshape(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = mint.split(qkv, split_size_or_sections=1)
            q, k, v = mint.squeeze(q, dim=0), mint.squeeze(k, dim=0), mint.squeeze(v, dim=0)
        else:
            q, k = self.q_proj(x_mod), self.k_proj(x_mod)
            v, mlp = mint.split(self.v_mlp(x_mod), [self.hidden_size, self.mlp_hidden_dim], dim=-1)
            q, k, v = (  # B L (H D) -> B L H D
                q.reshape(q.shape[0], q.shape[1], self.num_heads, -1),
                k.reshape(k.shape[0], k.shape[1], self.num_heads, -1),
                v.reshape(v.shape[0], v.shape[1], self.num_heads, -1),
            )

        q, k = self.norm(q, k, v)
        if not self.fused_qkv:
            q, k, v = q.swapdims(1, 2), k.swapdims(1, 2), v.swapdims(1, 2)  # B L H D -> B H L D

        q, k, v = self.alltoall(q), self.alltoall(k), self.alltoall(v)

        # compute attention
        if self._use_lr:
            cos, sin = pe[: pe.shape[0] // 2], pe[pe.shape[0] // 2 :]
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        else:
            q, k = apply_rope(q, k, pe)
        if self.flash_attention is not None:
            attn_1 = self.flash_attention(q, k, v, None, None, None, None)[-1]
        else:
            attn_1 = self.attention(q, k, v)
        attn_1 = attn_1.swapdims(1, 2).reshape(attn_1.shape[0], attn_1.shape[2], -1)  # B H L D -> B L (H D)

        attn_1 = self.alltoall(attn_1)

        # compute activation in mlp stream, cat again, and run the second linear layer
        output = self.linear2(mint.cat((attn_1, self.mlp_act(mlp)), 2))
        output = x + mod[2] * output
        return output


class LastLayer(nn.Cell):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, dtype: mstype.Type = mstype.float32):
        super().__init__()
        self.norm_final = mint.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype)
        self.linear = nn.Dense(hidden_size, patch_size * patch_size * out_channels, has_bias=True, dtype=dtype)
        self.adaLN_modulation = nn.SequentialCell(
            nn.SiLU(), nn.Dense(hidden_size, 2 * hidden_size, has_bias=True, dtype=dtype)
        )

    def construct(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x
