import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import mindspore as ms
from mindspore import Tensor, nn, ops

from mindone.diffusers.models.normalization import LayerNorm

from ..math import attention, rope


class EmbedND(nn.Cell):
    def __init__(self, dim: int, theta: int, axes_dim: List[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def construct(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = ops.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            axis=-3,
        )

        return emb.unsqueeze(1)


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = ops.exp(-math.log(max_period) * ops.arange(start=0, end=half, dtype=ms.float32) / half)

    args = t[:, None].float() * freqs[None]
    embedding = ops.cat([ops.cos(args), ops.sin(args)], axis=-1)
    if dim % 2:
        embedding = ops.cat([embedding, ops.zeros_like(embedding[:, :1])], axis=-1)
    if ops.is_floating_point(t):
        embedding = embedding.to(t.dtype)
    return embedding


class MLPEmbedder(nn.Cell):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Dense(in_dim, hidden_dim, has_bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Dense(hidden_dim, hidden_dim, has_bias=True)

    def construct(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(nn.Cell):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = ms.Parameter(ops.ones((dim,)), name="scale")

    def construct(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = ops.rsqrt(ops.mean(x**2, axis=-1, keep_dims=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(nn.Cell):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def construct(self, q: Tensor, k: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v.dtype), k.to(v.dtype)


def unfuse_qkv(qkv, num_heads):
    # Equivalent to: q, k, v = einops.rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
    bsz, seq = qkv.shape[0], qkv.shape[1]
    q, k, v = ops.chunk(
        qkv.reshape(bsz, seq, 3, num_heads, -1).transpose(2, 0, 3, 1, 4).reshape(3 * bsz, num_heads, seq, -1),
        chunks=3,
        axis=0,
    )
    return q, k, v


class SelfAttention(nn.Cell):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Dense(dim, dim)

    def construct(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = unfuse_qkv(qkv, self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = self.proj(x)
        return x


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Cell):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Dense(dim, self.multiplier * dim, has_bias=True)

    def construct(self, vec: Tensor) -> Tuple[ModulationOut, Optional[ModulationOut]]:
        out = self.lin(ops.silu(vec))[:, None, :].chunk(self.multiplier, axis=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class DoubleModulation(nn.Cell):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        assert double
        self.multiplier = 6
        self.lin = nn.Dense(dim, self.multiplier * dim, has_bias=True)

    def construct(self, vec: Tensor) -> Tuple[ModulationOut, Optional[ModulationOut]]:
        out = self.lin(ops.silu(vec))[:, None, :].chunk(self.multiplier, axis=-1)
        return out[:3], out[3:]


class SingleModulation(nn.Cell):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        assert not double
        self.multiplier = 3
        self.lin = nn.Dense(dim, self.multiplier * dim, has_bias=True)

    def construct(self, vec: Tensor) -> Tuple[ModulationOut, Optional[ModulationOut]]:
        out = self.lin(ops.silu(vec))[:, None, :].chunk(self.multiplier, axis=-1)
        return out


class DoubleStreamBlock(nn.Cell):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = DoubleModulation(hidden_size, double=True)
        self.img_norm1 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.img_norm2 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.SequentialCell(
            nn.Dense(hidden_size, mlp_hidden_dim, has_bias=True),
            nn.GELU(approximate=True),
            nn.Dense(mlp_hidden_dim, hidden_size, has_bias=True),
        )

        self.txt_mod = DoubleModulation(hidden_size, double=True)
        self.txt_norm1 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.txt_norm2 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.SequentialCell(
            nn.Dense(hidden_size, mlp_hidden_dim, has_bias=True),
            nn.GELU(approximate=True),
            nn.Dense(mlp_hidden_dim, hidden_size, has_bias=True),
        )

    def construct(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor) -> Tuple[Tensor, Tensor]:
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1[1]) * img_modulated + img_mod1[0]
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = unfuse_qkv(img_qkv, self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1[1]) * txt_modulated + txt_mod1[0]
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = unfuse_qkv(txt_qkv, self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = ops.cat((txt_q, img_q), axis=2)
        k = ops.cat((txt_k, img_k), axis=2)
        v = ops.cat((txt_v, img_v), axis=2)

        attn = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]  # noqa

        # calculate the img bloks
        img = img + img_mod1[2] * self.img_attn.proj(img_attn)
        img = img + img_mod2[2] * self.img_mlp((1 + img_mod2[1]) * self.img_norm2(img) + img_mod2[0])

        # calculate the txt bloks
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
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Dense(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Dense(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate=True)
        self.modulation = SingleModulation(hidden_size, double=False)

    def construct(self, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        mod = self.modulation(vec)
        x_mod = (1 + mod[1]) * self.pre_norm(x) + mod[0]
        qkv, mlp = ops.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], axis=-1)

        q, k, v = unfuse_qkv(qkv, self.num_heads)
        q, k = self.norm(q, k, v)

        # compute attention
        attn = attention(q, k, v, pe=pe)
        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(ops.cat((attn, self.mlp_act(mlp)), 2))
        return x + mod[2] * output


class LastLayer(nn.Cell):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Dense(hidden_size, patch_size * patch_size * out_channels, has_bias=True)
        self.adaLN_modulation = nn.SequentialCell(nn.SiLU(), nn.Dense(hidden_size, 2 * hidden_size, has_bias=True))

    def construct(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, axis=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x
