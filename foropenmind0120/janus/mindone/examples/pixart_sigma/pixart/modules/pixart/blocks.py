from typing import Literal, Optional, Tuple, Type, Union

import mindspore as ms
import mindspore.mint as mint
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, Tensor

from mindone.models.modules.flash_attention import MSFlashAttention

from ._layers import GELU, LayerNorm


def t2i_modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


class CrossAttention(nn.Cell):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        enable_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_linear = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.kv_linear = nn.Dense(dim, 2 * dim, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(p=proj_drop)
        self.enable_flash_attention = enable_flash_attention

        if self.enable_flash_attention:
            self.attention = MSFlashAttention(self.head_dim, self.num_heads, attention_dropout=attn_drop)
        else:
            self.attention = Attention(self.head_dim, attn_drop=attn_drop)

    def _rearange_in(self, x: Tensor) -> Tensor:
        # (b, n, h*d) -> (b, h, n, d)
        b, _, _ = x.shape
        x = ops.reshape(x, (b, -1, self.num_heads, self.head_dim))
        x = ops.transpose(x, (0, 2, 1, 3))
        return x

    def _rearange_out(self, x: Tensor) -> Tensor:
        # (b, h, n, d) -> (b, n, h*d)
        b, _, n, _ = x.shape
        x = ops.transpose(x, (0, 2, 1, 3))
        x = ops.reshape(x, (b, n, -1))
        return x

    def construct(self, x: Tensor, cond: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        B, N, _ = x.shape

        # (b, n, h*d) -> (b, n, h, d) -> (b, h, n, d)
        q = self.q_linear(x)
        q = self._rearange_in(q)

        # (b, n, 2*h*d) -> (b, n, 2, h, d) -> (2, b, h, n, d)
        kv = self.kv_linear(cond)
        kv = ops.reshape(kv, (B, -1, 2, self.num_heads, self.head_dim))
        kv = ops.transpose(kv, (2, 0, 3, 1, 4))
        k, v = kv.unbind(0)

        if self.enable_flash_attention and mask is not None:
            mask = ops.tile(~mask[:, None, None, :], (1, 1, N, 1))

        out = self.attention(q, k, v, mask=mask)
        out = self._rearange_out(out)
        return self.proj_drop(self.proj(out))


class KVCompressSelfAttention(nn.Cell):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        sampling: Literal[None, "conv", "ave", "uniform"] = None,
        sr_ratio: int = 1,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        enable_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.sr_ratio = sr_ratio
        self.sampling = sampling
        self.enable_flash_attention = enable_flash_attention

        self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(p=proj_drop)

        if self.enable_flash_attention:
            self.attention = MSFlashAttention(self.head_dim, self.num_heads, attention_dropout=attn_drop)
        else:
            self.attention = Attention(self.head_dim, attn_drop=attn_drop)

        if sr_ratio > 1 and self.sampling == "conv":
            # Avg Conv Init.
            self.sr = nn.Conv2d(
                dim,
                dim,
                sr_ratio,
                stride=sr_ratio,
                pad_mode="pad",
                group=dim,
                weight_init=1 / sr_ratio**2,
                bias_init=0,
            )
            self.norm = LayerNorm(dim)

        if qk_norm:
            self.q_norm = LayerNorm(dim)
            self.k_norm = LayerNorm(dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def downsample_2d(
        self,
        x: Tensor,
        H: int,
        W: int,
        scale_factor: int,
        sampling: Literal[None, "conv", "ave", "uniform"] = None,
    ) -> Tensor:
        B, _, C = x.shape
        if sampling is None or scale_factor == 1:
            return x

        x = x.reshape(B, H, W, C)
        x = ops.transpose(x, (0, 3, 1, 2))
        new_H, new_W = int(H / scale_factor), int(W / scale_factor)
        new_N = new_H * new_W

        if sampling == "ave":
            x = ops.interpolate(x, scale_factor=1 / scale_factor, mode="nearest")
            x = ops.transpose(x, (0, 2, 3, 1))
        elif sampling == "uniform":
            x = x[:, :, ::scale_factor, ::scale_factor]
            x = ops.transpose(x, (0, 2, 3, 1))
        else:
            x = self.sr(x).reshape(B, C, -1)
            x = ops.transpose(x, (0, 2, 1))
            x = self.norm(x)

        x = x.reshape(B, new_N, C)
        return x

    def _rearange_in(self, x: Tensor) -> Tensor:
        # (b, n, h*d) -> (b, h, n, d)
        b, _, _ = x.shape
        x = ops.reshape(x, (b, -1, self.num_heads, self.head_dim))
        x = ops.transpose(x, (0, 2, 1, 3))
        return x

    def _rearange_out(self, x: Tensor) -> Tensor:
        # (b, h, n, d) -> (b, n, h*d)
        b, _, n, _ = x.shape
        x = ops.transpose(x, (0, 2, 1, 3))
        x = ops.reshape(x, (b, n, -1))
        return x

    def construct(self, x: Tensor, mask: Optional[Tensor] = None, HW: Optional[Tuple[int, int]] = None) -> Tensor:
        B, N, C = x.shape

        if HW is None:
            H = W = int(N**0.5)
        else:
            H, W = HW

        # (b, n, 3*c) -> (b, n, 3, c) -> (3, b, n, c)
        qkv = self.qkv(x)
        qkv = ops.reshape(qkv, (B, N, 3, C))
        qkv = ops.transpose(qkv, (2, 0, 1, 3))
        q, k, v = qkv.unbind(0)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.sr_ratio > 1:
            k = self.downsample_2d(k, H, W, self.sr_ratio, sampling=self.sampling)
            v = self.downsample_2d(v, H, W, self.sr_ratio, sampling=self.sampling)

        q = self._rearange_in(q)
        k = self._rearange_in(k)
        v = self._rearange_in(v)

        if self.enable_flash_attention and mask is not None:
            mask = ops.tile(~mask[:, None, None, :], (1, 1, N, 1))

        out = self.attention(q, k, v, mask=mask)
        # (b, h, n, d) -> (b, n, h*d)
        out = self._rearange_out(out)

        return self.proj_drop(self.proj(out))


class Attention(nn.Cell):
    def __init__(self, dim_head: int, attn_drop: float = 0.0) -> None:
        super().__init__()
        self.scale = dim_head**-0.5
        self.attn_drop = nn.Dropout(p=attn_drop)

    def construct(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        k = ops.transpose(k, (0, 1, 3, 2))
        sim = ops.matmul(q, k) * self.scale

        # use fp32 for exponential inside
        sim = sim.to(ms.float32)
        if mask is not None:
            mask = mask[:, None, None, :]
            sim = ops.masked_fill(sim, ~mask, -ms.numpy.inf)
        attn = ops.softmax(sim, axis=-1).to(v.dtype)
        attn = self.attn_drop(attn)
        out = ops.matmul(attn, v)
        return out


class T2IFinalLayer(nn.Cell):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = LayerNorm(hidden_size, elementwise_affine=False, epsilon=1e-6)
        self.linear = nn.Dense(hidden_size, patch_size * patch_size * out_channels, has_bias=True)
        self.scale_shift_table = Parameter(ops.randn((2, hidden_size)) / hidden_size**0.5)
        self.out_channels = out_channels

    def construct(self, x: Tensor, t: Tensor) -> Tensor:
        shift, scale = mint.chunk(self.scale_shift_table[None] + t[:, None], 2, dim=1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class TimestepEmbedder(nn.Cell):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.SequentialCell(
            nn.Dense(frequency_embedding_size, hidden_size, has_bias=True),
            nn.SiLU(),
            nn.Dense(hidden_size, hidden_size, has_bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000) -> Tensor:
        half = dim // 2
        freqs = ops.exp(-ms.numpy.log(max_period) * ops.arange(start=0, end=half, dtype=ms.float32) / half)
        args = t[:, None].to(ms.float32) * freqs[None]
        embedding = ops.concat([ops.cos(args), ops.sin(args)], axis=-1)
        if dim % 2:
            embedding = ops.concat([embedding, ops.zeros_like(embedding[:, :1])], axis=-1)
        return embedding

    def construct(self, t: Tensor) -> Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class SizeEmbedder(TimestepEmbedder):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__(hidden_size=hidden_size, frequency_embedding_size=frequency_embedding_size)
        self.mlp = nn.SequentialCell(
            nn.Dense(frequency_embedding_size, hidden_size, has_bias=True),
            nn.SiLU(),
            nn.Dense(hidden_size, hidden_size, has_bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.outdim = hidden_size

    def construct(self, s: Tensor) -> Tensor:
        assert s.ndim == 2
        b = s.shape[0]
        s = ops.reshape(s, (-1,))
        s_freq = self.timestep_embedding(s, self.frequency_embedding_size)
        s_emb = self.mlp(s_freq)
        s_emb = ops.reshape(s_emb, (b, -1))
        return s_emb


class CaptionEmbedder(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        uncond_prob: float = 0.0,
        act_layer: Type[nn.Cell] = GELU,
        token_num: int = 120,
    ) -> None:
        super().__init__()
        self.y_proj = Mlp(
            in_features=in_channels,
            hidden_features=hidden_size,
            out_features=hidden_size,
            act_layer=act_layer,
            drop=0.0,
        )
        self.y_embedding = Parameter(ops.randn((token_num, in_channels)) / in_channels**0.5, requires_grad=False)
        self.uncond_prob = uncond_prob

    def token_drop(self, caption: Tensor, force_drop_ids: Optional[Tensor] = None) -> Tensor:
        if force_drop_ids is None:
            drop_ids = ops.rand(caption.shape[0]) < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = ops.where(drop_ids[:, None, None], self.y_embedding[None, ...].to(caption.dtype), caption)
        return caption

    def construct(self, caption: Tensor, force_drop_ids: Optional[Tensor] = None) -> Tensor:
        if self.training:
            assert caption.shape[1:] == self.y_embedding.shape
        if (self.training and self.uncond_prob > 0) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        caption = self.y_proj(caption)
        return caption


class Mlp(nn.Cell):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Type[nn.Cell] = GELU,
        norm_layer: Optional[Type[nn.Cell]] = None,
        bias: bool = True,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_channels=in_features, out_channels=hidden_features, has_bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(p=drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Dense(in_channels=hidden_features, out_channels=out_features, has_bias=bias)
        self.drop2 = nn.Dropout(p=drop)

    def construct(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class PatchEmbed(nn.Cell):
    def __init__(
        self,
        img_size: Optional[Union[int, Tuple[int, int]]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Type[nn.Cell]] = None,
        flatten: bool = True,
        bias: bool = True,
        strict_img_size: bool = True,
        dynamic_img_pad: bool = False,
    ) -> None:
        super().__init__()
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        if img_size is not None:
            self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
            self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        self.flatten = flatten
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, pad_mode="pad", has_bias=bias
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def construct(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        if self.img_size is not None:
            if self.strict_img_size:
                assert H == self.img_size[0], f"Input height ({H}) doesn't match model ({self.img_size[0]})."
                assert W == self.img_size[1], f"Input width ({W}) doesn't match model ({self.img_size[1]})."
            elif not self.dynamic_img_pad:
                assert (
                    H % self.patch_size[0] == 0
                ), f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]})."
                assert (
                    W % self.patch_size[1] == 0
                ), f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]})."

        if self.dynamic_img_pad:
            pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            x = ops.pad(x, (0, pad_w, 0, pad_h))

        x = self.proj(x)
        if self.flatten:
            # NCHW -> NLC
            x = ops.flatten(x, start_dim=2)
            x = ops.transpose(x, (0, 2, 1))

        x = self.norm(x)
        return x


class DropPath(nn.Cell):
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True) -> None:
        super().__init__()
        self.keep_prob = 1.0 - drop_prob
        self.scale_by_keep = scale_by_keep
        self.dropout = nn.Dropout(p=drop_prob)

    def construct(self, x: Tensor) -> Tensor:
        if self.keep_prob == 1.0 or not self.training:
            return x
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = self.dropout(ops.ones(shape))
        if not self.scale_by_keep:
            random_tensor = ops.mul(random_tensor, self.keep_prob)
        return x * random_tensor
