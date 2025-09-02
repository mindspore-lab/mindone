# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
from typing import Any, List, Optional, Tuple

import numpy as np

import mindspore as ms
import mindspore.mint as mint
import mindspore.nn as nn
from mindspore.common.initializer import XavierUniform, initializer

from mindone.diffusers.configuration_utils import ConfigMixin, register_to_config
from mindone.diffusers.models.layers_compat import unflatten
from mindone.diffusers.models.modeling_utils import ModelMixin
from mindone.models.utils import normal_, xavier_uniform_, zeros_

from ..utils.amp import autocast
from .attention import flash_attention

__all__ = ["WanModel"]


def sinusoidal_embedding_1d(dim: int, position: ms.Tensor) -> ms.Tensor:
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(ms.float32)

    # calculation
    sinusoid = mint.outer(position, mint.pow(10000, -mint.arange(half).to(position.dtype).div(half)))
    x = mint.cat([mint.cos(sinusoid), mint.sin(sinusoid)], dim=1)
    return x


def rope_params(max_seq_len: int, dim: int, theta: int = 10000) -> ms.Tensor:
    assert dim % 2 == 0
    freqs = mint.outer(mint.arange(max_seq_len), 1.0 / mint.pow(theta, mint.arange(0, dim, 2).to(ms.float32).div(dim)))
    freqs = mint.stack([mint.cos(freqs), mint.sin(freqs)], dim=-1)
    return freqs


def complex_mult(a: ms.Tensor, b: ms.Tensor) -> ms.Tensor:
    a_real, a_complex = mint.unbind(a, dim=-1)
    b_real, b_complex = mint.unbind(b, dim=-1)
    out_real = a_real * b_real - a_complex * b_complex
    out_complex = a_real * b_complex + b_real * a_complex
    return mint.stack([out_real, out_complex], dim=-1)


def rope_apply(x: ms.Tensor, grid_sizes: ms.Tensor, freqs: ms.Tensor) -> ms.Tensor:
    dtype = x.dtype
    x = x.to(ms.float32)
    n, c = x.shape[2], x.shape[3] // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = x[i, :seq_len].to(ms.float32).reshape(seq_len, n, -1, 2)
        freqs_i = mint.cat(
            [
                freqs[0][:f].view(f, 1, 1, -1, 2).expand((f, h, w, -1, 2)),
                freqs[1][:h].view(1, h, 1, -1, 2).expand((f, h, w, -1, 2)),
                freqs[2][:w].view(1, 1, w, -1, 2).expand((f, h, w, -1, 2)),
            ],
            dim=-2,
        ).reshape(seq_len, 1, -1, 2)

        # apply rotary embedding
        x_i = complex_mult(x_i, freqs_i).flatten(2)
        x_i = mint.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return mint.stack(output).to(dtype)


class WanRMSNorm(nn.Cell):
    def __init__(self, dim: int, eps: float = 1e-5, dtype: Any = ms.float32):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = ms.Parameter(ms.Tensor(np.ones(dim), dtype=dtype))

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x: ms.Tensor) -> ms.Tensor:
        return x * mint.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(mint.nn.LayerNorm):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False, dtype: Any = ms.float32):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps, dtype=dtype)

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        dtype = x.dtype
        with autocast(dtype=ms.float32):
            x = super().construct(x)
        return x.to(dtype)


class WanSelfAttention(nn.Cell):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        eps: float = 1e-6,
        dtype: Any = ms.float32,
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = mint.nn.Linear(dim, dim, dtype=dtype)
        self.k = mint.nn.Linear(dim, dim, dtype=dtype)
        self.v = mint.nn.Linear(dim, dim, dtype=dtype)
        self.o = mint.nn.Linear(dim, dim, dtype=dtype)
        self.norm_q = WanRMSNorm(dim, eps=eps, dtype=dtype) if qk_norm else mint.nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps, dtype=dtype) if qk_norm else mint.nn.Identity()

    def construct(self, x: ms.Tensor, seq_lens: ms.Tensor, grid_sizes: ms.Tensor, freqs: ms.Tensor) -> ms.Tensor:
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x: ms.Tensor) -> Tuple[ms.Tensor, ms.Tensor, ms.Tensor]:
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size,
        )

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanCrossAttention(WanSelfAttention):
    def construct(self, x: ms.Tensor, context: ms.Tensor, context_lens: Optional[ms.Tensor]) -> ms.Tensor:
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.shape[0], self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanAttentionBlock(nn.Cell):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        window_size: Tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        dtype: Any = ms.float32,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps, dtype=dtype)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps, dtype=dtype)
        self.norm3 = (
            WanLayerNorm(dim, eps, elementwise_affine=True, dtype=dtype) if cross_attn_norm else mint.nn.Identity()
        )
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps, dtype=dtype)
        self.norm2 = WanLayerNorm(dim, eps, dtype=dtype)
        self.ffn = nn.SequentialCell(
            mint.nn.Linear(dim, ffn_dim, dtype=dtype),
            mint.nn.GELU(approximate="tanh"),
            mint.nn.Linear(ffn_dim, dim, dtype=dtype),
        )

        # modulation
        self.modulation = ms.Parameter(ms.Tensor(np.random.randn(1, 6, dim) / dim**0.5, dtype=dtype))

    def construct(
        self,
        x: ms.Tensor,
        e: ms.Tensor,
        seq_lens: ms.Tensor,
        grid_sizes: ms.Tensor,
        freqs: ms.Tensor,
        context: ms.Tensor,
        context_lens: Optional[ms.Tensor],
    ) -> ms.Tensor:
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L1, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == ms.float32
        with autocast(dtype=ms.float32):
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        assert e[0].dtype == ms.float32

        # self-attention
        dtype = x.dtype
        y = self.self_attn(
            (self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2)).to(dtype), seq_lens, grid_sizes, freqs
        )
        with autocast(dtype=ms.float32):
            x = x + y * e[2].squeeze(2)
        x = x.to(dtype)

        # cross-attention & ffn function
        def cross_attn_ffn(x: ms.Tensor, context: ms.Tensor, context_lens: ms.Tensor, e: ms.Tensor) -> ms.Tensor:
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn((self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2)).to(dtype))
            with autocast(dtype=ms.float32):
                x = x + y * e[5].squeeze(2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x.to(dtype)


class Head(nn.Cell):
    def __init__(
        self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float = 1e-6, dtype: Any = ms.float32
    ):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps, dtype=dtype)
        self.head = mint.nn.Linear(dim, out_dim, dtype=dtype)

        # modulation
        self.modulation = ms.Parameter(ms.Tensor(np.random.randn(1, 2, dim) / dim**0.5, dtype=dtype))

    def construct(self, x: ms.Tensor, e: ms.Tensor) -> ms.Tensor:
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L1, C]
        """
        assert e.dtype == ms.float32
        with autocast(dtype=ms.float32):
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            x = self.head(self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2))
        return x


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = ["patch_size", "cross_attn_norm", "qk_norm", "text_dim", "window_size"]
    _no_split_modules = ["WanAttentionBlock"]

    @register_to_config
    def __init__(
        self,
        model_type: str = "t2v",
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        text_len: int = 512,
        in_dim: int = 16,
        dim: int = 2048,
        ffn_dim: int = 8192,
        freq_dim: int = 256,
        text_dim: int = 4096,
        out_dim: int = 16,
        num_heads: int = 16,
        num_layers: int = 32,
        window_size: Tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        dtype: Any = ms.float32,
    ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
            dtype (`mindspore.dtype`, *optional*, defaults to ms.float32):
                Data type for model parameters and computations
        """

        super().__init__()

        assert model_type in ["t2v", "i2v", "ti2v"]
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = mint.nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size, dtype=dtype)
        self.text_embedding = nn.SequentialCell(
            mint.nn.Linear(text_dim, dim, dtype=dtype),
            mint.nn.GELU(approximate="tanh"),
            mint.nn.Linear(dim, dim, dtype=dtype),
        )

        self.time_embedding = nn.SequentialCell(
            mint.nn.Linear(freq_dim, dim, dtype=dtype), mint.nn.SiLU(), mint.nn.Linear(dim, dim, dtype=dtype)
        )
        self.time_projection = nn.SequentialCell(mint.nn.SiLU(), mint.nn.Linear(dim, dim * 6, dtype=dtype))

        # blocks
        self.blocks = nn.CellList(
            [
                WanAttentionBlock(dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps, dtype=dtype)
                for _ in range(num_layers)
            ]
        )

        # head
        self.head = Head(dim, out_dim, patch_size, eps, dtype=dtype)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = mint.cat(
            [rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))],
            dim=1,
        )

        # initialize weights
        self.init_weights()

    def construct(
        self,
        x: List[ms.Tensor],
        t: ms.Tensor,
        context: List[ms.Tensor],
        seq_len: int,
        y: Optional[List[ms.Tensor]] = None,
    ) -> List[ms.Tensor]:
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == "i2v":
            assert y is not None

        if y is not None:
            x = [mint.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = mint.stack([ms.tensor(u.shape[2:], dtype=ms.int64) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = ms.tensor([u.shape[1] for u in x], dtype=ms.int64)
        assert seq_lens.max() <= seq_len
        x = mint.cat([mint.cat([u, u.new_zeros((1, seq_len - u.shape[1], u.shape[2]))], dim=1) for u in x])

        # time embeddings
        if len(t.shape) == 1:
            t = t.expand((t.shape[0], seq_len))
        with autocast(dtype=ms.float32):
            bt = t.shape[0]
            t = t.flatten()
            e = self.time_embedding(unflatten(sinusoidal_embedding_1d(self.freq_dim, t), 0, (bt, seq_len)).float())
            e0 = unflatten(self.time_projection(e), 2, (6, self.dim))
            assert e.dtype == ms.float32 and e0.dtype == ms.float32

        # context
        context_lens = None
        context = [u.to(self.dtype) for u in context]
        context = self.text_embedding(
            mint.stack([mint.cat([u, u.new_zeros((self.text_len - u.shape[0], u.shape[1]))]) for u in context])
        )

        # arguments
        kwargs = dict(
            e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes, freqs=self.freqs, context=context, context_lens=context_lens
        )

        x = x.to(self.dtype)
        for block in self.blocks:
            x = block(x, **kwargs)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def unpatchify(self, x: List[ms.Tensor], grid_sizes: ms.Tensor) -> List[ms.Tensor]:
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = mint.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self) -> None:
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for _, m in self.cells_and_names():
            if isinstance(m, mint.nn.Linear):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)

        # init embeddings
        patch_embedding_shape = self.patch_embedding.weight.shape
        patch_embedding_shape_flatten = (patch_embedding_shape[0], math.prod(patch_embedding_shape[1:]))
        data = initializer(
            XavierUniform(), patch_embedding_shape_flatten, self.patch_embedding.weight.dtype
        ).init_data()
        self.patch_embedding.weight.set_data(data.reshape(patch_embedding_shape))
        for _, m in self.text_embedding.cells_and_names():
            if isinstance(m, mint.nn.Linear):
                normal_(m.weight, std=0.02)
        for _, m in self.time_embedding.cells_and_names():
            if isinstance(m, mint.nn.Linear):
                normal_(m.weight, std=0.02)

        # init output layer
        zeros_(self.head.head.weight)
