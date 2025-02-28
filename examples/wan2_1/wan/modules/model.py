# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
from typing import List, Optional, Tuple

import numpy as np

import mindspore as ms
import mindspore.mint as mint
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, Tensor

from mindone.diffusers.configuration_utils import ConfigMixin, register_to_config
from mindone.diffusers.models.modeling_utils import ModelMixin
from mindone.models.utils import normal_, xavier_uniform_, zeros_

__all__ = ["WanModel"]


def sinusoidal_embedding_1d(dim: int, position: Tensor) -> Tensor:
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(ms.float32)

    # calculation
    sinusoid = mint.outer(position, mint.pow(10000, -mint.arange(half).to(position.dtype).div(half)))
    x = mint.cat([mint.cos(sinusoid), mint.sin(sinusoid)], dim=1)
    return x


def rope_params(max_seq_len: int, dim: int, theta: float = 10000) -> Tensor:
    assert dim % 2 == 0
    freqs = mint.outer(mint.arange(max_seq_len), 1.0 / mint.pow(theta, mint.arange(0, dim, 2).to(ms.float32).div(dim)))
    freqs = mint.stack([mint.cos(freqs), mint.sin(freqs)], dim=-1)
    return freqs


def complex_mult(a: Tensor, b: Tensor) -> Tensor:
    a_real, a_complex = a[..., 0], a[..., 1]
    b_real, b_complex = b[..., 0], b[..., 1]
    out_real = a_real * b_real - a_complex * b_complex
    out_complex = a_real * b_complex + b_real * a_complex
    return mint.stack([out_real, out_complex], dim=-1)


def rope_apply(x: Tensor, grid_sizes: Tensor, freqs: Tensor) -> Tensor:
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
        x_i = mint.cat([x_i.to(x.dtype), x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return mint.stack(output)


class WanRMSNorm(nn.Cell):
    def __init__(self, dim: int, eps: float = 1e-5, dtype: ms.Type = ms.float32) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(Tensor(np.ones(dim), dtype=dtype))

    def construct(self, x: Tensor) -> Tensor:
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x: Tensor) -> Tensor:
        return x * mint.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(mint.nn.LayerNorm):
    def __init__(
        self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False, dtype: ms.Type = ms.float32
    ) -> None:
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps, dtype=dtype)

    def construct(self, x: Tensor) -> Tensor:
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        # TODO: to float32
        return super().construct(x).type_as(x)


class WanSelfAttention(nn.Cell):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        eps=1e-6,
        dtype: ms.Type = ms.float32,
    ) -> None:
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

    def construct(self, x: Tensor, seq_lens: Tensor, grid_sizes: Tensor, freqs: Tensor) -> Tensor:
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        assert self.window_size == (-1, -1)

        x = ops.flash_attention_score(
            query=rope_apply(q, grid_sizes, freqs),
            key=rope_apply(k, grid_sizes, freqs),
            value=v,
            head_num=self.num_heads,
            actual_seq_kvlen=seq_lens,
            scalar_value=1 / math.sqrt(q.shape[-1]),
            input_layout="BSND",
        )

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):
    def construct(self, x: Tensor, context: Tensor, context_lens: Tensor) -> Tensor:
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
        x = ops.flash_attention_score(
            q,
            k,
            v,
            head_num=self.num_heads,
            actual_seq_kvlen=context_lens,
            scalar_value=1 / math.sqrt(q.shape[-1]),
            input_layout="BSND",
        )

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        eps: float = 1e-6,
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__(dim, num_heads, window_size, qk_norm, eps, dtype=dtype)

        self.k_img = mint.nn.Linear(dim, dim, dtype=dtype)
        self.v_img = mint.nn.Linear(dim, dim, dtype=dtype)
        self.norm_k_img = WanRMSNorm(dim, eps=eps, dtype=dtype) if qk_norm else mint.nn.Identity()

    def construct(self, x: Tensor, context: Tensor, context_lens: Tensor) -> Tensor:
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.shape[0], self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = ops.flash_attention_score(
            q,
            k_img,
            v_img,
            head_num=self.num_heads,
            scalar_value=1 / math.sqrt(q.shape[-1]),
            input_layout="BSND",
        )
        # compute attention
        x = ops.flash_attention_score(
            q,
            k,
            v,
            head_num=self.num_heads,
            actual_seq_kvlen=context_lens,
            scalar_value=1 / math.sqrt(q.shape[-1]),
            input_layout="BSND",
        )

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    "t2v_cross_attn": WanT2VCrossAttention,
    "i2v_cross_attn": WanI2VCrossAttention,
}


class WanAttentionBlock(nn.Cell):
    def __init__(
        self,
        cross_attn_type: str,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        window_size: Tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        dtype: ms.Type = ms.float32,
    ) -> None:
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
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](
            dim, num_heads, (-1, -1), qk_norm, eps, dtype=dtype
        )
        self.norm2 = WanLayerNorm(dim, eps, dtype=dtype)
        # TODO: mint.nn.GELU -> mint.nn.GELU(approximate="tanh")
        self.ffn = nn.SequentialCell(
            mint.nn.Linear(dim, ffn_dim, dtype=dtype),
            mint.nn.GELU(),
            mint.nn.Linear(ffn_dim, dim, dtype=dtype),
        )

        # modulation
        self.modulation = Parameter(Tensor(np.random.randn(1, 6, dim) / dim**0.5, dtype=dtype))

    def construct(
        self,
        x: Tensor,
        e: Tensor,
        seq_lens: Tensor,
        grid_sizes: Tensor,
        freqs: Tensor,
        context: Tensor,
        context_lens: Tensor,
    ) -> Tensor:
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        e = (self.modulation + e).chunk(6, dim=1)

        # self-attention
        y = self.self_attn(self.norm1(x) * (1 + e[1]) + e[0], seq_lens, grid_sizes, freqs)
        x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x: Tensor, context: Tensor, context_lens: Tensor, e: Tensor) -> Tensor:
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
            x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Head(nn.Cell):
    def __init__(self, dim: int, out_dim: int, patch_size: int, eps: float = 1e-6, dtype: ms.Type = ms.float32) -> None:
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
        self.modulation = Parameter(Tensor(np.random.randn(1, 2, dim) / dim**0.5, dtype=dtype))

    def construct(self, x: Tensor, e: Tensor) -> Tensor:
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        x = self.head(self.norm(x) * (1 + e[1]) + e[0])
        return x


class MLPProj(nn.Cell):
    def __init__(self, in_dim: int, out_dim: int, dtype: ms.Type = ms.float32) -> None:
        super().__init__()

        self.proj = nn.SequentialCell(
            mint.nn.LayerNorm(in_dim, dtype=dtype),
            mint.nn.Linear(in_dim, in_dim, dtype=dtype),
            mint.nn.GELU(),
            mint.nn.Linear(in_dim, out_dim, dtype=dtype),
            mint.nn.LayerNorm(out_dim, dtype=dtype),
        )

    def construct(self, image_embeds: Tensor) -> Tensor:
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


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
        dtype: ms.Type = ms.float32,
    ) -> None:
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
        """

        super().__init__()

        assert model_type in ["t2v", "i2v"]
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
        # TODO: mint.nn.GELU -> mint.nn.GELU(approximate="tanh")
        self.text_embedding = nn.SequentialCell(
            mint.nn.Linear(text_dim, dim, dtype=dtype),
            mint.nn.GELU(),
            mint.nn.Linear(dim, dim, dtype=dtype),
        )

        self.time_embedding = nn.SequentialCell(
            mint.nn.Linear(freq_dim, dim, dtype=dtype), mint.nn.SiLU(), mint.nn.Linear(dim, dim, dtype=dtype)
        )
        self.time_projection = nn.SequentialCell(mint.nn.SiLU(), mint.nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = "t2v_cross_attn" if model_type == "t2v" else "i2v_cross_attn"
        self.blocks = nn.CellList(
            [
                WanAttentionBlock(
                    cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps, dtype=dtype
                )
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

        if model_type == "i2v":
            self.img_emb = MLPProj(1280, dim, dtype=dtype)

        # initialize weights
        self.init_weights()

    def construct(
        self,
        x: List[Tensor],
        t: Tensor,
        context: List[Tensor],
        seq_len: int,
        clip_fea: Optional[Tensor] = None,
        y: List[Tensor] = None,
    ) -> List[Tensor]:
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
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == "i2v":
            assert clip_fea is not None and y is not None
        # params
        if y is not None:
            x = [mint.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = mint.stack([Tensor(u.shape[2:], dtype=ms.int32) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = Tensor([u.shape[1] for u in x], dtype=ms.int32)
        assert seq_lens.max() <= seq_len
        x = mint.cat([mint.cat([u, u.new_zeros((1, seq_len - u.shape[1], u.shape[2]))], dim=1) for u in x])

        # time embeddings
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).to(self.dtype))
        e0 = self.time_projection(e)
        # TODO: reshape -> unflatten
        e0 = e0.reshape(e0.shape[0], 6, self.dim, *e.shape[2:])

        # context
        context_lens = None
        context = self.text_embedding(
            mint.stack([mint.cat([u, u.new_zeros((self.text_len - u.shape[0], u.shape[1]))]) for u in context])
        )

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = mint.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes, freqs=self.freqs, context=context, context_lens=context_lens
        )

        for block in self.blocks:
            x = block(x, **kwargs)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def unpatchify(self, x: Tensor, grid_sizes: Tensor) -> List[Tensor]:
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
        xavier_uniform_(self.patch_embedding.weight)
        for _, m in self.text_embedding.cells_and_names():
            if isinstance(m, mint.nn.Linear):
                normal_(m.weight, std=0.02)
        for _, m in self.time_embedding.cells_and_names():
            if isinstance(m, mint.nn.Linear):
                normal_(m.weight, std=0.02)

        # init output layer
        zeros_(self.head.head.weight)
