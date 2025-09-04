# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
from typing import Any, List, Optional, Tuple, Union

import numpy as np

import mindspore as ms
import mindspore.mint as mint
import mindspore.nn as nn
from mindspore.common.initializer import XavierUniform, initializer

from mindone.diffusers.loaders import PeftAdapterMixin
from mindone.models.utils import xavier_uniform_, zeros_

from ...utils.amp import autocast
from ..model import complex_mult, flash_attention
from .s2v_utils import rope_precompute


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


def rope_apply(
    x: ms.Tensor,
    grid_sizes: Union[List[ms.Tensor], ms.Tensor],
    freqs: Union[List[ms.Tensor], ms.Tensor],
    start: Optional[List[ms.Tensor]] = None,
) -> ms.Tensor:
    dtype = x.dtype
    x = x.to(ms.float32)
    n, c = x.shape[2], x.shape[3] // 2

    # split freqs
    if type(freqs) is list:
        trainable_freqs = freqs[1]
        freqs = freqs[0]
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    output = x.clone()
    seq_bucket = [0]
    if not type(grid_sizes) is list:
        grid_sizes = [grid_sizes]
    for g in grid_sizes:
        if not type(g) is list:
            g = [mint.zeros_like(g), g]
        batch_size = g[0].shape[0]
        for i in range(batch_size):
            if start is None:
                f_o, h_o, w_o = g[0][i]
            else:
                f_o, h_o, w_o = start[i]

            f, h, w = g[1][i]
            t_f, t_h, t_w = g[2][i]
            seq_f, seq_h, seq_w = f - f_o, h - h_o, w - w_o
            seq_len = int(seq_f * seq_h * seq_w)
            if seq_len > 0:
                if t_f > 0:
                    if f_o >= 0:
                        f_sam = np.linspace(f_o.item(), (t_f + f_o).item() - 1, seq_f).astype(int).tolist()
                    else:
                        f_sam = np.linspace(-f_o.item(), (-t_f - f_o).item() + 1, seq_f).astype(int).tolist()
                    h_sam = np.linspace(h_o.item(), (t_h + h_o).item() - 1, seq_h).astype(int).tolist()
                    w_sam = np.linspace(w_o.item(), (t_w + w_o).item() - 1, seq_w).astype(int).tolist()

                    assert f_o * f >= 0 and h_o * h >= 0 and w_o * w >= 0
                    if f_o >= 0:
                        freqs_0 = freqs[0][f_sam]
                    else:
                        freqs_0 = freqs[0][f_sam]
                        freqs_0[..., 1] = -freqs_0[..., 1]
                    freqs_0 = freqs_0.view(seq_f, 1, 1, -1, 2)

                    freqs_i = mint.cat(
                        [
                            freqs_0.expand(seq_f, seq_h, seq_w, -1, 2),
                            freqs[1][h_sam].view(1, seq_h, 1, -1).expand(seq_f, seq_h, seq_w, -1, 2),
                            freqs[2][w_sam].view(1, 1, seq_w, -1).expand(seq_f, seq_h, seq_w, -1, 2),
                        ],
                        dim=-2,
                    ).reshape(seq_len, 1, -1, 2)
                elif t_f < 0:
                    freqs_i = trainable_freqs.unsqueeze(1)
                # apply rotary embedding
                # precompute multipliers
                x_i = x[i, seq_bucket[-1] : seq_bucket[-1] + seq_len].to(ms.float32).reshape(seq_len, n, -1, 2)
                x_i = complex_mult(x_i, freqs_i).flatten(2)
                output[i, seq_bucket[-1] : seq_bucket[-1] + seq_len] = x_i
        seq_bucket.append(seq_bucket[-1] + seq_len)
    return output.to(dtype)


class RMSNorm(nn.Cell):
    def __init__(self, dim: int, eps: float = 1e-5, dtype: Any = ms.float32):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = ms.Parameter(mint.ones(dim, dtype=dtype))

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x: ms.Tensor) -> ms.Tensor:
        return x * mint.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class LayerNorm(mint.nn.LayerNorm):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False, dtype: Any = ms.float32):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps, dtype=dtype)

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        dtype = x.dtype
        with autocast(dtype=ms.float32):
            x = super().construct(x)
        return x.to(dtype)


class SelfAttention(nn.Cell):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: tuple = (-1, -1),
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
        self.norm_q = RMSNorm(dim, eps=eps, dtype=dtype) if qk_norm else mint.nn.Identity()
        self.norm_k = RMSNorm(dim, eps=eps, dtype=dtype) if qk_norm else mint.nn.Identity()

    def construct(
        self,
        x: ms.Tensor,
        seq_lens: ms.Tensor,
        grid_sizes: Union[List, ms.Tensor],
        freqs: Union[List, ms.Tensor],
    ) -> ms.Tensor:
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


class SwinSelfAttention(SelfAttention):
    def construct(
        self,
        x: ms.Tensor,
        seq_lens: ms.Tensor,
        grid_sizes: Union[List, ms.Tensor],
        freqs: Union[List, ms.Tensor],
    ) -> ms.Tensor:
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        assert b == 1, "Only support batch_size 1"

        # query, key, value function
        def qkv_fn(x: ms.Tensor) -> Tuple[ms.Tensor, ms.Tensor, ms.Tensor]:
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)
        T, H, W = grid_sizes[0].tolist()

        # b (t h w) n d -> (b t) (h w) n d
        q = q.reshape(q.shape[0], T, H, W, *q.shape[2:])
        q = q.reshape(-1, H * W, *q.shape[2:])
        # b (t h w) n d -> (b t) (h w) n d
        k = k.reshape(k.shape[0], T, H, W, *k.shape[2:])
        k = k.reshape(-1, H * W, *k.shape[2:])
        # b (t h w) n d -> (b t) (h w) n d
        v = v.reshape(v.shape[0], T, H, W, *v.shape[2:])
        v = v.reshape(-1, H * W, *v.shape[2:])

        q = q[:-1]

        # 1 s n d -> t s n d
        ref_k = mint.tile(k[-1:], (k.shape[0] - 1, 1, 1, 1))  # t hw n d
        k = k[:-1]
        k = mint.cat([k[:1], k, k[-1:]])
        k = mint.cat([k[1:-1], k[2:], k[:-2], ref_k], dim=1)  # (bt) (3hw) n d

        # 1 s n d -> t s n d
        ref_v = mint.tile(v[-1:], (v.shape[0] - 1, 1, 1, 1))
        v = v[:-1]
        v = mint.cat([v[:1], v, v[-1:]])
        v = mint.cat([v[1:-1], v[2:], v[:-2], ref_v], dim=1)

        # q: b (t h w) n d
        # k: b (t h w) n d
        out = flash_attention(
            q=q,
            k=k,
            v=v,
            window_size=self.window_size,
        )
        out = mint.cat([out, ref_v[:1]], axis=0)
        # (b t) (h w) n d -> b (t h w) n d
        out = out.reshape(out.shape[0], T, H, W, *out.shape[2:])
        out = out.reshape(out.shape[0], T * H * W, *out.shape[2:])
        x = out

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


# Fix the reference frame RoPE to 1,H,W.
# Set the current frame RoPE to 1.
# Set the previous frame RoPE to 0.
class CasualSelfAttention(SelfAttention):
    def construct(
        self,
        x: ms.Tensor,
        seq_lens: ms.Tensor,
        grid_sizes: Union[List[int], ms.Tensor],
        freqs: Union[List[ms.Tensor], ms.Tensor],
    ) -> ms.Tensor:
        shifting = 3
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        assert b == 1, "Only support batch_size 1"

        # query, key, value function
        def qkv_fn(x: ms.Tensor) -> Tuple[ms.Tensor, ms.Tensor, ms.Tensor]:
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        T, H, W = grid_sizes[0].tolist()

        # b (t h w) n d -> (b t) (h w) n d
        q = q.reshape(q.shape[0], T, H, W, *q.shape[2:])
        q = q.reshape(-1, H * W, *q.shape[2:])
        # b (t h w) n d -> (b t) (h w) n d
        k = k.reshape(k.shape[0], T, H, W, *k.shape[2:])
        k = k.reshape(-1, H * W, *k.shape[2:])
        # b (t h w) n d -> (b t) (h w) n d
        v = v.reshape(v.shape[0], T, H, W, *v.shape[2:])
        v = v.reshape(-1, H * W, *v.shape[2:])

        q = q[:-1]

        grid_sizes = ms.tensor([[1, H, W]] * q.shape[0], dtype=ms.int64)
        start = [[shifting, 0, 0]] * q.shape[0]
        q = rope_apply(q, grid_sizes, freqs, start=start)

        ref_k = k[-1:]
        grid_sizes = ms.tensor([[1, H, W]], dtype=ms.int64)
        # start = [[shifting, H, W]]

        start = [[shifting + 10, 0, 0]]
        ref_k = rope_apply(ref_k, grid_sizes, freqs, start)
        # 1 s n d -> t s n d"
        ref_k = mint.tile(ref_k, (k.shape[0] - 1, 1, 1, 1))  # t hw n d

        k = k[:-1]
        k = mint.cat([*([k[:1]] * shifting), k])
        cat_k = []
        for i in range(shifting):
            cat_k.append(k[i : i - shifting])
        cat_k.append(k[shifting:])
        k = mint.cat(cat_k, dim=1)  # (bt) (3hw) n d

        grid_sizes = ms.tensor([[shifting + 1, H, W]] * q.shape[0], dtype=ms.int64)
        k = rope_apply(k, grid_sizes, freqs)
        k = mint.cat([k, ref_k], dim=1)

        # 1 s n d -> t s n d
        ref_v = mint.tile(v[-1:], (q.shape[0], 1, 1, 1))  # t hw n d
        v = v[:-1]
        v = mint.cat([*([v[:1]] * shifting), v])
        cat_v = []
        for i in range(shifting):
            cat_v.append(v[i : i - shifting])
        cat_v.append(v[shifting:])
        v = mint.cat(cat_v, dim=1)  # (bt) (3hw) n d
        v = mint.cat([v, ref_v], dim=1)

        # q: b (t h w) n d
        # k: b (t h w) n d
        outs = []
        for i in range(q.shape[0]):
            out = flash_attention(q=q[i : i + 1], k=k[i : i + 1], v=v[i : i + 1], window_size=self.window_size)
            outs.append(out)
        out = mint.cat(outs, dim=0)
        out = mint.cat([out, ref_v[:1]], axis=0)
        # (b t) (h w) n d -> b (t h w) n d
        out = out.reshape(out.shape[0], T, H, W, *out.shape[2:])
        out = out.reshape(out.shape[0], T * H * W, *out.shape[2:])
        x = out

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class MotionerAttentionBlock(nn.Cell):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        window_size: tuple = (-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        self_attn_block: str = "SelfAttention",
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
        self.norm1 = LayerNorm(dim, eps, dtype=dtype)
        if self_attn_block == "SelfAttention":
            self.self_attn = SelfAttention(dim, num_heads, window_size, qk_norm, eps, dtype=dtype)
        elif self_attn_block == "SwinSelfAttention":
            self.self_attn = SwinSelfAttention(dim, num_heads, window_size, qk_norm, eps, dtype=dtype)
        elif self_attn_block == "CasualSelfAttention":
            self.self_attn = CasualSelfAttention(dim, num_heads, window_size, qk_norm, eps, dtype=dtype)

        self.norm2 = LayerNorm(dim, eps, dtype=dtype)
        self.ffn = nn.SequentialCell(
            mint.nn.Linear(dim, ffn_dim, dtype=dtype),
            mint.nn.GELU(approximate="tanh"),
            mint.nn.Linear(ffn_dim, dim, dtype=dtype),
        )

    def construct(
        self,
        x: ms.Tensor,
        seq_lens: ms.Tensor,
        grid_sizes: Union[List, ms.Tensor],
        freqs: Union[List, ms.Tensor],
    ) -> ms.Tensor:
        # self-attention
        y = self.self_attn(self.norm1(x), seq_lens, grid_sizes, freqs)
        x = x + y
        y = self.ffn(self.norm2(x))
        x = x + y
        return x


class Head(nn.Cell):
    def __init__(self, dim: int, out_dim: int, patch_size: tuple, eps: float = 1e-6, dtype: Any = ms.float32):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = LayerNorm(dim, eps, dtype=dtype)
        self.head = mint.nn.Linear(dim, out_dim, dtype=dtype)

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        x = self.head(self.norm(x))
        return x


class MotionerTransformers(nn.Cell, PeftAdapterMixin):
    def __init__(
        self,
        patch_size: tuple = (1, 2, 2),
        in_dim: int = 16,
        dim: int = 2048,
        ffn_dim: int = 8192,
        freq_dim: int = 256,
        out_dim: int = 16,
        num_heads: int = 16,
        num_layers: int = 32,
        window_size: tuple = (-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        self_attn_block: str = "SelfAttention",
        motion_token_num: int = 1024,
        enable_tsm: bool = False,
        motion_stride: int = 4,
        expand_ratio: int = 2,
        trainable_token_pos_emb: bool = False,
        dtype: Any = ms.float32,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.enable_tsm = enable_tsm
        self.motion_stride = motion_stride
        self.expand_ratio = expand_ratio
        self.sample_c = self.patch_size[0]

        # embeddings
        self.patch_embedding = mint.nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size, dtype=dtype)

        # blocks
        self.blocks = nn.CellList(
            [
                MotionerAttentionBlock(
                    dim,
                    ffn_dim,
                    num_heads,
                    window_size,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    self_attn_block=self_attn_block,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = mint.cat(
            [rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))],
            dim=1,
        )

        self.gradient_checkpointing = False

        self.motion_side_len = int(math.sqrt(motion_token_num))
        assert self.motion_side_len**2 == motion_token_num
        self.token = ms.Parameter(mint.zeros(1, motion_token_num, dim, dtype=dtype).contiguous())

        self.trainable_token_pos_emb = trainable_token_pos_emb
        if trainable_token_pos_emb:
            x = mint.zeros([1, motion_token_num, num_heads, d])
            x[..., ::2] = 1

            gride_sizes = [
                [
                    ms.tensor([0, 0, 0]).unsqueeze(0).repeat(1, 1),
                    ms.tensor([1, self.motion_side_len, self.motion_side_len]).unsqueeze(0).repeat(1, 1),
                    ms.tensor([1, self.motion_side_len, self.motion_side_len]).unsqueeze(0).repeat(1, 1),
                ]
            ]
            token_freqs = rope_apply(x, gride_sizes, self.freqs)
            token_freqs = token_freqs[0, :, 0].reshape(motion_token_num, -1, 2)
            token_freqs = token_freqs * 0.01
            self.token_freqs = ms.Parameter(token_freqs)

    def after_patch_embedding(self, x: List[ms.Tensor]) -> List[ms.Tensor]:
        return x

    def construct(
        self,
        x: List[ms.Tensor],
    ) -> ms.Tensor:
        """
        x:              A list of videos each with shape [C, T, H, W].
        t:              [B].
        context:        A list of text embeddings each with shape [L, C].
        """
        # params
        freqs = self.freqs

        if self.trainable_token_pos_emb:
            with autocast(dtype=ms.float32):
                token_freqs = self.token_freqs.to(ms.float32)
                token_freqs = token_freqs / token_freqs.norm(dim=-1, keepdim=True)
                freqs = [freqs, token_freqs]

        if self.enable_tsm:
            sample_idx = [
                sample_indices(u.shape[1], stride=self.motion_stride, expand_ratio=self.expand_ratio, c=self.sample_c)
                for u in x
            ]
            x = [mint.flip(mint.flip(u, [1])[:, idx], [1]) for idx, u in zip(sample_idx, x)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        x = self.after_patch_embedding(x)

        seq_f, seq_h, seq_w = x[0].shape[-3:]
        batch_size = len(x)
        if not self.enable_tsm:
            grid_sizes = mint.stack([ms.tensor(u.shape[2:], dtype=ms.int64) for u in x])
            grid_sizes = [[mint.zeros_like(grid_sizes), grid_sizes, grid_sizes]]
            seq_f = 0
        else:
            grid_sizes = []
            for idx in sample_idx[0][::-1][:: self.sample_c]:
                tsm_frame_grid_sizes = [
                    [
                        ms.tensor([idx, 0, 0]).unsqueeze(0).repeat(batch_size, 1),
                        ms.tensor([idx + 1, seq_h, seq_w]).unsqueeze(0).repeat(batch_size, 1),
                        ms.tensor([1, seq_h, seq_w]).unsqueeze(0).repeat(batch_size, 1),
                    ]
                ]
                grid_sizes += tsm_frame_grid_sizes
            seq_f = sample_idx[0][-1] + 1

        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = ms.tensor([u.size(1) for u in x], dtype=ms.int64)
        x = mint.cat([u for u in x])

        batch_size = len(x)

        token_grid_sizes = [
            [
                ms.tensor([seq_f, 0, 0]).unsqueeze(0).repeat(batch_size, 1),
                ms.tensor([seq_f + 1, self.motion_side_len, self.motion_side_len]).unsqueeze(0).repeat(batch_size, 1),
                ms.tensor([1 if not self.trainable_token_pos_emb else -1, seq_h, seq_w])
                .unsqueeze(0)
                .repeat(batch_size, 1),
            ]
        ]

        grid_sizes = grid_sizes + token_grid_sizes
        token_len = self.token.shape[1]
        token = self.token.clone().repeat(x.shape[0], 1, 1).contiguous()
        seq_lens = seq_lens + ms.tensor([t.size(0) for t in token], dtype=ms.int64)
        x = mint.cat([x, token], dim=1)
        # arguments
        kwargs = dict(
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=freqs,
        )

        for idx, block in enumerate(self.blocks):
            if self.training and self.gradient_checkpointing:
                x = ms.recompute(block, x, **kwargs)
            else:
                x = block(x, **kwargs)
        # head
        out = x[:, -token_len:]
        return out

    def unpatchify(self, x: ms.Tensor, grid_sizes: ms.Tensor) -> List[ms.Tensor]:
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = mint.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self) -> None:
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


class FramePackMotioner(nn.Cell):
    def __init__(
        self,
        inner_dim: int = 1024,
        num_heads: int = 16,
        zip_frame_buckets: List[int] = [1, 2, 16],
        drop_mode: str = "drop",
        dtype: Any = ms.float32,
    ):
        super().__init__()
        self.proj = mint.nn.Conv3d(16, inner_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2), dtype=dtype)
        self.proj_2x = mint.nn.Conv3d(16, inner_dim, kernel_size=(2, 4, 4), stride=(2, 4, 4), dtype=dtype)
        self.proj_4x = mint.nn.Conv3d(16, inner_dim, kernel_size=(4, 8, 8), stride=(4, 8, 8), dtype=dtype)
        self.zip_frame_buckets = ms.tensor(zip_frame_buckets, dtype=ms.int64)

        self.inner_dim = inner_dim
        self.num_heads = num_heads

        assert (inner_dim % num_heads) == 0 and (inner_dim // num_heads) % 2 == 0
        d = inner_dim // num_heads
        self.freqs = mint.cat(
            [rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))],
            dim=1,
        )
        self.drop_mode = drop_mode

    def construct(self, motion_latents: List[ms.Tensor], add_last_motion: int = 2) -> tuple:
        mot = []
        mot_remb = []
        for m in motion_latents:
            lat_height, lat_width = m.shape[2], m.shape[3]
            padd_lat = mint.zeros(16, self.zip_frame_buckets.sum(), lat_height, lat_width).to(dtype=m.dtype)
            overlap_frame = min(padd_lat.shape[1], m.shape[1])
            if overlap_frame > 0:
                padd_lat[:, -overlap_frame:] = m[:, -overlap_frame:]

            if add_last_motion < 2 and self.drop_mode != "drop":
                zero_end_frame = self.zip_frame_buckets[: self.zip_frame_buckets.__len__() - add_last_motion - 1].sum()
                padd_lat[:, -zero_end_frame:] = 0

            padd_lat = padd_lat.unsqueeze(0)
            clean_latents_4x, clean_latents_2x, clean_latents_post = padd_lat[
                :, :, -self.zip_frame_buckets.sum() :, :, :
            ].split(
                list(self.zip_frame_buckets)[::-1], dim=2
            )  # 16, 2 ,1

            # patchfy
            clean_latents_post = self.proj(clean_latents_post).flatten(2).transpose(1, 2)
            clean_latents_2x = self.proj_2x(clean_latents_2x).flatten(2).transpose(1, 2)
            clean_latents_4x = self.proj_4x(clean_latents_4x).flatten(2).transpose(1, 2)

            if add_last_motion < 2 and self.drop_mode == "drop":
                clean_latents_post = clean_latents_post[:, :0] if add_last_motion < 2 else clean_latents_post
                clean_latents_2x = clean_latents_2x[:, :0] if add_last_motion < 1 else clean_latents_2x

            motion_lat = mint.cat([clean_latents_post, clean_latents_2x, clean_latents_4x], dim=1)

            # rope
            start_time_id = -(self.zip_frame_buckets[:1].sum())
            end_time_id = start_time_id + self.zip_frame_buckets[0]
            grid_sizes = (
                []
                if add_last_motion < 2 and self.drop_mode == "drop"
                else [
                    [
                        ms.tensor([start_time_id, 0, 0]).unsqueeze(0).repeat(1, 1),
                        ms.tensor([end_time_id, lat_height // 2, lat_width // 2]).unsqueeze(0).repeat(1, 1),
                        ms.tensor([self.zip_frame_buckets[0], lat_height // 2, lat_width // 2])
                        .unsqueeze(0)
                        .repeat(1, 1),
                    ]
                ]
            )

            start_time_id = -(self.zip_frame_buckets[:2].sum())
            end_time_id = start_time_id + self.zip_frame_buckets[1] // 2
            grid_sizes_2x = (
                []
                if add_last_motion < 1 and self.drop_mode == "drop"
                else [
                    [
                        ms.tensor([start_time_id, 0, 0]).unsqueeze(0).repeat(1, 1),
                        ms.tensor([end_time_id, lat_height // 4, lat_width // 4]).unsqueeze(0).repeat(1, 1),
                        ms.tensor([self.zip_frame_buckets[1], lat_height // 2, lat_width // 2])
                        .unsqueeze(0)
                        .repeat(1, 1),
                    ]
                ]
            )

            start_time_id = -(self.zip_frame_buckets[:3].sum())
            end_time_id = start_time_id + self.zip_frame_buckets[2] // 4
            grid_sizes_4x = [
                [
                    ms.tensor([start_time_id, 0, 0]).unsqueeze(0).repeat(1, 1),
                    ms.tensor([end_time_id, lat_height // 8, lat_width // 8]).unsqueeze(0).repeat(1, 1),
                    ms.tensor([self.zip_frame_buckets[2], lat_height // 2, lat_width // 2]).unsqueeze(0).repeat(1, 1),
                ]
            ]

            grid_sizes = grid_sizes + grid_sizes_2x + grid_sizes_4x

            motion_rope_emb = rope_precompute(
                motion_lat.view(1, motion_lat.shape[1], self.num_heads, self.inner_dim // self.num_heads),
                grid_sizes,
                self.freqs,
                start=None,
            )

            mot.append(motion_lat)
            mot_remb.append(motion_rope_emb)
        return mot, mot_remb


def sample_indices(N: int, stride: int, expand_ratio: int, c: int) -> List[int]:
    indices = []
    current_start = 0

    while current_start < N:
        bucket_width = int(stride * (expand_ratio ** (len(indices) / stride)))

        interval = int(bucket_width / stride * c)
        current_end = min(N, current_start + bucket_width)
        bucket_samples = []
        for i in range(current_end - 1, current_start - 1, -interval):
            for near in range(c):
                bucket_samples.append(i - near)

        indices += bucket_samples[::-1]
        current_start += bucket_width

    return indices
