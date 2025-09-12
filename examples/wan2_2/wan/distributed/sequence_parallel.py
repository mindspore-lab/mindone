# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from typing import Any, List, Optional, Tuple

import mindspore as ms
import mindspore.mint as mint

from mindone.diffusers.models.layers_compat import unflatten

from ..modules.model import WanModel, WanSelfAttention, complex_mult, sinusoidal_embedding_1d
from ..utils.amp import autocast
from .ulysses import distributed_attention
from .util import gather_forward, get_rank, get_world_size


def pad_freqs(original_tensor: ms.Tensor, target_len: int) -> ms.Tensor:
    seq_len, s1, s2, _ = original_tensor.shape
    pad_size = target_len - seq_len
    padding_tensor = mint.ones((pad_size, s1, s2, 2), dtype=original_tensor.dtype)
    padded_tensor = mint.cat([original_tensor, padding_tensor], dim=0)
    return padded_tensor


def rope_apply(x: ms.Tensor, grid_sizes: ms.Tensor, freqs: ms.Tensor) -> ms.Tensor:
    """
    x:          [B, L, N, C].
    grid_sizes: [B, 3].
    freqs:      [M, C // 2].
    """
    dtype = x.dtype
    x = x.to(ms.float32)
    s, n, c = x.shape[1], x.shape[2], x.shape[3] // 2
    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = x[i, :seq_len].to(ms.float32).reshape(s, n, -1, 2)
        freqs_i = mint.cat(
            [
                freqs[0][:f].view(f, 1, 1, -1, 2).expand((f, h, w, -1, 2)),
                freqs[1][:h].view(1, h, 1, -1, 2).expand((f, h, w, -1, 2)),
                freqs[2][:w].view(1, 1, w, -1, 2).expand((f, h, w, -1, 2)),
            ],
            dim=-2,
        ).reshape(seq_len, 1, -1, 2)

        # apply rotary embedding
        sp_size = get_world_size()
        sp_rank = get_rank()
        freqs_i = pad_freqs(freqs_i, s * sp_size)
        s_per_rank = s
        freqs_i_rank = freqs_i[(sp_rank * s_per_rank) : ((sp_rank + 1) * s_per_rank), :, :, :]
        x_i = complex_mult(x_i, freqs_i_rank).flatten(2)
        x_i = mint.cat([x_i, x[i, s:]])

        # append to collection
        output.append(x_i)
    return mint.stack(output).to(dtype)


def sp_dit_forward(
    self: WanModel,
    x: List[ms.Tensor],
    t: ms.Tensor,
    context: List[ms.Tensor],
    seq_len: int,
    y: Optional[List[ms.Tensor]] = None,
) -> List[ms.Tensor]:
    """
    x:              A list of videos each with shape [C, T, H, W].
    t:              [B].
    context:        A list of text embeddings each with shape [L, C].
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

    # Context Parallel
    x = mint.chunk(x, get_world_size(), dim=1)[get_rank()]
    e = mint.chunk(e, get_world_size(), dim=1)[get_rank()]
    e0 = mint.chunk(e0, get_world_size(), dim=1)[get_rank()]

    # arguments
    kwargs = dict(
        e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes, freqs=self.freqs, context=context, context_lens=context_lens
    )

    x = x.to(self.dtype)
    for block in self.blocks:
        x = block(x, **kwargs)

    # head
    x = self.head(x, e)

    # Context Parallel
    x = gather_forward(x, dim=1)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return [u.float() for u in x]


def sp_attn_forward(
    self: WanSelfAttention,
    x: ms.Tensor,
    seq_lens: ms.Tensor,
    grid_sizes: ms.Tensor,
    freqs: ms.Tensor,
    dtype: Any = ms.bfloat16,
) -> ms.Tensor:
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    half_dtypes = (ms.float16, ms.bfloat16)

    def half(x: ms.Tensor) -> ms.Tensor:
        return x if x.dtype in half_dtypes else x.to(dtype)

    # query, key, value function
    def qkv_fn(x: ms.Tensor) -> Tuple[ms.Tensor, ms.Tensor, ms.Tensor]:
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)
    q = rope_apply(q, grid_sizes, freqs)
    k = rope_apply(k, grid_sizes, freqs)

    x = distributed_attention(
        half(q),
        half(k),
        half(v),
        seq_lens,
        window_size=self.window_size,
    ).to(q.dtype)

    # output
    x = x.flatten(2)
    x = self.o(x)
    return x
