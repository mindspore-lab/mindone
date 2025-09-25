# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
from typing import Any, Optional, Tuple

import mindspore as ms
import mindspore.mint as mint
import mindspore.ops as ops

from mindone.diffusers.models.layers_compat import unflatten

__all__ = ["flash_attention", "attention"]


def flash_attention(
    q: ms.Tensor,
    k: ms.Tensor,
    v: ms.Tensor,
    q_lens: Optional[ms.Tensor] = None,
    k_lens: Optional[ms.Tensor] = None,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    q_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    deterministic: bool = False,
    dtype: Any = ms.bfloat16,
    version: Optional[Any] = None,
) -> ms.Tensor:
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (ms.float16, ms.bfloat16)
    assert dtype in half_dtypes
    assert q.shape[-1] <= 256
    assert not causal
    assert window_size == (-1, -1)

    # params
    b, lq, lk, head_num, out_dtype = q.shape[0], q.shape[1], k.shape[1], q.shape[2], q.dtype

    def half(x: ms.Tensor) -> ms.Tensor:
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = ms.tensor([lq] * b, dtype=ms.int32)
    else:
        q = half(mint.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = ms.tensor([lk] * b, dtype=ms.int32)
    else:
        k = half(mint.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(mint.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])

    # apply attention
    x = ops.flash_attention_score(
        q,
        k,
        v,
        head_num,
        actual_seq_qlen=mint.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=ms.int32),
        actual_seq_kvlen=mint.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=ms.int32),
        scalar_value=softmax_scale,
        keep_prob=1.0 - dropout_p,
        input_layout="TND",
    )
    x = unflatten(x, 0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q: ms.Tensor,
    k: ms.Tensor,
    v: ms.Tensor,
    q_lens: Optional[ms.Tensor] = None,
    k_lens: Optional[ms.Tensor] = None,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    q_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    deterministic: bool = False,
    dtype: Any = ms.bfloat16,
    fa_version: Optional[Any] = None,
) -> ms.Tensor:
    return flash_attention(
        q=q,
        k=k,
        v=v,
        q_lens=q_lens,
        k_lens=k_lens,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        q_scale=q_scale,
        causal=causal,
        window_size=window_size,
        deterministic=deterministic,
        dtype=dtype,
        version=fa_version,
    )
