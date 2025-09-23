# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from typing import Tuple

import mindspore as ms
import mindspore.mint.distributed as dist

from ..modules.attention import flash_attention
from .util import all_to_all


def distributed_attention(
    q: ms.Tensor, k: ms.Tensor, v: ms.Tensor, seq_lens: ms.Tensor, window_size: Tuple[int, int] = (-1, -1)
) -> ms.Tensor:
    """
    Performs distributed attention based on DeepSpeed Ulysses attention mechanism.
    please refer to https://arxiv.org/pdf/2309.14509

    Args:
        q:           [B, Lq // p, Nq, C1].
        k:           [B, Lk // p, Nk, C1].
        v:           [B, Lk // p, Nk, C2]. Nq must be divisible by Nk.
        seq_lens:    [B], length of each sequence in batch
        window_size: (left right). If not (-1, -1), apply sliding window local attention.
    """
    if not dist.is_initialized():
        raise ValueError("distributed group should be initialized.")

    # gather q/k/v sequence
    q = all_to_all(q, scatter_dim=2, gather_dim=1)
    k = all_to_all(k, scatter_dim=2, gather_dim=1)
    v = all_to_all(v, scatter_dim=2, gather_dim=1)

    # apply attention
    x = flash_attention(q, k, v, k_lens=seq_lens, window_size=window_size)

    # scatter q/k/v sequence
    x = all_to_all(x, scatter_dim=1, gather_dim=2)
    return x
