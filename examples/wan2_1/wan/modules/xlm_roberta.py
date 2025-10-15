# This code is adapted from https://github.com/Wan-Video/Wan2.1
# with modifications to run on MindSpore.

# Modified from transformers.models.xlm_roberta.modeling_xlm_roberta
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import mindspore as ms
import mindspore.mint as mint
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

from mindone.transformers.modeling_attn_mask_utils import dtype_to_min

__all__ = ["XLMRoberta", "xlm_roberta_large"]


class SelfAttention(nn.Cell):
    def __init__(
        self, dim: int, num_heads: int, dropout: float = 0.1, eps: float = 1e-5, dtype: ms.Type = ms.float32
    ) -> None:
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = eps

        # layers
        self.q = mint.nn.Linear(dim, dim, dtype=dtype)
        self.k = mint.nn.Linear(dim, dim, dtype=dtype)
        self.v = mint.nn.Linear(dim, dim, dtype=dtype)
        self.o = mint.nn.Linear(dim, dim, dtype=dtype)
        self.dropout = mint.nn.Dropout(dropout)

    def construct(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        x:   [B, L, C].
        """
        b, s, c, n, d = *x.shape, self.num_heads, self.head_dim

        # compute query, key, value
        q = self.q(x).reshape(b, s, n, d).permute(0, 2, 1, 3)
        k = self.k(x).reshape(b, s, n, d).permute(0, 2, 1, 3)
        v = self.v(x).reshape(b, s, n, d).permute(0, 2, 1, 3)

        # compute attention
        p = self.dropout.p if self.training else 0.0
        # TODO: check mask
        x = ops.flash_attention_score(q, k, v, self.num_heads, attn_mask=mask, keep_prob=1 - p)
        x = x.permute(0, 2, 1, 3).reshape(b, s, c)

        # output
        x = self.o(x)
        x = self.dropout(x)
        return x


class AttentionBlock(nn.Cell):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        post_norm: bool,
        dropout: float = 0.1,
        eps: float = 1e-5,
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.post_norm = post_norm
        self.eps = eps

        # layers
        self.attn = SelfAttention(dim, num_heads, dropout, eps, dtype=dtype)
        self.norm1 = mint.nn.LayerNorm(dim, eps=eps, dtype=dtype)
        self.ffn = nn.SequentialCell(
            mint.nn.Linear(dim, dim * 4, dtype=dtype),
            mint.nn.GELU(),
            mint.nn.Linear(dim * 4, dim, dtype=dtype),
            mint.nn.Dropout(dropout),
        )
        self.norm2 = mint.nn.LayerNorm(dim, eps=eps, dtype=dtype)

    def construct(self, x: Tensor, mask: Tensor) -> Tensor:
        if self.post_norm:
            x = self.norm1(x + self.attn(x, mask))
            x = self.norm2(x + self.ffn(x))
        else:
            x = x + self.attn(self.norm1(x), mask)
            x = x + self.ffn(self.norm2(x))
        return x


class XLMRoberta(nn.Cell):
    """
    XLMRobertaModel with no pooler and no LM head.
    """

    def __init__(
        self,
        vocab_size: int = 250002,
        max_seq_len: int = 514,
        type_size: int = 1,
        pad_id: int = 1,
        dim: int = 1024,
        num_heads: int = 16,
        num_layers: int = 24,
        post_norm: bool = True,
        dropout: float = 0.1,
        eps: float = 1e-5,
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.type_size = type_size
        self.pad_id = pad_id
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.post_norm = post_norm
        self.eps = eps

        # embeddings
        self.token_embedding = mint.nn.Embedding(vocab_size, dim, padding_idx=pad_id, dtype=dtype)
        self.type_embedding = mint.nn.Embedding(type_size, dim, dtype=dtype)
        self.pos_embedding = mint.nn.Embedding(max_seq_len, dim, padding_idx=pad_id, dtype=dtype)
        self.dropout = mint.nn.Dropout(dropout)

        # blocks
        self.blocks = nn.CellList(
            [AttentionBlock(dim, num_heads, post_norm, dropout, eps, dtype=dtype) for _ in range(num_layers)]
        )

        # norm layer
        self.norm = mint.nn.LayerNorm(dim, eps=eps, dtype=dtype)

    def construct(self, ids: Tensor) -> Tensor:
        """
        ids: [B, L] of mindspore.Tensor.
        """
        b, s = ids.shape
        mask = ids.ne(self.pad_id).to(ms.int32)

        # embeddings
        x = (
            self.token_embedding(ids)
            + self.type_embedding(mint.zeros_like(ids))
            + self.pos_embedding(self.pad_id + mint.cumsum(mask, dim=1) * mask)
        )
        if self.post_norm:
            x = self.norm(x)
        x = self.dropout(x)

        # blocks
        mask = mint.where(mask.view(b, 1, 1, s).gt(0), 0.0, dtype_to_min(x.dtype))
        for block in self.blocks:
            x = block(x, mask)

        # output
        if not self.post_norm:
            x = self.norm(x)
        return x


def xlm_roberta_large(pretrained: bool = False, return_tokenizer: bool = False, dtype: ms.Type = ms.float32, **kwargs):
    """
    XLMRobertaLarge adapted from Huggingface.
    """
    # params
    cfg = dict(
        vocab_size=250002,
        max_seq_len=514,
        type_size=1,
        pad_id=1,
        dim=1024,
        num_heads=16,
        num_layers=24,
        post_norm=True,
        dropout=0.1,
        eps=1e-5,
    )
    cfg.update(**kwargs)

    model = XLMRoberta(**cfg, dtype=dtype)
    return model
