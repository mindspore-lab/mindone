# Modified from transformers.models.t5.modeling_t5
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging
import math
from typing import Any, Callable, Dict, Optional, Union

import numpy as np

import mindspore as ms
import mindspore.mint as mint
import mindspore.mint.nn.functional as F
import mindspore.nn as nn
from mindspore import Parameter, Tensor
from mindspore.nn.utils import no_init_parameters

from mindone.models.utils import normal_, ones_
from mindone.transformers.modeling_attn_mask_utils import dtype_to_min

from ..utils.utils import load_pth
from .tokenizers import HuggingfaceTokenizer

__all__ = ["T5Model", "T5Encoder", "T5Decoder", "T5EncoderModel"]


def fp16_clamp(x: Tensor) -> Tensor:
    if x.dtype == ms.float16 and mint.isinf(x).any():
        clamp = Tensor(np.finfo(np.float16).max - 1000)
        x = mint.clamp(x, min=-clamp, max=clamp)
    return x


def init_weights(m: Any) -> None:
    if isinstance(m, T5LayerNorm):
        ones_(m.weight)
    elif isinstance(m, T5Model):
        normal_(m.token_embedding.weight, std=1.0)
    elif isinstance(m, T5FeedForward):
        normal_(m.gate[0].weight, std=m.dim**-0.5)
        normal_(m.fc1.weight, std=m.dim**-0.5)
        normal_(m.fc2.weight, std=m.dim_ffn**-0.5)
    elif isinstance(m, T5Attention):
        normal_(m.q.weight, std=(m.dim * m.dim_attn) ** -0.5)
        normal_(m.k.weight, std=m.dim**-0.5)
        normal_(m.v.weight, std=m.dim**-0.5)
        normal_(m.o.weight, std=(m.num_heads * m.dim_attn) ** -0.5)
    elif isinstance(m, T5RelativeEmbedding):
        normal_(m.embedding.weight, std=(2 * m.num_buckets * m.num_heads) ** -0.5)


class GELU(nn.Cell):
    def construct(self, x: Tensor) -> Tensor:
        return 0.5 * x * (1.0 + mint.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * mint.pow(x, 3.0))))


class T5LayerNorm(nn.Cell):
    def __init__(self, dim: int, eps: float = 1e-6, dtype: ms.Type = ms.float32) -> None:
        super(T5LayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(Tensor(np.ones(dim), dtype=dtype))

    def construct(self, x: Tensor) -> Tensor:
        x = x * mint.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        if self.weight.dtype in [ms.float16, ms.bfloat16]:
            x = x.type_as(self.weight)
        return self.weight * x


class T5Attention(nn.Cell):
    def __init__(
        self, dim: int, dim_attn: int, num_heads: int, dropout: float = 0.1, dtype: ms.Type = ms.float32
    ) -> None:
        assert dim_attn % num_heads == 0
        super(T5Attention, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.num_heads = num_heads
        self.head_dim = dim_attn // num_heads

        # layers
        self.q = mint.nn.Linear(dim, dim_attn, bias=False, dtype=dtype)
        self.k = mint.nn.Linear(dim, dim_attn, bias=False, dtype=dtype)
        self.v = mint.nn.Linear(dim, dim_attn, bias=False, dtype=dtype)
        self.o = mint.nn.Linear(dim_attn, dim, bias=False, dtype=dtype)
        self.dropout = mint.nn.Dropout(dropout)

    def construct(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        pos_bias: Optional[Tensor] = None,
    ) -> Tensor:
        """
        x:          [B, L1, C].
        context:    [B, L2, C] or None.
        mask:       [B, L2] or [B, L1, L2] or None.
        """
        # check inputs
        context = x if context is None else context
        b, n, c = x.shape[0], self.num_heads, self.head_dim

        # compute query, key, value
        q = self.q(x).view(b, -1, n, c)
        k = self.k(context).view(b, -1, n, c)
        v = self.v(context).view(b, -1, n, c)

        # attention bias
        attn_bias = x.new_zeros((b, n, q.shape[1], k.shape[1]))
        if pos_bias is not None:
            attn_bias += pos_bias
        if mask is not None:
            assert mask.ndim in [2, 3]
            mask = mask.view(b, 1, 1, -1) if mask.ndim == 2 else mask.unsqueeze(1)
            attn_bias.masked_fill_(mask == 0, dtype_to_min(x.dtype))

        # compute attention (T5 does not use scaling)
        attn = mint.einsum("binc,bjnc->bnij", q, k) + attn_bias
        attn = F.softmax(attn.float(), dim=-1).type_as(attn)
        x = mint.einsum("bnij,bjnc->binc", attn, v)

        # output
        x = x.reshape(b, -1, n * c)
        x = self.o(x)
        x = self.dropout(x)
        return x


class T5FeedForward(nn.Cell):
    def __init__(self, dim: int, dim_ffn: int, dropout: float = 0.1, dtype: ms.Type = ms.float32) -> None:
        super(T5FeedForward, self).__init__()
        self.dim = dim
        self.dim_ffn = dim_ffn

        # layers
        self.gate = nn.SequentialCell(mint.nn.Linear(dim, dim_ffn, bias=False, dtype=dtype), GELU())
        self.fc1 = mint.nn.Linear(dim, dim_ffn, bias=False, dtype=dtype)
        self.fc2 = mint.nn.Linear(dim_ffn, dim, bias=False, dtype=dtype)
        self.dropout = mint.nn.Dropout(dropout)

    def construct(self, x: Tensor) -> Tensor:
        x = self.fc1(x) * self.gate(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class T5SelfAttention(nn.Cell):
    def __init__(
        self,
        dim: int,
        dim_attn: int,
        dim_ffn: int,
        num_heads: int,
        num_buckets: int,
        shared_pos: bool = True,
        dropout: float = 0.1,
        dtype: ms.Type = ms.float32,
    ) -> None:
        super(T5SelfAttention, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.norm1 = T5LayerNorm(dim, dtype=dtype)
        self.attn = T5Attention(dim, dim_attn, num_heads, dropout, dtype=dtype)
        self.norm2 = T5LayerNorm(dim, dtype=dtype)
        self.ffn = T5FeedForward(dim, dim_ffn, dropout, dtype=dtype)
        self.pos_embedding = (
            None if shared_pos else T5RelativeEmbedding(num_buckets, num_heads, bidirectional=True, dtype=dtype)
        )

    def construct(self, x: Tensor, mask: Optional[Tensor] = None, pos_bias: Optional[Tensor] = None) -> Tensor:
        e = pos_bias if self.shared_pos else self.pos_embedding(x.shape[1], x.shape[1])
        x = fp16_clamp(x + self.attn(self.norm1(x), mask=mask, pos_bias=e))
        x = fp16_clamp(x + self.ffn(self.norm2(x)))
        return x


class T5CrossAttention(nn.Cell):
    def __init__(
        self,
        dim: int,
        dim_attn: int,
        dim_ffn: int,
        num_heads: int,
        num_buckets: int,
        shared_pos: bool = True,
        dropout: float = 0.1,
        dtype: ms.Type = ms.float32,
    ) -> None:
        super(T5CrossAttention, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.norm1 = T5LayerNorm(dim, dtype=dtype)
        self.self_attn = T5Attention(dim, dim_attn, num_heads, dropout, dtype=dtype)
        self.norm2 = T5LayerNorm(dim, dtype=dtype)
        self.cross_attn = T5Attention(dim, dim_attn, num_heads, dropout, dtype=dtype)
        self.norm3 = T5LayerNorm(dim, dtype=dtype)
        self.ffn = T5FeedForward(dim, dim_ffn, dropout, dtype=dtype)
        self.pos_embedding = (
            None if shared_pos else T5RelativeEmbedding(num_buckets, num_heads, bidirectional=False, dtype=dtype)
        )

    def construct(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        encoder_states: Optional[Tensor] = None,
        encoder_mask: Optional[Tensor] = None,
        pos_bias: Optional[Tensor] = None,
    ) -> Tensor:
        e = pos_bias if self.shared_pos else self.pos_embedding(x.shape[1], x.shape[1])
        x = fp16_clamp(x + self.self_attn(self.norm1(x), mask=mask, pos_bias=e))
        x = fp16_clamp(x + self.cross_attn(self.norm2(x), context=encoder_states, mask=encoder_mask))
        x = fp16_clamp(x + self.ffn(self.norm3(x)))
        return x


class T5RelativeEmbedding(nn.Cell):
    def __init__(
        self, num_buckets: int, num_heads: int, bidirectional: bool, max_dist: int = 128, dtype: ms.Type = ms.float32
    ) -> None:
        super(T5RelativeEmbedding, self).__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.max_dist = max_dist

        # layers
        self.embedding = mint.nn.Embedding(num_buckets, num_heads, dtype=dtype)

    def construct(self, lq: int, lk: int) -> Tensor:
        rel_pos = mint.arange(lk).unsqueeze(0) - mint.arange(lq).unsqueeze(1)
        rel_pos = self._relative_position_bucket(rel_pos)
        rel_pos_embeds = self.embedding(rel_pos)
        rel_pos_embeds = rel_pos_embeds.permute(2, 0, 1).unsqueeze(0)  # [1, N, Lq, Lk]
        return rel_pos_embeds.contiguous()

    def _relative_position_bucket(self, rel_pos: Tensor) -> Tensor:
        # preprocess
        if self.bidirectional:
            num_buckets = self.num_buckets // 2
            rel_buckets = (rel_pos > 0).to(ms.int32) * num_buckets
            rel_pos = mint.abs(rel_pos)
        else:
            num_buckets = self.num_buckets
            rel_buckets = 0
            rel_pos = -mint.min(rel_pos, mint.zeros_like(rel_pos))

        # embeddings for small and large positions
        max_exact = num_buckets // 2
        rel_pos_large = max_exact + (
            mint.log(rel_pos.float() / max_exact) / math.log(self.max_dist / max_exact) * (num_buckets - max_exact)
        ).to(ms.int32)
        rel_pos_large = mint.min(rel_pos_large, mint.full_like(rel_pos_large, num_buckets - 1))
        rel_buckets += mint.where(rel_pos < max_exact, rel_pos, rel_pos_large)
        return rel_buckets


class T5Encoder(nn.Cell):
    def __init__(
        self,
        vocab: Union[int, mint.nn.Embedding],
        dim: int,
        dim_attn: int,
        dim_ffn: int,
        num_heads: int,
        num_layers: int,
        num_buckets: int,
        shared_pos: bool = True,
        dropout: float = 0.1,
        dtype: ms.Type = ms.float32,
    ) -> None:
        super(T5Encoder, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos
        self.dtype = dtype

        # layers
        self.token_embedding = (
            vocab if isinstance(vocab, mint.nn.Embedding) else mint.nn.Embedding(vocab, dim, dtype=dtype)
        )
        self.pos_embedding = (
            T5RelativeEmbedding(num_buckets, num_heads, bidirectional=True, dtype=dtype) if shared_pos else None
        )
        self.dropout = mint.nn.Dropout(dropout)
        self.blocks = nn.CellList(
            [
                T5SelfAttention(dim, dim_attn, dim_ffn, num_heads, num_buckets, shared_pos, dropout, dtype=dtype)
                for _ in range(num_layers)
            ]
        )
        self.norm = T5LayerNorm(dim, dtype=dtype)

        # initialize weights
        self.apply(init_weights)

    def construct(self, ids: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = self.token_embedding(ids)
        x = self.dropout(x)
        e = self.pos_embedding(x.shape[1], x.shape[1]) if self.shared_pos else None
        for block in self.blocks:
            x = block(x, mask, pos_bias=e)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class T5Decoder(nn.Cell):
    def __init__(
        self,
        vocab: Union[int, mint.nn.Embedding],
        dim: int,
        dim_attn: int,
        dim_ffn: int,
        num_heads: int,
        num_layers: int,
        num_buckets: int,
        shared_pos: bool = True,
        dropout: float = 0.1,
        dtype: ms.Type = ms.float32,
    ) -> None:
        super(T5Decoder, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos
        self.dtype = dtype

        # layers
        self.token_embedding = (
            vocab if isinstance(vocab, mint.nn.Embedding) else mint.nn.Embedding(vocab, dim, dtype=dtype)
        )
        self.pos_embedding = (
            T5RelativeEmbedding(num_buckets, num_heads, bidirectional=False, dtype=dtype) if shared_pos else None
        )
        self.dropout = mint.nn.Dropout(dropout)
        self.blocks = nn.CellList(
            [
                T5CrossAttention(dim, dim_attn, dim_ffn, num_heads, num_buckets, shared_pos, dropout, dtype=dtype)
                for _ in range(num_layers)
            ]
        )
        self.norm = T5LayerNorm(dim, dtype=dtype)

        # initialize weights
        self.apply(init_weights)

    def construct(
        self,
        ids: Tensor,
        mask: Optional[Tensor] = None,
        encoder_states: Optional[Tensor] = None,
        encoder_mask: Optional[Tensor] = None,
    ) -> Tensor:
        b, s = ids.shape

        # causal mask
        if mask is None:
            mask = mint.tril(mint.ones((1, s, s)))
        elif mask.ndim == 2:
            mask = mint.tril(mask.unsqueeze(1).expand(-1, s, -1))

        # layers
        x = self.token_embedding(ids)
        x = self.dropout(x)
        e = self.pos_embedding(x.shape[1], x.shape[1]) if self.shared_pos else None
        for block in self.blocks:
            x = block(x, mask, encoder_states, encoder_mask, pos_bias=e)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class T5Model(nn.Cell):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        dim_attn: int,
        dim_ffn: int,
        num_heads: int,
        encoder_layers: int,
        decoder_layers: int,
        num_buckets: int,
        shared_pos: bool = True,
        dropout: float = 0.1,
        dtype: ms.Type = ms.float32,
    ) -> None:
        super(T5Model, self).__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.num_buckets = num_buckets
        self.dtype = dtype

        # layers
        self.token_embedding = mint.nn.Embedding(vocab_size, dim, dtype=dtype)
        self.encoder = T5Encoder(
            self.token_embedding,
            dim,
            dim_attn,
            dim_ffn,
            num_heads,
            encoder_layers,
            num_buckets,
            shared_pos,
            dropout,
            dtype=dtype,
        )
        self.decoder = T5Decoder(
            self.token_embedding,
            dim,
            dim_attn,
            dim_ffn,
            num_heads,
            decoder_layers,
            num_buckets,
            shared_pos,
            dropout,
            dtype=dtype,
        )
        self.head = mint.nn.Linear(dim, vocab_size, bias=False, dtype=dtype)

        # initialize weights
        self.apply(init_weights)

    def construct(self, encoder_ids: Tensor, encoder_mask: Tensor, decoder_ids: Tensor, decoder_mask: Tensor) -> Tensor:
        x = self.encoder(encoder_ids, encoder_mask)
        x = self.decoder(decoder_ids, decoder_mask, x, encoder_mask)
        x = self.head(x)
        return x


def _t5(
    name: str,
    encoder_only: bool = False,
    decoder_only: bool = False,
    return_tokenizer: bool = False,
    tokenizer_kwargs: Dict[str, Any] = {},
    dtype: ms.Type = ms.float32,
    **kwargs,
):
    # sanity check
    assert not (encoder_only and decoder_only)

    # params
    if encoder_only:
        model_cls = T5Encoder
        kwargs["vocab"] = kwargs.pop("vocab_size")
        kwargs["num_layers"] = kwargs.pop("encoder_layers")
        _ = kwargs.pop("decoder_layers")
    elif decoder_only:
        model_cls = T5Decoder
        kwargs["vocab"] = kwargs.pop("vocab_size")
        kwargs["num_layers"] = kwargs.pop("decoder_layers")
        _ = kwargs.pop("encoder_layers")
    else:
        model_cls = T5Model

    # init model
    model = model_cls(dtype=dtype, **kwargs)

    # init tokenizer
    if return_tokenizer:
        from .tokenizers import HuggingfaceTokenizer

        tokenizer = HuggingfaceTokenizer(f"google/{name}", **tokenizer_kwargs)
        return model, tokenizer
    else:
        return model


def umt5_xxl(**kwargs):
    cfg = dict(
        vocab_size=256384,
        dim=4096,
        dim_attn=4096,
        dim_ffn=10240,
        num_heads=64,
        encoder_layers=24,
        decoder_layers=24,
        num_buckets=32,
        shared_pos=False,
        dropout=0.1,
    )
    cfg.update(**kwargs)
    return _t5("umt5-xxl", **cfg)


class T5EncoderModel:
    def __init__(
        self,
        text_len: int,
        dtype=ms.bfloat16,
        checkpoint_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        shard_fn: Optional[Callable] = None,
    ) -> None:
        self.text_len = text_len
        self.dtype = dtype
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path

        # init model
        with no_init_parameters():
            model = umt5_xxl(encoder_only=True, return_tokenizer=False, dtype=dtype)
        model.set_train(False)
        for param in model.trainable_params():
            param.requires_grad = False

        if checkpoint_path is not None:
            logging.info(f"loading {checkpoint_path}")
            if checkpoint_path.endswith(".pth"):
                param_dict = load_pth(checkpoint_path, dtype=model.dtype)
                ms.load_param_into_net(model, param_dict)
            else:
                ms.load_checkpoint(checkpoint_path, model)
        model.init_parameters_data()

        self.model = model
        if shard_fn is not None:
            self.model = shard_fn(self.model)

        # init tokenizer
        self.tokenizer = HuggingfaceTokenizer(name=tokenizer_path, seq_len=text_len, clean="whitespace")

    def __call__(self, texts):
        ids, mask = self.tokenizer(texts, return_mask=True, add_special_tokens=True, return_tensors="np")
        ids, mask = Tensor(ids), Tensor(mask)
        seq_lens = mask.gt(0).sum(dim=1).to(ms.int32)
        context = self.model(ids, mask)
        return [u[:v] for u, v in zip(context, seq_lens)]
