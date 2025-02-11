import logging
from typing import Optional, Tuple

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, Tensor
from mindspore.ops.operations.nn_ops import FlashAttentionScore

from ..activation import ACT2FN

logger = logging.getLogger(__name__)


class MistralRMSNorm(nn.Cell):
    def __init__(self, hidden_size: int, eps: float = 1e-6, dtype: ms.dtype = ms.float32) -> None:
        super().__init__()
        self.weight = Parameter(ops.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def construct(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(ms.float32)
        variance = ops.pow(hidden_states, 2)
        variance = ops.mean(variance, axis=-1, keep_dims=True)
        hidden_states = hidden_states * ops.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class MistralRotaryEmbedding(nn.Cell):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0) -> None:
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.inv_freq = 1.0 / (self.base ** (ops.arange(0, self.dim, 2, dtype=ms.float32) / self.dim))

    def construct(self, x: Tensor, position_ids: Tensor) -> Tuple[Tensor, Tensor]:
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = ops.broadcast_to(self.inv_freq[None, :, None], (position_ids.shape[0], -1, 1))
        position_ids_expanded = position_ids[:, None, :].to(ms.float32)
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        freqs = ops.matmul(inv_freq_expanded.to(ms.float32), position_ids_expanded.to(ms.float32))
        freqs = ops.transpose(freqs, (0, 2, 1))
        emb = ops.concat((freqs, freqs), axis=-1)
        cos = ops.cos(emb)
        sin = ops.sin(emb)
        return cos.to(x.dtype), sin.to(x.dtype)


def rotate_half(x: Tensor) -> Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return ops.concat((-x2, x1), axis=-1)


def apply_rotary_pos_emb(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, unsqueeze_dim: int = 1
) -> Tuple[Tensor, Tensor]:
    cos = ops.unsqueeze(cos, unsqueeze_dim)
    sin = ops.unsqueeze(sin, unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MistralMLP(nn.Cell):
    def __init__(
        self,
        intermediate_size: int = 14336,
        hidden_size: int = 4096,
        hidden_act: str = "silu",
        dtype: ms.dtype = ms.float32,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Dense(self.hidden_size, self.intermediate_size, has_bias=False, dtype=dtype)
        self.up_proj = nn.Dense(self.hidden_size, self.intermediate_size, has_bias=False, dtype=dtype)
        self.down_proj = nn.Dense(self.intermediate_size, self.hidden_size, has_bias=False, dtype=dtype)
        self.act_fn = ACT2FN[hidden_act]

    def construct(self, hidden_state: Tensor) -> Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


def repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :]
    hidden_states = ops.broadcast_to(hidden_states, (batch, num_key_value_heads, n_rep, slen, head_dim))
    hidden_states = ops.reshape(hidden_states, (batch, num_key_value_heads * n_rep, slen, head_dim))
    return hidden_states


class MistralAttention(nn.Cell):
    def __init__(
        self,
        hidden_size: int = 4096,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        max_position_embeddings: int = 32768,
        rope_theta: float = 1000000.0,
        attention_dropout: float = 0.0,
        dtype: ms.dtype = ms.float32,
    ) -> None:
        super().__init__()

        self.attention_dropout = attention_dropout
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Dense(self.hidden_size, self.num_heads * self.head_dim, has_bias=False, dtype=dtype)
        self.k_proj = nn.Dense(self.hidden_size, self.num_key_value_heads * self.head_dim, has_bias=False, dtype=dtype)
        self.v_proj = nn.Dense(self.hidden_size, self.num_key_value_heads * self.head_dim, has_bias=False, dtype=dtype)
        self.o_proj = nn.Dense(self.num_heads * self.head_dim, self.hidden_size, has_bias=False, dtype=dtype)

        self.rotary_emb = MistralRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def construct(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_cache: Optional[Tensor] = None,
        past_value_cache: Optional[Tensor] = None,
        return_key_value_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = ops.reshape(query_states, (bsz, q_len, self.num_heads, self.head_dim))
        query_states = ops.transpose(query_states, (0, 2, 1, 3))

        key_states = ops.reshape(key_states, (bsz, q_len, self.num_key_value_heads, self.head_dim))
        key_states = ops.transpose(key_states, (0, 2, 1, 3))

        value_states = ops.reshape(value_states, (bsz, q_len, self.num_key_value_heads, self.head_dim))
        value_states = ops.transpose(value_states, (0, 2, 1, 3))

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if return_key_value_cache:
            key_cache, value_cache = key_states, value_states
        else:
            key_cache, value_cache = None, None

        if past_key_cache is not None and past_value_cache is not None:
            key_states = ops.concat([past_key_cache, key_states], axis=-2)
            value_states = ops.concat([past_value_cache, value_states], axis=-2)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        key_states = ops.transpose(key_states, (0, 1, 3, 2))
        attn_weights = ops.matmul(query_states, key_states) / ms.numpy.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = ops.softmax(attn_weights.to(ms.float32), axis=-1).to(query_states.dtype)
        attn_weights = ops.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = ops.matmul(attn_weights, value_states)

        attn_output = ops.transpose(attn_output, (0, 2, 1, 3))
        attn_output = ops.reshape(attn_output, (bsz, q_len, -1))
        attn_output = self.o_proj(attn_output)

        return attn_output, key_cache, value_cache


class MistralFlashAttention(nn.Cell):
    def __init__(
        self,
        hidden_size: int = 4096,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        max_position_embeddings: int = 32768,
        rope_theta: float = 1000000.0,
        attention_dropout: float = 0.0,
        dtype: ms.dtype = ms.float32,
    ) -> None:
        super().__init__()

        self.attention_dropout = attention_dropout
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Dense(self.hidden_size, self.num_heads * self.head_dim, has_bias=False, dtype=dtype)
        self.k_proj = nn.Dense(self.hidden_size, self.num_key_value_heads * self.head_dim, has_bias=False, dtype=dtype)
        self.v_proj = nn.Dense(self.hidden_size, self.num_key_value_heads * self.head_dim, has_bias=False, dtype=dtype)
        self.o_proj = nn.Dense(self.num_heads * self.head_dim, self.hidden_size, has_bias=False, dtype=dtype)

        self.rotary_emb = MistralRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

        self.flash_attention = FlashAttentionScore(
            self.num_heads, keep_prob=1 - self.attention_dropout, scale_value=self.head_dim**-0.5, input_layout="BSND"
        )

    def construct(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_cache: Optional[Tensor] = None,
        past_value_cache: Optional[Tensor] = None,
        return_key_value_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = ops.reshape(query_states, (bsz, q_len, self.num_heads, self.head_dim))
        query_states = ops.transpose(query_states, (0, 2, 1, 3))

        key_states = ops.reshape(key_states, (bsz, q_len, self.num_key_value_heads, self.head_dim))
        key_states = ops.transpose(key_states, (0, 2, 1, 3))

        value_states = ops.reshape(value_states, (bsz, q_len, self.num_key_value_heads, self.head_dim))
        value_states = ops.transpose(value_states, (0, 2, 1, 3))

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if return_key_value_cache:
            key_cache, value_cache = key_states, value_states
        else:
            key_cache, value_cache = None, None

        if past_key_cache is not None and past_value_cache is not None:
            key_states = ops.concat([key_states, past_key_cache], axis=-2)
            value_states = ops.concat([value_states, past_value_cache], axis=-2)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Reshape to the expected shape and dtype for Flash Attention
        query_states = ops.transpose(query_states, (0, 2, 1, 3))
        key_states = ops.transpose(key_states, (0, 2, 1, 3))
        value_states = ops.transpose(value_states, (0, 2, 1, 3))
        attention_mask = attention_mask.to(ms.uint8)

        _, _, _, attn_output = self.flash_attention(
            query_states, key_states, value_states, None, None, None, attention_mask
        )
        attn_output = ops.reshape(attn_output, (bsz, q_len, -1))
        attn_output = self.o_proj(attn_output)

        return attn_output, key_cache, value_cache
