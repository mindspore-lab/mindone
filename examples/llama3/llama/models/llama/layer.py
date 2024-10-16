import logging
from typing import Optional, Tuple

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, Tensor
from mindspore.ops.operations.nn_ops import FlashAttentionScore

from ..activation import ACT2FN

logger = logging.getLogger(__name__)


class LlamaRMSNorm(nn.Cell):
    def __init__(self, hidden_size: int, eps: float = 1e-6, dtype: ms.Type = ms.float32) -> None:
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


class LlamaLayerNorm(nn.LayerNorm):
    def construct(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(ms.float32)
        hidden_states, _, _ = self.layer_norm(
            hidden_states, self.gamma.to(hidden_states.dtype), self.beta.to(hidden_states.dtype)
        )
        return hidden_states.to(input_dtype)


class LlamaMLP(nn.Cell):
    def __init__(
        self,
        intermediate_size: int = 14336,
        hidden_size: int = 4096,
        hidden_act: str = "silu",
        dtype: ms.Type = ms.float32,
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
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :]
    hidden_states = ops.broadcast_to(hidden_states, (batch, num_key_value_heads, n_rep, slen, head_dim))
    hidden_states = ops.reshape(hidden_states, (batch, num_key_value_heads * n_rep, slen, head_dim))
    return hidden_states


class LlamaAttention(nn.Cell):
    def __init__(
        self,
        hidden_size: int = 4096,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__()

        self.attention_dropout = attention_dropout
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Dense(self.hidden_size, self.num_heads * self.head_dim, has_bias=attention_bias, dtype=dtype)
        self.k_proj = nn.Dense(
            self.hidden_size, self.num_key_value_heads * self.head_dim, has_bias=attention_bias, dtype=dtype
        )
        self.v_proj = nn.Dense(
            self.hidden_size, self.num_key_value_heads * self.head_dim, has_bias=attention_bias, dtype=dtype
        )
        self.o_proj = nn.Dense(self.num_heads * self.head_dim, self.hidden_size, has_bias=attention_bias, dtype=dtype)

    def construct(self, hidden_states: Tensor, encoder_hidden_states: Optional[Tensor] = None) -> Tensor:
        bsz, q_len, _ = hidden_states.shape

        kv_hidden_states = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        _, kv_len, _ = kv_hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(kv_hidden_states)
        value_states = self.v_proj(kv_hidden_states)

        query_states = ops.reshape(query_states, (bsz, q_len, self.num_heads, self.head_dim))
        query_states = ops.transpose(query_states, (0, 2, 1, 3))

        key_states = ops.reshape(key_states, (bsz, kv_len, self.num_key_value_heads, self.head_dim))
        key_states = ops.transpose(key_states, (0, 2, 1, 3))

        value_states = ops.reshape(value_states, (bsz, kv_len, self.num_key_value_heads, self.head_dim))
        value_states = ops.transpose(value_states, (0, 2, 1, 3))

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        key_states = ops.transpose(key_states, (0, 1, 3, 2))
        attn_weights = ops.matmul(query_states, key_states) / ms.numpy.sqrt(self.head_dim)

        # upcast attention to fp32
        attn_weights = attn_weights.to(ms.float32)
        attn_weights = ops.softmax(attn_weights, axis=-1).to(query_states.dtype)
        attn_weights = ops.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = ops.matmul(attn_weights, value_states)

        attn_output = ops.transpose(attn_output, (0, 2, 1, 3))
        attn_output = ops.reshape(attn_output, (bsz, q_len, -1))
        attn_output = self.o_proj(attn_output)

        return attn_output


class LlamaFlashAttention(LlamaAttention):
    def __init__(
        self,
        hidden_size: int = 4096,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            attention_dropout=attention_dropout,
            attention_bias=attention_bias,
            dtype=dtype,
        )
        self.flash_attention = FlashAttentionScore(
            self.num_heads, keep_prob=1 - self.attention_dropout, scale_value=self.head_dim**-0.5, input_layout="BSND"
        )

    def construct(self, hidden_states: Tensor, encoder_hidden_states: Optional[Tensor] = None) -> Tensor:
        bsz, q_len, _ = hidden_states.shape

        kv_hidden_states = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        _, kv_len, _ = kv_hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(kv_hidden_states)
        value_states = self.v_proj(kv_hidden_states)

        query_states = ops.reshape(query_states, (bsz, q_len, self.num_heads, self.head_dim))
        query_states = ops.transpose(query_states, (0, 2, 1, 3))

        key_states = ops.reshape(key_states, (bsz, kv_len, self.num_key_value_heads, self.head_dim))
        key_states = ops.transpose(key_states, (0, 2, 1, 3))

        value_states = ops.reshape(value_states, (bsz, kv_len, self.num_key_value_heads, self.head_dim))
        value_states = ops.transpose(value_states, (0, 2, 1, 3))

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Reshape to the expected shape and dtype for Flash Attention
        query_states = ops.transpose(query_states, (0, 2, 1, 3))
        key_states = ops.transpose(key_states, (0, 2, 1, 3))
        value_states = ops.transpose(value_states, (0, 2, 1, 3))

        _, _, _, attn_output = self.flash_attention(query_states, key_states, value_states, None, None, None, None)
        attn_output = ops.reshape(attn_output, (bsz, q_len, -1))
        attn_output = self.o_proj(attn_output)

        return attn_output


class PatchEmbed3D(nn.Cell):
    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        in_channels: int = 8,
        hidden_size: int = 4096,
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_channels,
            hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            pad_mode="pad",
            has_bias=False,
            dtype=dtype,
        )

    def construct(self, x: Tensor) -> Tensor:
        _, t, _, h, w = x.shape
        assert t % self.patch_size[0] == 0
        assert h % self.patch_size[1] == 0
        assert w % self.patch_size[2] == 0

        x = ops.transpose(x, (0, 2, 1, 3, 4))
        x = self.proj(x)  # (B C T H W)
        x = x.flatten(start_dim=2).swapaxes(1, 2)
        return x


class TimestepEmbedder(nn.Cell):
    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        hidden_act: str = "silu",
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__()
        self.mlp = nn.SequentialCell(
            nn.Dense(frequency_embedding_size, hidden_size, has_bias=False, dtype=dtype),
            ACT2FN[hidden_act],
            nn.Dense(hidden_size, hidden_size, has_bias=False, dtype=dtype),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.dtype = dtype

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
        t_emb = self.mlp(t_freq.to(self.dtype))
        return t_emb


class CaptionEmbedder(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__()
        self.proj = nn.SequentialCell(
            nn.Dense(in_channels, hidden_size, has_bias=False, dtype=dtype),
            LlamaLayerNorm((hidden_size,), dtype=dtype),
        )

    def construct(self, caption: Tensor) -> Tensor:
        caption_emb = self.proj(caption)
        return caption_emb
