from typing import Optional, Sequence, Tuple, Union

import numpy as np

import mindspore as ms
import mindspore.mint as mint
import mindspore.mint.nn.functional as F
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, Tensor
from mindspore.communication import get_group_size
from mindspore.ops.operations.nn_ops import FlashAttentionScore

from ...acceleration import get_sequence_parallel_group
from .activation import ACT2FN


class LlamaRMSNorm(nn.Cell):
    def __init__(self, hidden_size: Union[int, Sequence[int]], eps: float = 1e-6):
        super().__init__()
        self.weight = Parameter(np.ones(hidden_size).astype(np.float32))  # keep normalization at FP32
        self.variance_epsilon = eps

    def construct(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        hidden_states, _ = ops.rms_norm(hidden_states.to(ms.float32), self.weight, epsilon=self.variance_epsilon)
        return hidden_states.to(input_dtype)


class LlamaMLP(nn.Cell):
    def __init__(
        self,
        intermediate_size: int = 8192,
        hidden_size: int = 3072,
        hidden_act: str = "silu",
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = mint.nn.Linear(self.hidden_size, self.intermediate_size, bias=False, dtype=dtype)
        self.up_proj = mint.nn.Linear(self.hidden_size, self.intermediate_size, bias=False, dtype=dtype)
        self.down_proj = mint.nn.Linear(self.intermediate_size, self.hidden_size, bias=False, dtype=dtype)
        self.act_fn = ACT2FN[hidden_act]

    def construct(self, hidden_state: Tensor) -> Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


def repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :]
    hidden_states = mint.broadcast_to(hidden_states, (batch, num_key_value_heads, n_rep, slen, head_dim))
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
        self.q_proj = mint.nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=attention_bias, dtype=dtype)
        self.k_proj = mint.nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=attention_bias, dtype=dtype
        )
        self.v_proj = mint.nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=attention_bias, dtype=dtype
        )
        self.o_proj = mint.nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=attention_bias, dtype=dtype)

        if (sp_group := get_sequence_parallel_group()) is not None:
            self.sp_group_size = get_group_size(sp_group)
            self.alltoall = ops.AlltoAll(self.sp_group_size, 1, 2, group=sp_group)
        else:
            self.sp_group_size = None
            self.alltoall = nn.Identity()

    def construct(self, hidden_states: Tensor, encoder_hidden_states: Optional[Tensor] = None) -> Tensor:
        bsz, q_len, _ = hidden_states.shape

        kv_hidden_states = hidden_states if encoder_hidden_states is None else encoder_hidden_states

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(kv_hidden_states)
        value_states = self.v_proj(kv_hidden_states)

        query_states = ops.reshape(query_states, (bsz, -1, self.num_heads, self.head_dim))
        query_states = mint.permute(query_states, (0, 2, 1, 3))
        query_states = self.alltoall(query_states)

        key_states = ops.reshape(key_states, (bsz, -1, self.num_key_value_heads, self.head_dim))
        key_states = mint.permute(key_states, (0, 2, 1, 3))
        key_states = self.alltoall(key_states)

        value_states = ops.reshape(value_states, (bsz, -1, self.num_key_value_heads, self.head_dim))
        value_states = mint.permute(value_states, (0, 2, 1, 3))
        value_states = self.alltoall(value_states)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        key_states = mint.permute(key_states, (0, 1, 3, 2))
        attn_weights = mint.matmul(query_states, key_states) / mint.sqrt(Tensor(self.head_dim))

        # upcast attention to fp32
        attn_weights = attn_weights.to(ms.float32)
        attn_weights = F.softmax(attn_weights, dim=-1).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = mint.matmul(attn_weights, value_states)

        attn_output = mint.permute(attn_output, (0, 2, 1, 3))
        attn_output = self.alltoall(attn_output)
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
        not_recompute_fa: bool = False,
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
        num_heads = self.num_heads // self.sp_group_size if self.sp_group_size is not None else self.num_heads
        self.flash_attention = FlashAttentionScore(
            num_heads, keep_prob=1 - self.attention_dropout, scale_value=self.head_dim**-0.5, input_layout="BNSD"
        )
        if not_recompute_fa:
            self.flash_attention.recompute(False)

    def construct(self, hidden_states: Tensor, encoder_hidden_states: Optional[Tensor] = None) -> Tensor:
        bsz, q_len, _ = hidden_states.shape

        kv_hidden_states = hidden_states if encoder_hidden_states is None else encoder_hidden_states

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(kv_hidden_states)
        value_states = self.v_proj(kv_hidden_states)

        query_states = ops.reshape(query_states, (bsz, -1, self.num_heads, self.head_dim))
        query_states = mint.permute(query_states, (0, 2, 1, 3))
        query_states = self.alltoall(query_states)

        key_states = ops.reshape(key_states, (bsz, -1, self.num_key_value_heads, self.head_dim))
        key_states = mint.permute(key_states, (0, 2, 1, 3))
        key_states = self.alltoall(key_states)

        value_states = ops.reshape(value_states, (bsz, -1, self.num_key_value_heads, self.head_dim))
        value_states = mint.permute(value_states, (0, 2, 1, 3))
        value_states = self.alltoall(value_states)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        _, _, _, attn_output = self.flash_attention(query_states, key_states, value_states, None, None, None, None)
        attn_output = mint.permute(attn_output, (0, 2, 1, 3))
        attn_output = self.alltoall(attn_output)
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
        # assert t % self.patch_size[0] == 0
        # assert h % self.patch_size[1] == 0
        # assert w % self.patch_size[2] == 0

        x = mint.permute(x, (0, 2, 1, 3, 4))
        x = self.proj(x)  # (B C T H W)
        x = mint.flatten(x, start_dim=2)
        x = mint.permute(x, (0, 2, 1))
        return x


class LinearPatchEmbed3D(nn.Cell):
    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        in_channels: int = 8,
        hidden_size: int = 4096,
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = mint.nn.Linear(
            patch_size[0] * patch_size[1] * patch_size[2] * in_channels, hidden_size, bias=False, dtype=dtype
        )

    def construct(self, x: Tensor) -> Tensor:
        b, t, c, h, w = x.shape
        # assert t % self.patch_size[0] == 0
        # assert h % self.patch_size[1] == 0
        # assert w % self.patch_size[2] == 0

        p0, p1, p2 = self.patch_size[0], self.patch_size[1], self.patch_size[2]
        nt, nh, nw = t // p0, h // p1, w // p2
        x = ops.reshape(x, (b, nt, p0, c, nh, p1, nw, p2))
        x = mint.permute(x, (0, 1, 4, 6, 3, 2, 5, 7))  # (B, nt, nh, nw, c, p0, p1, p2)
        x = ops.reshape(x, (b, nt * nh * nw, -1))
        x = self.proj(x)
        return x


class TimestepEmbedder(nn.Cell):
    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        hidden_act: str = "silu",
        max_period: int = 10000,
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__()
        self.mlp = nn.SequentialCell(
            mint.nn.Linear(frequency_embedding_size, hidden_size, bias=False, dtype=dtype),
            ACT2FN[hidden_act],
            mint.nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype),
        )
        self.frequency_embedding_size = frequency_embedding_size
        half = frequency_embedding_size // 2
        self._freqs = Tensor(np.exp(-np.log(max_period) * np.arange(start=0, stop=half, dtype=np.float32) / half)[None])
        self._dtype = dtype

    @property
    def dtype(self):
        return self._dtype

    def timestep_embedding(self, t: Tensor) -> Tensor:
        args = ops.unsqueeze(t, 1).to(ms.float32) * self._freqs
        embedding = mint.cat([mint.cos(args), mint.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = mint.cat([embedding, mint.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def construct(self, t: Tensor) -> Tensor:
        t_freq = self.timestep_embedding(t)
        t_emb = self.mlp(t_freq.to(self.dtype))
        return t_emb
