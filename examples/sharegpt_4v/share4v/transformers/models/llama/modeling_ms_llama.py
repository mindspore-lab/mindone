import logging
import numbers
from typing import Literal, Optional, Tuple, Union

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, Tensor
from mindspore.common.initializer import Initializer, initializer
from mindspore.ops.operations.nn_ops import FlashAttentionScore

from ...activations_ms import ACT2FN

logger = logging.getLogger(__name__)


class Embedding(nn.Embedding):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        use_one_hot: bool = False,
        embedding_table: Union[Tensor, str, Initializer, numbers.Number] = "normal",
        dtype: ms.dtype = ms.float32,
        padding_idx: Optional[int] = None,
    ):
        """Initialize Embedding."""
        super(nn.Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.use_one_hot = use_one_hot
        self.dtype = dtype
        self.init_tensor = initializer(embedding_table, [vocab_size, embedding_size], dtype=dtype)
        self.padding_idx = padding_idx
        if padding_idx is not None:
            self.padding_idx = padding_idx
            if isinstance(self.init_tensor, Tensor) and self.init_tensor.init is not None:
                self.init_tensor = self.init_tensor.init_data()
            self.init_tensor = self.init_tensor.asnumpy()
            self.init_tensor[self.padding_idx] = 0
            self.init_tensor = Tensor(self.init_tensor)
        self.weight = Parameter(self.init_tensor)
        self.expand = ops.ExpandDims()
        self.reshape_flat = ops.Reshape()
        self.shp_flat = (-1,)
        self.gather = ops.Gather()
        self.one_hot = ops.OneHot()
        self.on_value = Tensor(1.0, self.dtype)
        self.off_value = Tensor(0.0, self.dtype)
        self.array_mul = ops.MatMul()
        self.reshape = ops.Reshape()
        self.get_shp = ops.Shape()
        self.concat = ops.Concat()

    def construct(self, ids):
        out_shape = self.get_shp(ids) + (self.embedding_size,)
        flat_ids = self.reshape_flat(ids, self.shp_flat)

        if self.use_one_hot:
            one_hot_ids = self.one_hot(flat_ids, self.vocab_size, self.on_value, self.off_value)
            output_for_reshape = self.array_mul(one_hot_ids, self.weight)
        else:
            output_for_reshape = self.gather(self.weight, flat_ids, 0)

        output = self.reshape(output_for_reshape, out_shape)
        return output

    def extend_repr(self):
        return (
            f"vocab_size={self.vocab_size}, embedding_size={self.embedding_size}, use_one_hot={self.use_one_hot}, "
            f"embedding_table={self.weight}, dtype={self.dtype}, padding_idx={self.padding_idx}"
        )


class LlamaRMSNorm(nn.Cell):
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


class LlamaRotaryEmbedding(nn.Cell):
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


class LlamaMLP(nn.Cell):
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


class LlamaAttention(nn.Cell):
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

        self.rotary_emb = LlamaRotaryEmbedding(
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
        attn_weights = ops.matmul(query_states, key_states) / ms.numpy.sqrt(
            ms.Tensor(self.head_dim, query_states.dtype)
        )

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


class LlamaFlashAttention(nn.Cell):
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

        self.rotary_emb = LlamaRotaryEmbedding(
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


MISTRAL_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attention": LlamaFlashAttention,
}


class LlamaDecoderLayer(nn.Cell):
    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        rms_norm_eps: float = 1e-5,
        max_position_embeddings: int = 32768,
        rope_theta: float = 1000000.0,
        attention_dropout: float = 0.0,
        hidden_act: str = "silu",
        attn_implementation: Literal["eager", "flash_attention"] = "eager",
        dtype: ms.dtype = ms.float32,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        self.self_attn = MISTRAL_ATTENTION_CLASSES[attn_implementation](
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            attention_dropout=attention_dropout,
            dtype=dtype,
        )

        self.mlp = LlamaMLP(
            intermediate_size=intermediate_size, hidden_size=hidden_size, hidden_act=hidden_act, dtype=dtype
        )
        self.input_layernorm = LlamaRMSNorm(hidden_size, eps=rms_norm_eps, dtype=dtype)
        self.post_attention_layernorm = LlamaRMSNorm(hidden_size, eps=rms_norm_eps, dtype=dtype)

    def construct(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_cache: Optional[Tensor] = None,
        past_value_cache: Optional[Tensor] = None,
        return_key_value_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, key_cache, value_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_cache=past_key_cache,
            past_value_cache=past_value_cache,
            return_key_value_cache=return_key_value_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, key_cache, value_cache


class LlamaModel(nn.Cell):
    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 32,
        num_hidden_layers: int = 32,
        num_key_value_heads: int = 8,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 1000000.0,
        vocab_size: int = 32064,
        attention_dropout: float = 0.0,
        hidden_act: str = "silu",
        pad_token_id: Optional[int] = None,
        attn_implementation: Literal["eager", "flash_attention"] = "eager",
        dtype: ms.dtype = ms.float32,
    ) -> None:
        super().__init__()
        self.padding_idx = pad_token_id
        self.vocab_size = vocab_size
        self.attn_implementation = attn_implementation

        self.embed_tokens = Embedding(vocab_size, hidden_size, padding_idx=self.padding_idx, dtype=dtype)
        self.layers = nn.CellList(
            [
                LlamaDecoderLayer(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    rms_norm_eps=rms_norm_eps,
                    max_position_embeddings=max_position_embeddings,
                    rope_theta=rope_theta,
                    attention_dropout=attention_dropout,
                    hidden_act=hidden_act,
                    attn_implementation=attn_implementation,
                    dtype=dtype,
                )
                for _ in range(num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(hidden_size, eps=rms_norm_eps, dtype=dtype)

    def get_input_embeddings(self) -> nn.Cell:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Cell) -> None:
        self.embed_tokens = value

    def construct(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        past_key_cache_list: Optional[Tensor] = None,
        past_value_cache_list: Optional[Tensor] = None,
        return_key_value_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = past_key_cache_list.shape[-2] if past_key_cache_list is not None else 0
        cache_position = ops.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], dtype=ms.int32)
        if position_ids is None:
            position_ids = ops.unsqueeze(cache_position, 0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position)

        hidden_states = inputs_embeds

        if return_key_value_cache:
            key_cache_list, value_cache_list = [], []
        else:
            key_cache_list, value_cache_list = None, None

        for i, decoder_layer in enumerate(self.layers):
            if past_key_cache_list is not None and past_value_cache_list is not None:
                past_key_cache, past_value_cache = past_key_cache_list[i], past_value_cache_list[i]
            else:
                past_key_cache, past_value_cache = None, None

            hidden_states, key_cache, value_cache = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_cache=past_key_cache,
                past_value_cache=past_value_cache,
                return_key_value_cache=return_key_value_cache,
            )

            if return_key_value_cache:
                key_cache_list.append(key_cache)
                value_cache_list.append(value_cache)

        hidden_states = self.norm(hidden_states)

        if return_key_value_cache:
            key_cache_list = ops.stack(key_cache_list)
            value_cache_list = ops.stack(value_cache_list)

        return hidden_states, key_cache_list, value_cache_list

    def _update_causal_mask(self, attention_mask: Tensor, input_tensor: Tensor, cache_position: Tensor) -> Tensor:
        dtype = input_tensor.dtype
        sequence_length = input_tensor.shape[1]
        target_length = attention_mask.shape[-1]

        if len(attention_mask.shape) == 4:
            # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
            if attention_mask.max() != 0:
                raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
            causal_mask = attention_mask
        else:
            fill_value = -ms.numpy.inf if self.attn_implementation == "eager" else 1.0
            causal_mask = ops.full((sequence_length, target_length), fill_value=fill_value, dtype=dtype)
            exclude_mask = ops.arange(target_length) > cache_position.reshape(-1, 1)
            causal_mask = ops.masked_fill(causal_mask, ~exclude_mask, Tensor(0, dtype=dtype))
            causal_mask = ops.broadcast_to(causal_mask[None, None, :, :], (input_tensor.shape[0], 1, -1, -1))
            if len(attention_mask.shape) == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = ops.masked_fill(
                    causal_mask[:, :, :, :mask_length], padding_mask, Tensor(fill_value, dtype=dtype)
                )

        return causal_mask


class LlamaForCausalLM(nn.Cell):
    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 32,
        num_hidden_layers: int = 32,
        num_key_value_heads: int = 8,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 1000000.0,
        vocab_size: int = 32000,
        attention_dropout: float = 0.0,
        hidden_act: str = "silu",
        pad_token_id: Optional[int] = None,
        attn_implementation: Literal["eager", "flash_attention"] = "eager",
        dtype: ms.dtype = ms.float32,
        **kwargs,
    ) -> None:
        super().__init__()

        self.model = LlamaModel(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=num_key_value_heads,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            vocab_size=vocab_size,
            attention_dropout=attention_dropout,
            hidden_act=hidden_act,
            pad_token_id=pad_token_id,
            attn_implementation=attn_implementation,
            dtype=dtype,
        )
        self.vocab_size = vocab_size
        self.lm_head = nn.Dense(hidden_size, vocab_size, has_bias=False, dtype=dtype)

    def get_input_embeddings(self) -> nn.Cell:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Cell) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Cell:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Cell) -> None:
        self.lm_head = new_embeddings

    def set_decoder(self, decoder: nn.Cell) -> None:
        self.model = decoder

    def get_decoder(self) -> nn.Cell:
        return self.model

    @ms.jit
    def construct(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        past_key_cache_list: Optional[Tensor] = None,
        past_value_cache_list: Optional[Tensor] = None,
        return_key_value_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        hidden_states, key_cache_list, value_cache_list = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_cache_list=past_key_cache_list,
            past_value_cache_list=past_value_cache_list,
            return_key_value_cache=return_key_value_cache,
        )
        logits = self.lm_head(hidden_states).to(ms.float32)
        return logits, key_cache_list, value_cache_list
