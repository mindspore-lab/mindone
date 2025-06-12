# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from typing import Optional, Tuple, Union

import numpy as np
from transformers import LlamaConfig
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging

import mindspore as ms
from mindspore import Parameter, Tensor, nn, ops
from mindspore.common import initializer as init
from mindspore.ops.operations.nn_ops import FlashAttentionScore

from ...activations import ACT2FN
from ...cache_utils import get_max_length, get_seq_length, init_static_cache, update
from ...mindspore_adapter import recompute_except_output
from ...mindspore_adapter.utils import _MIN_FP16  # , dtype_to_min

# from ...mindspore_adapter.attention import FlashAttention2
from ...mindspore_utils import ALL_LAYERNORM_LAYERS
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from ...modeling_utils import MSPreTrainedModel as PreTrainedModel

logger = logging.get_logger(__name__)


_CONFIG_FOR_DOC = "LlamaConfig"


class LlamaRMSNorm(nn.Cell):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = Parameter(Tensor(np.ones(hidden_size), ms.float32), name="weight")
        self.variance_epsilon = eps

    def construct(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(ms.float32)
        variance = hidden_states.pow(2).mean(-1, keep_dims=True)
        hidden_states = hidden_states * ops.rsqrt(variance + self.variance_epsilon)
        out = self.weight * hidden_states.to(input_dtype)
        return out


ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


class LlamaRotaryEmbedding(nn.Cell):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (np.arange(0, self.dim, 2).astype(np.float32) / self.dim))
        # self.inv_freq = Parameter(Tensor(inv_freq, ms.float32), requires_grad=False, name="inv_freq_buffer")
        self.inv_freq = Tensor(inv_freq, ms.float32)
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings

    # with no grad
    def construct(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].to(ms.float32).broadcast_to((position_ids.shape[0], -1, 1))
        position_ids_expanded = position_ids[:, None, :].to(ms.float32)
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        freqs = ops.matmul(inv_freq_expanded, position_ids_expanded).swapdims(1, 2)
        emb = ops.cat((freqs, freqs), axis=-1)
        cos = emb.cos()
        sin = emb.sin()
        cos, sin = cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        cos, sin = ops.stop_gradient(cos), ops.stop_gradient(sin)
        return cos, sin


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def construct(self, x, position_ids):
        # difference to the original RoPE: a scaling factor is aplied to the position ids
        position_ids = position_ids.to(ms.float32) / self.scaling_factor
        cos, sin = super().construct(x, position_ids)
        return cos, sin


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def construct(self, x, position_ids):
        # difference to the original RoPE: inv_freq is recomputed when the sequence length > original length
        seq_len = ops.max(position_ids)[0] + 1
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (ops.arange(0, self.dim, 2, dtype=ms.float32) / self.dim))
            x = ops.depend(x, ops.assign(self.inv_freq, inv_freq))

        cos, sin = super().construct(x, position_ids)
        return cos, sin


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return ops.cat((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`ms.Tensor`): The query tensor.
        k (`ms.Tensor`): The key tensor.
        cos (`ms.Tensor`): The cosine part of the rotary embedding.
        sin (`ms.Tensor`): The sine part of the rotary embedding.
        position_ids (`ms.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(ms.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Dense(self.hidden_size, self.intermediate_size, has_bias=config.mlp_bias)
        self.up_proj = nn.Dense(self.hidden_size, self.intermediate_size, has_bias=config.mlp_bias)
        self.down_proj = nn.Dense(self.intermediate_size, self.hidden_size, has_bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

        # setting config var to self attribute
        _name_list = [
            "pretraining_tp",
        ]
        for name in _name_list:
            setattr(self, name, getattr(config, name))

    def construct(self, x):
        if self.pretraining_tp > 1:
            slice = self.intermediate_size // self.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, axis=0)
            up_proj_slices = self.up_proj.weight.split(slice, axis=0)
            down_proj_slices = self.down_proj.weight.split(slice, axis=1)

            gate_proj = ops.cat([ops.dense(x, gate_proj_slices[i]) for i in range(self.pretraining_tp)], axis=-1)
            up_proj = ops.cat([ops.dense(x, up_proj_slices[i]) for i in range(self.pretraining_tp)], axis=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, axis=2)
            down_proj = [ops.dense(intermediate_states[i], down_proj_slices[i]) for i in range(self.pretraining_tp)]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def repeat_kv(hidden_states: ms.Tensor, n_rep: int) -> ms.Tensor:
    """
    This is the equivalent of ops.repeat_interleave(x, axis=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].broadcast_to((batch, num_key_value_heads, n_rep, slen, head_dim))
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttention(nn.Cell):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Dense(self.hidden_size, self.num_heads * self.head_dim, has_bias=config.attention_bias)
        self.k_proj = nn.Dense(
            self.hidden_size, self.num_key_value_heads * self.head_dim, has_bias=config.attention_bias
        )
        self.v_proj = nn.Dense(
            self.hidden_size, self.num_key_value_heads * self.head_dim, has_bias=config.attention_bias
        )
        self.o_proj = nn.Dense(self.hidden_size, self.hidden_size, has_bias=config.attention_bias)
        self._init_rope()

        _name_list = [
            "pretraining_tp",
        ]
        for name in _name_list:
            setattr(self, name, getattr(config, name))

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_value: Optional[Tuple[ms.Tensor, ms.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[ms.Tensor] = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.shape

        if self.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.pretraining_tp
            query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.pretraining_tp, axis=0)
            key_slices = self.k_proj.weight.split(key_value_slicing, axis=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, axis=0)

            query_states = [ops.dense(hidden_states, query_slices[i]) for i in range(self.pretraining_tp)]
            query_states = ops.cat(query_states, axis=-1)

            key_states = [ops.dense(hidden_states, key_slices[i]) for i in range(self.pretraining_tp)]
            key_states = ops.cat(key_states, axis=-1)

            value_states = [ops.dense(hidden_states, value_slices[i]) for i in range(self.pretraining_tp)]
            value_states = ops.cat(value_states, axis=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).swapdims(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).swapdims(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).swapdims(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None and use_cache:
            key_states, value_states = update(past_key_value, key_states, value_states, cache_position)
            past_key_value = (key_states, value_states)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = ops.matmul(query_states, key_states.swapdims(2, 3)) / (self.head_dim**0.5)

        attn_weights = ops.cast(attn_weights, ms.float32)
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + ops.cast(causal_mask, attn_weights.dtype)

        # upcast attention to fp32
        attn_weights = ops.softmax(attn_weights, axis=-1, dtype=ms.float32).to(query_states.dtype)
        attn_weights = ops.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = ops.matmul(attn_weights, value_states)

        # assert attn_output.shape == (bsz, self.num_heads, q_len, self.head_dim)

        attn_output = attn_output.swapdims(1, 2)

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.pretraining_tp, axis=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.pretraining_tp, axis=1)
            attn_output = sum([ops.dense(attn_output[i], o_proj_slices[i]) for i in range(self.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaFlashAttention2(LlamaAttention):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        scale_factor = 1 / math.sqrt(self.head_dim)
        self.flash_attention = FlashAttentionScore(
            self.num_heads, keep_prob=1 - self.attention_dropout, scale_value=scale_factor, input_layout="BNSD"
        )

    def convert_mask_to_fa_format(self, attention_mask):
        if attention_mask is not None:
            if attention_mask.dtype == ms.bool_:
                # flip mask, since ms FA treats 1 as discard, 0 as retain.
                attention_mask = 1 - attention_mask
                attention_mask = attention_mask.to(ms.uint8)
            else:
                attention_mask = attention_mask.to(ms.float16)
                attention_mask = ops.select(
                    ops.equal(attention_mask, _MIN_FP16),
                    ops.ones((), ms.uint8),
                    ops.zeros((), ms.uint8),
                )

        return attention_mask

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_value: Optional[Tuple[ms.Tensor, ms.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[ms.Tensor] = None,
        **kwargs,
    ):
        # assert output_attentions == False

        bsz, q_len, _ = hidden_states.shape

        if self.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.pretraining_tp
            query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.pretraining_tp, axis=0)
            key_slices = self.k_proj.weight.split(key_value_slicing, axis=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, axis=0)

            query_states = [ops.dense(hidden_states, query_slices[i]) for i in range(self.pretraining_tp)]
            query_states = ops.cat(query_states, axis=-1)

            key_states = [ops.dense(hidden_states, key_slices[i]) for i in range(self.pretraining_tp)]
            key_states = ops.cat(key_states, axis=-1)

            value_states = [ops.dense(hidden_states, value_slices[i]) for i in range(self.pretraining_tp)]
            value_states = ops.cat(value_states, axis=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).swapdims(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).swapdims(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).swapdims(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            key_states, value_states = update(past_key_value, key_states, value_states, cache_position)
            past_key_value = (key_states, value_states)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # 1. flash attention
        if attention_mask is not None:  # no matter the length, we just slice it
            attention_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        # flip mask to ms FA format, 1 - drop, 0 - retain
        attention_mask = (-attention_mask).to(ms.bool_)
        _, _, _, attn_output = self.flash_attention(
            query_states, key_states, value_states, None, None, None, attention_mask
        )
        # assert attn_output.shape == (bsz, self.num_heads, q_len, self.head_dim)

        # 2. vanilla attention
        # attn_weights = ops.matmul(query_states, key_states.swapdims(2, 3)) / (self.head_dim ** 0.5)
        #
        # if attention_mask is not None:  # no matter the length, we just slice it
        #     causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        #     attn_weights = attn_weights + causal_mask
        #
        # # upcast attention to fp32
        # attn_weights = ops.softmax(attn_weights, axis=-1, dtype=ms.float32).to(query_states.dtype)
        # attn_weights = ops.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        # attn_output = ops.matmul(attn_weights, value_states)
        # # assert attn_output.shape == (bsz, self.num_heads, q_len, self.head_dim)

        attn_output = attn_output.swapdims(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.pretraining_tp, axis=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.pretraining_tp, axis=1)
            attn_output = sum([ops.dense(attn_output[i], o_proj_slices[i]) for i in range(self.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attention_2": LlamaFlashAttention2,
    # "sdpa": None,  # not support sdpa
}


class LlamaDecoderLayer(nn.Cell):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.output_identity = nn.Identity()

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[ms.Tensor, ms.Tensor]]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[ms.Tensor] = None,
        **kwargs,
    ) -> Tuple[ms.Tensor, Optional[Tuple[ms.Tensor, ms.Tensor]]]:
        """
        Args:
            hidden_states (`ms.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`ms.Tensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(ms.Tensor)`, *optional*): cached past key and value projection states
            cache_position (`ms.Tensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.output_identity(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a MindSpore [mindspore.nn.Cell](https://www.mindspore.cn/docs/zh-CN/r2.3.1/api_python/nn/mind
    spore.nn.Cell.html?highlight=cell#mindspore.nn.Cell) subclass.
    Use it as a regular MindSpore Cell and refer to the MindSpore documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    _no_split_modules = ["LlamaDecoderLayer"]
    _supports_flash_attn_2 = True
    _supports_sdpa = False
    _supports_cache_class = False
    _supports_quantized_cache = False
    _supports_static_cache = False

    def _init_weights(self, cell):
        std = self.config.initializer_range
        if isinstance(cell, nn.Dense):
            cell.weight.set_data(
                init.initializer(init.Normal(mean=0.0, sigma=std), cell.weight.shape, cell.weight.dtype)
            )
            if cell.bias is not None:
                cell.bias.set_data(init.initializer(init.Zero(), cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Embedding):
            cell.embedding_table.set_data(
                init.initializer(
                    init.Normal(mean=0.0, sigma=std), cell.embedding_table.shape, cell.embedding_table.dtype
                )
            )
            if cell.padding_idx is not None:
                cell.embedding_table.data[cell.padding_idx] = 0.0


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`ms.Tensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(ms.Tensor, ms.Tensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Only one formats are allowed:
            - Tuple of `tuple(ms.Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`ms.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Ignore `return_dict`.
        cache_position (`ms.Tensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.layers = nn.CellList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # set congig var to self attribute
        _name_list = [
            "output_attentions",
            "output_hidden_states",
            "use_return_dict",
            "use_cache",
            "_attn_implementation",
            "pretraining_tp",
            "vocab_size",
        ]
        for name in _name_list:
            setattr(self, name, getattr(config, name))

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding):
        if not isinstance(value, nn.Embedding):
            raise NotImplementedError
        ori_name = value.embedding_table.name

        self.embed_tokens = value

        self.embed_tokens.embedding_table.name = ori_name

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if gradient_checkpointing_kwargs is None:
            # gradient_checkpointing_kwargs = {"mp_comm_recompute": True, "parallel_optimizer_comm_recompute": True}
            gradient_checkpointing_kwargs = {}

        # llama layers
        for decoder_layer in self.layers:
            assert isinstance(decoder_layer, LlamaDecoderLayer)
            for name, cell in decoder_layer.name_cells().items():
                if "output_identity" in name:
                    assert isinstance(cell, nn.Identity)
                    pass
                else:
                    # cell._recompute()
                    recompute_except_output(cell, **gradient_checkpointing_kwargs)
        recompute_except_output(self.embed_tokens, **gradient_checkpointing_kwargs)
        recompute_except_output(self.norm, **gradient_checkpointing_kwargs)

        logger.info(f"{self.__class__.__name__}: enable recompute.")

    def prepare_static_cache(self, input_embeds, max_cache_len):
        bs = input_embeds.shape[0]
        max_batch_size, cache_dtype = (
            getattr(self.config, "num_beams", 1) * bs,
            self.dtype,
        )
        past_key_values = init_static_cache(
            config=self.config, max_batch_size=max_batch_size, max_cache_len=max_cache_len, dtype=cache_dtype
        )
        return past_key_values

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def construct(
        self,
        input_ids: ms.Tensor = None,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_values: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = False,
        cache_position: Optional[ms.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.use_cache
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        if self.training:
            use_cache = False

        # assert ((input_ids is None) and (inputs_embeds is not None)) or \
        #        ((input_ids is not None) and (inputs_embeds is None))
        # # assert (input_ids is None) ^ (inputs_embeds is None)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = get_seq_length(past_key_values) if past_key_values is not None else 0
            cache_position = ops.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], dtype=ms.int32)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_key_values)

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_caches = () if use_cache else None

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values[layer_idx] if past_key_values is not None else None,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_caches += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, next_caches, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_caches,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: ms.Tensor,
        input_tensor: ms.Tensor,
        cache_position: ms.Tensor,
        past_key_values: Tuple[Tuple[ms.Tensor, ms.Tensor]],
        output_attentions: bool = False,
    ):
        # if self._attn_implementation == "flash_attention_2":
        #     return attention_mask

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = get_seq_length(past_key_values) if past_key_values is not None else 0

        sequence_length = input_tensor.shape[1]

        if past_key_values is not None:
            target_length = get_max_length(past_key_values)
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, ms.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        if attention_mask is not None and len(attention_mask.shape) == 4:
            # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
            # if attention_mask.max() != 0:
            #     raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
            causal_mask = attention_mask
        else:
            causal_mask = ops.broadcast_to(_MIN_FP16, (sequence_length, target_length))
            if sequence_length != 1:
                causal_mask = ops.triu(causal_mask, diagonal=1)
            _mask_position = ops.arange(target_length) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask * _mask_position
            causal_mask = causal_mask[None, None, :, :].broadcast_to((input_tensor.shape[0], 1, -1, -1))
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for -in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, _MIN_FP16
                )

        return causal_mask


class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        _name_list = ["output_attentions", "output_hidden_states", "use_return_dict", "pretraining_tp", "vocab_size"]
        for name in _name_list:
            setattr(self, name, getattr(config, name))

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def construct(
        self,
        input_ids: ms.Tensor = None,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[ms.Tensor, ms.Tensor]]] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        labels: Optional[ms.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = False,
        cache_position: Optional[ms.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        hidden_states = outputs[0]

        if self.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.pretraining_tp, axis=0)
            logits = [ops.dense(hidden_states, lm_head_slices[i]) for i in range(self.pretraining_tp)]
            logits = ops.cat(logits, axis=-1)
        else:
            logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            loss = self.cross_entropy_loss(shift_logits.float(), shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        use_cache=False,
        **kwargs,
    ):
        past_length = 0
        if past_key_values is not None:
            # Past key values are always initialized with a `Cache` object -> no need for if-else anymore
            past_length = cache_position[0] if cache_position is not None else get_seq_length(past_key_values)
            max_cache_length = get_max_length(past_key_values) if get_max_length(past_key_values) is not None else None
            cache_length = past_length if max_cache_length is None else ops.minimum(max_cache_length, past_length)

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as input)

            if attention_mask is not None and int(attention_mask.sum(-1).max()) > input_ids.shape[1]:
                input_ids = input_ids[:, -(int(attention_mask.sum(-1).max()) - int(past_length)) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, int(past_length) :]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.to(ms.int32).cumsum(-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 1)
            if past_key_values and past_length > 0:
                cur_len = attention_mask.sum(-1).max()
                position_ids = position_ids[:, cur_len - input_ids.shape[1] : cur_len]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_length == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # TODO: use `next_tokens` directly instead.
            if not isinstance(input_ids, Tensor):
                input_ids = Tensor(input_ids, dtype=ms.int32)

            # Padding to max_len when no cache
            if past_key_values is None:
                pad_len = max(0, attention_mask.shape[1] - input_ids.shape[1])
                input_ids = ops.pad(input_ids, (0, pad_len), value=0)

            model_inputs = {"input_ids": input_ids}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]

        if cache_position is None:
            cache_position = ops.arange(past_length, past_length + input_length)
        elif use_cache:
            if input_length < cache_position.shape[0]:
                assert cache_position.shape[0] == attention_mask.shape[-1]
                cur_len = int(attention_mask.sum(-1).max())
                cache_position = cache_position[cur_len - input_length : cur_len]
            else:
                cache_position = cache_position[-input_length:]

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": ms.mutable(past_key_values) if past_key_values is not None else None,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        raise NotImplementedError


@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Dense(config.hidden_size, self.num_labels, has_bias=False)

        self.loss_fct_regression = nn.MSELoss()
        self.loss_fct_single_label_classification = nn.CrossEntropyLoss()
        self.loss_fct_multi_label_classification = nn.BCEWithLogitsLoss()

        problem_type_map = {
            "regression": 0,
            "single_label_classification": 1,
            "multi_label_classification": 2,
            None: None,
        }
        self.problem_type = problem_type_map[config.problem_type]
        self.pad_token_id = config.pad_token_id

        _name_list = ["output_attentions", "output_hidden_states", "use_return_dict", "pretraining_tp", "vocab_size"]
        for name in _name_list:
            setattr(self, name, getattr(config, name))

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def construct(
        self,
        input_ids: ms.Tensor = None,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[ms.Tensor, ms.Tensor]]] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        labels: Optional[ms.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = False,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`ms.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        # if self.pad_token_id is None and batch_size != 1:
        #     raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = ops.equal(input_ids, self.pad_token_id).to(ms.int32).argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
            else:
                sequence_lengths = -1

        pooled_logits = logits[ops.arange(batch_size), sequence_lengths]

        loss = None
        if labels is not None:
            if self.problem_type is None:
                if self.num_labels == 1:
                    problem_type = 0  # "regression"
                elif self.num_labels > 1 and (labels.dtype in (ms.int32, ms.int64)):
                    problem_type = 1  # "single_label_classification"
                else:
                    problem_type = 2  # "multi_label_classification"
            else:
                problem_type = self.problem_type

            if problem_type == 0:  # "regression"
                if self.num_labels == 1:
                    loss = self.loss_fct_regression(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = self.loss_fct_regression(pooled_logits, labels)
            elif problem_type == 1:  # "single_label_classification"
                loss = self.loss_fct_single_label_classification(
                    pooled_logits.view(-1, self.num_labels), labels.view(-1).int()
                )
            elif problem_type == 2:  # "multi_label_classification"
                loss = self.loss_fct_multi_label_classification(pooled_logits, labels)

        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
