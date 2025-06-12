# coding=utf-8
# Copyright 2024 Meta Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""MindSpore Chameleon model."""

import math
from functools import cached_property
from typing import Optional, Tuple, Union

import numpy as np
from transformers.models.chameleon.configuration_chameleon import ChameleonConfig, ChameleonVQVAEConfig

import mindspore as ms
import mindspore.mint as mint
import mindspore.mint.nn.functional as F
import mindspore.ops as ops
from mindspore import nn
from mindspore.mint.nn import CrossEntropyLoss

from mindone.models.utils import normal_, ones_, zeros_

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...mindspore_adapter import dtype_to_min
from ...mindspore_utils import ALL_LAYERNORM_LAYERS
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import logging

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "ChameleonConfig"
_CHECKPOINT_FOR_DOC = "meta/chameleon-7b"
_EXPECTED_OUTPUT_SHAPE = [1, 7, 4096]
_SEQ_CLASS_EXPECTED_LOSS = 1.03
_SEQ_CLASS_EXPECTED_OUTPUT = "'LABEL_0'"


class ChameleonRMSNorm(nn.Cell):
    def __init__(self, hidden_size, eps=1e-6):
        """
        ChameleonRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = ms.Parameter(np.ones(hidden_size, dtype=np.float32))
        self.variance_epsilon = eps

    def construct(self, hidden_states):
        result, _ = ops.rms_norm(
            hidden_states.to(ms.float32), self.weight.to(ms.float32), epsilon=self.variance_epsilon
        )
        return result.to(hidden_states.dtype)


ALL_LAYERNORM_LAYERS.append(ChameleonRMSNorm)


class ChameleonRotaryEmbedding(nn.Cell):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (np.arange(0, self.dim, 2, dtype=np.float32) / self.dim))
        self.inv_freq = ms.Tensor(inv_freq)
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings

    def construct(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand((position_ids.shape[0], -1, 1))
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = mint.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class ChameleonLinearScalingRotaryEmbedding(ChameleonRotaryEmbedding):
    """ChameleonRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def construct(self, x, position_ids):
        # difference to the original RoPE: a scaling factor is applied to the position ids
        position_ids = position_ids.float() / self.scaling_factor
        cos, sin = super().construct(x, position_ids)
        return cos, sin


class ChameleonDynamicNTKScalingRotaryEmbedding(ChameleonRotaryEmbedding):
    """ChameleonRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def construct(self, x, position_ids):
        # difference to the original RoPE: inv_freq is recomputed when the sequence length > original length
        seq_len = mint.max(position_ids) + 1
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (mint.arange(0, self.dim, 2, dtype=ms.int64).to(dtype=ms.float32) / self.dim))
            self.inv_freq = inv_freq

        cos, sin = super().construct(x, position_ids)
        return cos, sin


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mint.cat((-x2, x1), dim=-1)


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


class ChameleonMLP(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = mint.nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = mint.nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = mint.nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def construct(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class ChameleonLayerNorm(mint.nn.LayerNorm):
    """
    LayerNorm but computes stats only over the last dim because Chameleon applies gamma and beta
    from each shard separately to each head, instead of reducing. We can apply each head's own
    gamma/beta by repeat-interleaving weights from each shard, but the stats have to be computed
    in the last dimension. This module applies gamma/beta manually to fulfill this requirement.
    """

    def __init__(self, hidden_size, *args, **kwargs):
        super().__init__(hidden_size, *args, **kwargs)
        self.normalized_shape = (hidden_size[-1],)

    def construct(self, hidden_states):
        hidden_states = F.layer_norm(hidden_states, self.normalized_shape, None, None, eps=1e-5)
        hidden_states = hidden_states * self.weight + self.bias
        return hidden_states


def repeat_kv(hidden_states: ms.Tensor, n_rep: int) -> ms.Tensor:
    """
    This is the equivalent of mint.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand((batch, num_key_value_heads, n_rep, slen, head_dim))
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class ChameleonAttention(nn.Cell):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: ChameleonConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the construct call if caching is used. Please make sure to provide a `layer_idx` "
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
        self.model_parallel_size = config.model_parallel_size

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = mint.nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = mint.nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = mint.nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = mint.nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self.q_norm = ChameleonLayerNorm((self.num_heads, self.head_dim))
        self.k_norm = ChameleonLayerNorm((self.num_key_value_heads, self.head_dim))
        self._init_rope()

    # copied from transformers.models.llama.modeling_llama.LlamaAttention._init_rope with Llama->Chameleon
    # TODO(joao): add me back asap :)
    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = ChameleonRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = ChameleonLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = ChameleonDynamicNTKScalingRotaryEmbedding(
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
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[ms.Tensor] = None,
        **kwargs,
    ) -> Tuple[ms.Tensor, Optional[ms.Tensor], Optional[Tuple[ms.Tensor]]]:
        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.reshape(-1, self.num_heads, self.head_dim)
        query_states = self.q_norm(query_states)

        key_states = key_states.reshape(-1, self.num_key_value_heads, self.head_dim)
        key_states = self.k_norm(key_states)

        query_states = query_states.reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; position_ids needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = mint.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=ms.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = mint.matmul(attn_weights, value_states)

        if attn_output.shape != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.shape}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class ChameleonSdpaAttention(ChameleonAttention):
    """
    Chameleon attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `ChameleonAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[ms.Tensor] = None,
        **kwargs,
    ) -> Tuple[ms.Tensor, Optional[ms.Tensor], Optional[Tuple[ms.Tensor]]]:
        # TODO: MindSpore does not have SDPA API now, we use eager temporarily
        return super().construct(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )


CHAMELEON_ATTENTION_CLASSES = {"eager": ChameleonAttention, "sdpa": ChameleonAttention}


class ChameleonDecoderLayer(nn.Cell):
    def __init__(self, config: ChameleonConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = CHAMELEON_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = ChameleonMLP(config)
        self.input_layernorm = ChameleonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = ChameleonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_value: Optional[Cache] = None,
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
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class ChameleonSwinDecoderLayer(nn.Cell):
    def __init__(self, config: ChameleonConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = CHAMELEON_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = ChameleonMLP(config)
        self.input_layernorm = ChameleonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = ChameleonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[ms.Tensor] = None,
        **kwargs,
    ) -> Tuple[ms.Tensor, Optional[Tuple[ms.Tensor, ms.Tensor]]]:
        """
        Args:
            hidden_states (`ms.Tensor`):
                input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`ms.Tensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            position_ids (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings
            past_key_value (`Tuple(ms.Tensor)`, *optional*): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            cache_position (`ms.Tensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
        """

        residual = hidden_states

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class ChameleonVQVAEVectorQuantizer(nn.Cell):
    """
    A module for vector quantization using learned embedding vectors.

    This module implements the quantization process similar to te one described in
    the VQ-VAE (Vector Quantized Variational AutoEncoder) paper. It quantizes continuous
    input vectors into discrete codebook vectors, which are learned during training.
    Current implementation improves over previous ones by avoiding costly matrix multiplications
    and allowing for post-hoc remapping of indices.
    """

    def __init__(self, config):
        super().__init__()
        self.num_embeddings = config.num_embeddings
        self.embedding_dim = config.embed_dim
        self.beta = getattr(config, "beta", 0.25)

        self.embedding = mint.nn.Embedding(self.num_embeddings, self.embedding_dim)

    def construct(self, hidden_state: ms.Tensor):
        hidden_state = hidden_state.permute(0, 2, 3, 1).contiguous()
        hidden_state_flattened = hidden_state.view(-1, self.embedding_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        distances = (
            mint.sum(hidden_state_flattened**2, dim=1, keepdim=True)
            + mint.sum(self.embedding.weight**2, dim=1)
            - 2 * mint.einsum("bd,dn->bn", hidden_state_flattened, self.embedding.weight.transpose(0, 1))
        )

        min_encoding_indices = mint.argmin(distances, dim=1)
        hidden_state_quant = self.embedding(min_encoding_indices).view(hidden_state.shape)

        # compute loss for embedding
        loss = mint.mean((ops.stop_gradient(hidden_state_quant) - hidden_state) ** 2) + self.beta * mint.mean(
            (hidden_state_quant - ops.stop_gradient(hidden_state)) ** 2
        )

        # preserve gradients
        hidden_state_quant = hidden_state + ops.stop_gradient(hidden_state_quant - hidden_state)

        # reshape back to match original input shape
        hidden_state_quant = hidden_state_quant.permute(0, 3, 1, 2).contiguous()

        return hidden_state_quant, loss, min_encoding_indices


class ChameleonVQVAEEncoderConvDownsample(nn.Cell):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = mint.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def construct(self, hidden_states):
        # no asymmetric padding in torch conv, must do it ourselves
        hidden_states = F.pad(hidden_states, pad=(0, 1, 0, 1), mode="constant", value=0)
        hidden_states = self.conv(hidden_states)
        return hidden_states


class ChameleonVQVAEEncoderResnetBlock(nn.Cell):
    def __init__(
        self,
        config,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = mint.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = mint.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = mint.nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.dropout = mint.nn.Dropout(config.dropout)
        self.conv2 = mint.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = mint.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = mint.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def construct(self, hidden_states):
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states *= mint.sigmoid(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)
        hidden_states *= mint.sigmoid(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                residual = self.conv_shortcut(residual)
            else:
                residual = self.nin_shortcut(residual)

        return residual + hidden_states


class ChameleonVQVAEEncoderAttnBlock(nn.Cell):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = mint.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = mint.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = mint.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = mint.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = mint.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def construct(self, hidden_states):
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        query_states = self.q(hidden_states)
        key_states = self.k(hidden_states)
        value_states = self.v(hidden_states)

        # compute attention
        batch_size, channels, height, width = query_states.shape
        query_states = query_states.reshape(batch_size, channels, height * width).permute(0, 2, 1)
        key_states = key_states.reshape(batch_size, channels, height * width)
        attn_weights = mint.bmm(query_states, key_states)
        attn_weights = attn_weights * (int(channels) ** (-0.5))
        attn_weights = F.softmax(attn_weights, dim=2)

        # attend to values
        value_states = value_states.reshape(batch_size, channels, height * width)
        attn_weights = attn_weights.permute(0, 2, 1)
        attn_output = mint.bmm(value_states, attn_weights).reshape(batch_size, channels, height, width)

        attn_output = self.proj_out(attn_output)
        return residual + attn_output


class ChameleonVQVAEEncoder(nn.Cell):
    def __init__(self, config):
        super().__init__()

        self.num_resolutions = len(config.channel_multiplier)
        self.num_res_blocks = config.num_res_blocks
        base_channels = config.base_channels
        resolution = config.resolution
        in_channels = config.in_channels
        double_latent = config.double_latent
        latent_channels = config.latent_channels
        channel_multiplier = config.channel_multiplier

        self.conv_in = mint.nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_channel_multiplier = (1,) + tuple(channel_multiplier)
        self.in_channel_multiplier = in_channel_multiplier
        self.down = nn.CellList()
        for i_level in range(self.num_resolutions):
            block = nn.CellList()
            attn = nn.CellList()
            block_in = base_channels * in_channel_multiplier[i_level]
            block_out = base_channels * channel_multiplier[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ChameleonVQVAEEncoderResnetBlock(
                        config=config,
                        in_channels=block_in,
                        out_channels=block_out,
                    )
                )
                block_in = block_out
                if (
                    config.attn_resolutions is not None
                    and curr_res in config.attn_resolutions
                    and config.attn_type == "vanilla"
                ):
                    attn.append(ChameleonVQVAEEncoderAttnBlock(block_in))

            down = nn.Cell()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = ChameleonVQVAEEncoderConvDownsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        self.mid = nn.Cell()
        self.mid.block_1 = ChameleonVQVAEEncoderResnetBlock(
            config=config,
            in_channels=block_in,
            out_channels=block_in,
        )
        self.mid.attn_1 = (
            ChameleonVQVAEEncoderAttnBlock(block_in) if config.attn_type == "vanilla" else mint.nn.Identity()
        )
        self.mid.block_2 = ChameleonVQVAEEncoderResnetBlock(
            config=config,
            in_channels=block_in,
            out_channels=block_in,
        )

        self.norm_out = mint.nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = mint.nn.Conv2d(
            block_in,
            2 * latent_channels if double_latent else latent_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def construct(self, pixel_values: ms.Tensor):
        # downsampling
        hidden_states = [self.conv_in(pixel_values)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                hidden_state = self.down[i_level].block[i_block](
                    hidden_states[-1],
                )
                if len(self.down[i_level].attn) > 0:
                    hidden_state = self.down[i_level].attn[i_block](hidden_state)
                hidden_states.append(hidden_state)
            if i_level != self.num_resolutions - 1:
                hidden_states.append(self.down[i_level].downsample(hidden_states[-1]))

        # middle
        last_hidden_state = hidden_states[-1]
        last_hidden_state = self.mid.block_1(last_hidden_state)
        last_hidden_state = self.mid.attn_1(last_hidden_state)
        last_hidden_state = self.mid.block_2(last_hidden_state)

        # end
        last_hidden_state = self.norm_out(last_hidden_state)
        last_hidden_state *= mint.sigmoid(last_hidden_state)
        last_hidden_state = self.conv_out(last_hidden_state)
        return last_hidden_state


class ChameleonImageVocabularyMapping:
    """
    A class for mapping discrete image tokens from VQGAN to BPE tokens.
    """

    def __init__(self, vocab_map):
        self.vocab_map = vocab_map
        self.image_token_id = vocab_map.get("<image>")

    @cached_property
    def val2name(self):
        return {v: k for k, v in self.vocab_map.items()}

    @cached_property
    def image_tokens(self):
        return sorted([val for name, val in self.vocab_map.items() if name.startswith("IMGIMG")])

    @cached_property
    def bpe2img(self):
        img_tkn_chr_mapping = {chr(ord("A") + i): str(i) for i in range(10)}

        def remap(old_name: str) -> str:
            return "".join(img_tkn_chr_mapping.get(c, c) for c in old_name[len("IMGIMG") : -1])

        return {tok: int(remap(self.val2name[tok])) for tok in self.image_tokens}

    @cached_property
    def img2bpe(self):
        return {v: k for k, v in self.bpe2img.items()}

    @cached_property
    def bpe2img_search_tensors(self):
        return ms.tensor(sorted(self.bpe2img.keys())), ms.tensor(sorted(self.bpe2img.values()))

    @cached_property
    def img2bpe_mapping_tensor(self):
        mapping = mint.zeros(max(self.img2bpe.keys()) + 1, dtype=ms.int32)
        for k, v in self.img2bpe.items():
            mapping[k] = v
        return mapping

    def convert_img2bpe(self, img_batch: ms.Tensor) -> ms.Tensor:
        img_tokens = self.img2bpe_mapping_tensor[img_batch]
        return img_tokens


class ChameleonPreTrainedModel(PreTrainedModel):
    config_class = ChameleonConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ChameleonDecoderLayer", "ChameleonSwinDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values", "causal_mask"]
    _supports_flash_attn_2 = False
    _supports_sdpa = True
    _supports_quantized_cache = False
    _supports_cache_class = False
    _supports_static_cache = False
    _supports_param_buffer_assignment = False

    def _init_weights(self, module):
        std = self.config.initializer_range

        if isinstance(module, (mint.nn.Linear, mint.nn.Conv2d)):
            normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                zeros_(module.bias)
        elif isinstance(module, (mint.nn.GroupNorm, mint.nn.LayerNorm)):
            zeros_(module.bias)
            ones_(module.weight)
        elif isinstance(module, ChameleonRMSNorm):
            ones_(module.weight)
        elif isinstance(module, mint.nn.Embedding):
            normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight[module.padding_idx] = 0


class ChameleonVQVAE(ChameleonPreTrainedModel):
    config_class = ChameleonVQVAEConfig
    _no_split_modules = ["ChameleonVQVAEVectorQuantizer"]

    def __init__(self, config: ChameleonVQVAEConfig):
        super().__init__(config)

        self.encoder = ChameleonVQVAEEncoder(config)
        self.quantize = ChameleonVQVAEVectorQuantizer(config)
        self.quant_conv = mint.nn.Conv2d(config.latent_channels, config.embed_dim, 1)
        self.post_quant_conv = mint.nn.Conv2d(config.embed_dim, config.latent_channels, 1)

    def encode(self, pixel_values: ms.Tensor):
        hidden_states = self.encoder(pixel_values)
        hidden_states = self.quant_conv(hidden_states)
        quant, emb_loss, indices = self.quantize(hidden_states)
        return quant, emb_loss, indices


class ChameleonModel(ChameleonPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`ChameleonDecoderLayer`]

    Args:
        config: ChameleonConfig
    """

    def __init__(self, config: ChameleonConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = mint.nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.vocabulary_mapping = ChameleonImageVocabularyMapping(config.vocabulary_map)
        decoder_layer = ChameleonDecoderLayer if not self.config.swin_norm else ChameleonSwinDecoderLayer
        self.layers = nn.CellList([decoder_layer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = ChameleonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.vqmodel = ChameleonVQVAE._from_config(config.vq_config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_image_tokens(self, pixel_values: ms.Tensor):
        """
        Tokenizes images into discrete tokens with VQGAN module. Converts
        obtained image tokens into BPE tokens and wraps with "boi" and "eoi"
        special tokens.

        Args:
            pixel_values (`ms.Tensor` of shape `(batch_size, num_channels, image_size, image_size)):
                The tensors corresponding to the input images.
        """
        batch_size = pixel_values.shape[0]
        _, _, image_toks = self.vqmodel.encode(pixel_values)
        bpe_toks = self.vocabulary_mapping.convert_img2bpe(image_toks)
        bpe_toks = bpe_toks.view(batch_size, -1)
        return bpe_toks

    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        pixel_values: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[ms.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        if pixel_values is not None:
            image_tokens = self.get_image_tokens(pixel_values)
            special_image_mask = input_ids == self.vocabulary_mapping.image_token_id
            if input_ids[special_image_mask].numel() != image_tokens.numel():
                n_image_tokens_in_text = (input_ids == self.vocabulary_mapping.image_token_id).sum()
                n_image_features = image_tokens.shape[0] * image_tokens.shape[1]
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens_in_text}, features {n_image_features}"
                )
            image_tokens = image_tokens.to(input_ids.dtype)
            input_ids = input_ids.masked_scatter(special_image_mask, image_tokens)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = mint.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1])

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = ms.recompute(
                    decoder_layer,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    # Copied from transformers.models.llama.modeling_llama.LlamaModel._update_causal_mask
    def _update_causal_mask(
        self,
        attention_mask: ms.Tensor,
        input_tensor: ms.Tensor,
        cache_position: ms.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0

        dtype = input_tensor.dtype
        sequence_length = input_tensor.shape[1]
        target_length = (
            attention_mask.shape[-1]
            if isinstance(attention_mask, ms.Tensor)
            else past_seen_tokens + sequence_length + 1
        )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        return causal_mask

    @staticmethod
    # Copied from transformers.models.llama.modeling_llama.LlamaPreTrainedModel._prepare_4d_causal_attention_mask_with_cache_position
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: ms.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: ms.Type,
        cache_position: ms.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`ms.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            cache_position (`ms.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`ms.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = dtype_to_min(dtype)
            # TODO: to mint.full
            causal_mask = ops.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype)
            if sequence_length != 1:
                causal_mask = mint.triu(causal_mask, diagonal=1)
            causal_mask *= mint.arange(target_length) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand((batch_size, 1, -1, -1))
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


class ChameleonForConditionalGeneration(ChameleonPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = ChameleonModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = mint.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

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

    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        pixel_values: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        labels: Optional[ms.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[ms.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
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
        logits = self.lm_head(hidden_states)

        # Disallow image tokens which does not include special begin-image and end-image tokens
        image_tokens = self.model.vocabulary_mapping.image_tokens
        logits[:, :, image_tokens] = dtype_to_min(logits.dtype)

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            loss = loss_fct(shift_logits, shift_labels)

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
        pixel_values=None,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to construct image inputs to the model

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            pixel_values=pixel_values,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            **kwargs,
        )

        if cache_position[0] != 0:
            # If we're in cached decoding stage, pixel values should be `None` because input ids do not contain special image token anymore
            # Otherwise we need pixel values to be passed to model
            model_inputs["pixel_values"] = None

        return model_inputs


__all__ = ["ChameleonForConditionalGeneration", "ChameleonModel", "ChameleonPreTrainedModel", "ChameleonVQVAE"]
