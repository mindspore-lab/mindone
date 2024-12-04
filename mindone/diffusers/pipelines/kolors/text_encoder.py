# Copyright 2024 ChatGLM3-6B Model Team, Kwai-Kolors Team and The HuggingFace Team. All rights reserved.
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
from typing import List, Optional, Tuple

import numpy as np
from transformers import PretrainedConfig

import mindspore as ms
from mindspore import nn, ops

from mindone.transformers import MSPreTrainedModel
from mindone.transformers.modeling_outputs import BaseModelOutputWithPast

from ...models.normalization import LayerNorm, RMSNorm
from ...utils import logging

logger = logging.get_logger(__name__)


class ChatGLMConfig(PretrainedConfig):
    model_type = "chatglm"

    def __init__(
        self,
        num_layers=28,
        padded_vocab_size=65024,
        hidden_size=4096,
        ffn_hidden_size=13696,
        kv_channels=128,
        num_attention_heads=32,
        seq_length=2048,
        hidden_dropout=0.0,
        classifier_dropout=None,
        attention_dropout=0.0,
        layernorm_epsilon=1e-5,
        rmsnorm=True,
        apply_residual_connection_post_layernorm=False,
        post_layer_norm=True,
        add_bias_linear=False,
        add_qkv_bias=False,
        bias_dropout_fusion=True,
        multi_query_attention=False,
        multi_query_group_num=1,
        apply_query_key_layer_scaling=True,
        attention_softmax_in_fp32=True,
        fp32_residual_connection=False,
        quantization_bit=0,
        pre_seq_len=None,
        prefix_projection=False,
        **kwargs,
    ):
        self.num_layers = num_layers
        self.vocab_size = padded_vocab_size
        self.padded_vocab_size = padded_vocab_size
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.kv_channels = kv_channels
        self.num_attention_heads = num_attention_heads
        self.seq_length = seq_length
        self.hidden_dropout = hidden_dropout
        self.classifier_dropout = classifier_dropout
        self.attention_dropout = attention_dropout
        self.layernorm_epsilon = layernorm_epsilon
        self.rmsnorm = rmsnorm
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.post_layer_norm = post_layer_norm
        self.add_bias_linear = add_bias_linear
        self.add_qkv_bias = add_qkv_bias
        self.bias_dropout_fusion = bias_dropout_fusion
        self.multi_query_attention = multi_query_attention
        self.multi_query_group_num = multi_query_group_num
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.fp32_residual_connection = fp32_residual_connection
        self.quantization_bit = quantization_bit
        self.pre_seq_len = pre_seq_len
        self.prefix_projection = prefix_projection
        super().__init__(**kwargs)


class CoreAttention(nn.Cell):
    def __init__(self, config: ChatGLMConfig, layer_number):
        super(CoreAttention, self).__init__()

        self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)

        projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_partition = projection_size
        self.hidden_size_per_attention_head = projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff
        self.coeff = coeff

        self.attention_dropout = nn.Dropout(p=config.attention_dropout)
        self.min_fp16 = ms.tensor(np.finfo(np.float16).min, dtype=ms.float16)
        self.min_fp32 = ms.tensor(np.finfo(np.float32).min, dtype=ms.float32)
        self.min_fp64 = ms.tensor(np.finfo(np.float64).min, dtype=ms.float64)
        self.min_bf16 = ms.tensor(float.fromhex("-0x1.fe00000000000p+127"), dtype=ms.bfloat16)

    def dtype_to_min(self, dtype):
        if dtype == ms.float16:
            return self.min_fp16
        if dtype == ms.float32:
            return self.min_fp32
        if dtype == ms.float64:
            return self.min_fp64
        if dtype == ms.bfloat16:
            return self.min_bf16
        else:
            raise ValueError(f"Only support get minimum value of (float16, ), but got {dtype}")

    def construct(self, query_layer, key_layer, value_layer, attention_mask):
        # Raw attention scores

        # [b, np, sq, sk]
        output_size = (query_layer.shape[1], query_layer.shape[2], query_layer.shape[0], key_layer.shape[0])

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

        # preallocting input tensor: [b * np, sq, sk]
        matmul_input_buffer = ops.zeros(
            (output_size[0] * output_size[1], output_size[2], output_size[3]),
            dtype=query_layer.dtype,
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = ops.baddbmm(
            matmul_input_buffer,
            query_layer.swapaxes(0, 1),  # [b * np, sq, hn]
            key_layer.swapaxes(0, 1).swapaxes(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(output_size)

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        if self.attention_softmax_in_fp32:
            attention_scores = attention_scores.float()
        if self.coeff is not None:
            attention_scores = attention_scores * self.coeff
        if attention_mask is None and attention_scores.shape[2] == attention_scores.shape[3]:
            attention_mask = ops.ones((output_size[0], 1, output_size[2], output_size[3]))
            attention_mask = ops.tril(attention_mask).bool()
            attention_mask = ops.logical_not(attention_mask)
        if attention_mask is not None:
            attention_scores = ops.masked_fill(
                attention_scores, attention_mask, self.dtype_to_min(attention_scores.dtype)
            )
        attention_probs = ops.softmax(attention_scores, axis=-1)
        attention_probs = attention_probs.type_as(value_layer)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_dropout(attention_probs)
        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.shape[1], value_layer.shape[2], query_layer.shape[0], value_layer.shape[3])
        # change view [sk, b * np, hn]
        value_layer = value_layer.view(value_layer.shape[0], output_size[0] * output_size[1], -1)
        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
        # matmul: [b * np, sq, hn]
        context_layer = ops.bmm(attention_probs, value_layer.swapaxes(0, 1))
        # change view [b, np, sq, hn]
        context_layer = context_layer.view(output_size)
        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3)
        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.shape[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer


def split_tensor_along_last_dim(
    tensor: ms.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
) -> List[ms.Tensor]:
    """Split a tensor along its last dimension.

    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.

    Returns:
        A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = tensor.shape[last_dim] // num_partitions
    # Split.
    tensor_list = ops.split(tensor, last_dim_size, axis=last_dim)
    if contiguous_split_chunks:
        return tuple(chunk for chunk in tensor_list)

    return tensor_list


def apply_rotary_pos_emb(x: ms.Tensor, rope_cache: ms.Tensor) -> ms.Tensor:
    # x: [sq, b, np, hn]
    sq, _, np, _ = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # truncate to support variable sizes
    rope_cache = rope_cache[:sq]
    xshaped = x.reshape(sq, -1, np, rot_dim // 2, 2)
    rope_cache = rope_cache.view(sq, -1, 1, xshaped.shape[3], 2)
    x_out2 = ops.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(start_dim=3)
    return ops.cat((x_out2, x_pass), axis=-1)


class SelfAttention(nn.Cell):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h] and returns output of the same size.
    """

    def __init__(self, config: ChatGLMConfig, layer_number):
        super(SelfAttention, self).__init__()
        self.layer_number = max(1, layer_number)

        self.projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_attention_head = self.projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads

        self.multi_query_attention = config.multi_query_attention
        self.qkv_hidden_size = 3 * self.projection_size
        if self.multi_query_attention:
            self.num_multi_query_groups_per_partition = config.multi_query_group_num
            self.qkv_hidden_size = (
                self.projection_size + 2 * self.hidden_size_per_attention_head * config.multi_query_group_num
            )
        self.query_key_value = nn.Dense(
            config.hidden_size,
            self.qkv_hidden_size,
            has_bias=config.add_bias_linear or config.add_qkv_bias,
        )

        self.core_attention = CoreAttention(config, self.layer_number)

        # Output.
        self.dense = nn.Dense(
            self.projection_size,
            config.hidden_size,
            has_bias=config.add_bias_linear,
        )

    def _allocate_memory(self, inference_max_sequence_len, batch_size, dtype=None):
        if self.multi_query_attention:
            num_attention_heads = self.num_multi_query_groups_per_partition
        else:
            num_attention_heads = self.num_attention_heads_per_partition
        return ops.zeros(
            (inference_max_sequence_len, batch_size, num_attention_heads, self.hidden_size_per_attention_head),
            dtype=dtype,
        )

    def construct(self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True):
        # hidden_states: [sq, b, h]

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        # =====================
        # Query, Key, and Value
        # =====================

        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        mixed_x_layer = self.query_key_value(hidden_states)

        if self.multi_query_attention:
            (query_layer, key_layer, value_layer) = mixed_x_layer.split(
                [
                    self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                ],
                axis=-1,
            )
            query_layer = query_layer.view(
                query_layer.shape[:-1] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            )
            key_layer = key_layer.view(
                key_layer.shape[:-1] + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            )
            value_layer = value_layer.view(
                value_layer.shape[:-1]
                + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            )
        else:
            new_tensor_shape = mixed_x_layer.shape[:-1] + (
                self.num_attention_heads_per_partition,
                3 * self.hidden_size_per_attention_head,
            )
            mixed_x_layer = mixed_x_layer.view(new_tensor_shape)

            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

        # apply relative positional encoding (rotary embedding)
        if rotary_pos_emb is not None:
            query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
            key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)

        # adjust key and value for inference
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            key_layer = ops.cat((cache_k, key_layer), axis=0)
            value_layer = ops.cat((cache_v, value_layer), axis=0)
        if use_cache:
            kv_cache = (key_layer, value_layer)
        else:
            kv_cache = None

        if self.multi_query_attention:
            key_layer = key_layer.unsqueeze(-2)
            key_layer = key_layer.broadcast_to(
                (-1, -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1)
            )
            key_layer = key_layer.view(
                key_layer.shape[:2] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            )
            value_layer = value_layer.unsqueeze(-2)
            value_layer = value_layer.broadcast_to(
                (-1, -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1)
            )
            value_layer = value_layer.view(
                value_layer.shape[:2] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            )

        # ==================================
        # core attention computation
        # ==================================

        context_layer = self.core_attention(query_layer, key_layer, value_layer, attention_mask)

        # =================
        # Output. [sq, b, h]
        # =================

        output = self.dense(context_layer)

        return output, kv_cache


class MLP(nn.Cell):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h hidden dimension, perform nonlinear transformation,
    and project the state back into h hidden dimension.
    """

    def __init__(self, config: ChatGLMConfig):
        super(MLP, self).__init__()

        self.add_bias = config.add_bias_linear

        # Project to 4h. If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        self.dense_h_to_4h = nn.Dense(
            config.hidden_size,
            config.ffn_hidden_size * 2,
            has_bias=self.add_bias,
        )

        def swiglu(x):
            x = ops.chunk(x, 2, axis=-1)
            return ops.silu(x[0]) * x[1]

        self.activation_func = swiglu

        # Project back to h.
        self.dense_4h_to_h = nn.Dense(
            config.ffn_hidden_size,
            config.hidden_size,
            has_bias=self.add_bias,
        )

    def construct(self, hidden_states):
        # [s, b, 4hp]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        # [s, b, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        return output


class GLMBlock(nn.Cell):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an output of the same size.
    """

    def __init__(self, config: ChatGLMConfig, layer_number):
        super(GLMBlock, self).__init__()
        self.layer_number = layer_number

        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm

        self.fp32_residual_connection = config.fp32_residual_connection

        LayerNormFunc = RMSNorm if config.rmsnorm else LayerNorm
        # Layernorm on the input data.
        self.input_layernorm = LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon)

        # Self attention.
        self.self_attention = SelfAttention(config, layer_number)
        self.hidden_dropout = config.hidden_dropout

        # Layernorm on the attention output
        self.post_attention_layernorm = LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon)

        # MLP
        self.mlp = MLP(config)

    def construct(
        self,
        hidden_states,
        attention_mask,
        rotary_pos_emb,
        kv_cache=None,
        use_cache=True,
    ):
        # hidden_states: [s, b, h]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output, kv_cache = self.self_attention(
            layernorm_output, attention_mask, rotary_pos_emb, kv_cache=kv_cache, use_cache=use_cache
        )

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        layernorm_input = ops.dropout(attention_output, p=self.hidden_dropout, training=self.training)
        layernorm_input = residual + layernorm_input

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output = self.mlp(layernorm_output)

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        output = ops.dropout(mlp_output, p=self.hidden_dropout, training=self.training)
        output = residual + output

        return output, kv_cache


class GLMTransformer(nn.Cell):
    """Transformer class."""

    def __init__(self, config: ChatGLMConfig):
        super(GLMTransformer, self).__init__()

        self.fp32_residual_connection = config.fp32_residual_connection
        self.post_layer_norm = config.post_layer_norm

        # Number of layers.
        self.num_layers = config.num_layers

        # Transformer layers.
        def build_layer(layer_number):
            return GLMBlock(config, layer_number)

        self.layers = nn.CellList([build_layer(i + 1) for i in range(self.num_layers)])

        if self.post_layer_norm:
            LayerNormFunc = RMSNorm if config.rmsnorm else LayerNorm
            # Final layer norm before output.
            self.final_layernorm = LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon)

        self.gradient_checkpointing = False

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def construct(
        self,
        hidden_states,
        attention_mask,
        rotary_pos_emb,
        kv_caches=None,
        use_cache: Optional[bool] = True,
        output_hidden_states: Optional[bool] = False,
    ):
        if not kv_caches:
            kv_caches = [None for _ in range(self.num_layers)]
        presents = () if use_cache else None
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        all_self_attentions = None
        all_hidden_states = () if output_hidden_states else None
        for index in range(self.num_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer = self._get_layer(index)
            if self.gradient_checkpointing and self.training:
                raise NotImplementedError("Gradient checkpointing is not yet supported.")
            else:
                layer_ret = layer(
                    hidden_states, attention_mask, rotary_pos_emb, kv_cache=kv_caches[index], use_cache=use_cache
                )
            hidden_states, kv_cache = layer_ret
            if use_cache:
                presents = presents + (kv_cache,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Final layer norm.
        if self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states, presents, all_hidden_states, all_self_attentions


class ChatGLMPreTrainedModel(MSPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    is_parallelizable = False
    supports_gradient_checkpointing = True
    config_class = ChatGLMConfig
    base_model_prefix = "transformer"
    _no_split_modules = ["GLMBlock"]

    def _init_weights(self, module: nn.Cell):
        """Initialize the weights."""
        return

    def get_masks(self, input_ids, past_key_values, padding_mask=None):
        batch_size, seq_length = input_ids.shape
        full_attention_mask = ops.ones((batch_size, seq_length, seq_length))
        full_attention_mask = ops.tril(full_attention_mask)
        past_length = 0
        if past_key_values:
            past_length = past_key_values[0][0].shape[0]
        if past_length:
            full_attention_mask = ops.cat(
                (ops.ones((batch_size, seq_length, past_length)), full_attention_mask), axis=-1
            )
        if padding_mask is not None:
            full_attention_mask = full_attention_mask * padding_mask.unsqueeze(1)
        if not past_length and padding_mask is not None:
            full_attention_mask -= padding_mask.unsqueeze(-1) - 1
        full_attention_mask = (full_attention_mask < 0.5).bool()
        full_attention_mask = full_attention_mask.unsqueeze(1)
        return full_attention_mask

    def get_position_ids(self, input_ids):
        batch_size, seq_length = input_ids.shape
        position_ids = ops.arange(seq_length, dtype=ms.int32).unsqueeze(0).tile((batch_size, 1))
        return position_ids

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, GLMTransformer):
            module.gradient_checkpointing = value


def default_init(cls, *args, **kwargs):
    return cls(*args, **kwargs)


class Embedding(nn.Cell):
    """Language model embeddings."""

    def __init__(self, config: ChatGLMConfig):
        super(Embedding, self).__init__()

        self.hidden_size = config.hidden_size
        # Word embeddings (parallel).
        self.word_embeddings = nn.Embedding(
            config.padded_vocab_size,
            self.hidden_size,
        )
        self.fp32_residual_connection = config.fp32_residual_connection

    def construct(self, input_ids):
        # Embeddings.
        words_embeddings = self.word_embeddings(input_ids)
        embeddings = words_embeddings
        # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
        embeddings = embeddings.swapaxes(0, 1)
        # If the input flag for fp32 residual connection is set, convert for float.
        if self.fp32_residual_connection:
            embeddings = embeddings.float()
        return embeddings


class RotaryEmbedding(nn.Cell):
    def __init__(self, dim, original_impl=False, dtype=None):
        super().__init__()
        self.inv_freq = ms.Parameter(
            1.0 / (10000 ** ms.tensor((np.arange(0, dim, 2)), dtype=dtype) / dim), requires_grad=False, name="inv_freq"
        )
        self.dim = dim
        self.original_impl = original_impl

    def forward_impl(self, seq_len: int, n_elem: int, dtype: ms.dtype, base: int = 10000):
        """Enhanced Transformer with Rotary Position Embedding.

        Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
        transformers/rope/__init__.py. MIT License:
        https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
        """
        # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        theta = 1.0 / (base ** (ops.arange(0, n_elem, 2, dtype=ms.float32) / n_elem))

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = ops.arange(seq_len, dtype=ms.float32)

        # Calculate the product of position index and $\theta_i$
        idx_theta = ops.outer(seq_idx, theta).float()

        cache = ops.stack([ops.cos(idx_theta), ops.sin(idx_theta)], axis=-1)

        # this is to mimic the behaviour of complex32, else we will get different results
        if dtype in (ms.float16, ms.bfloat16, ms.int8):
            cache = ms.bfloat16() if dtype == ms.bfloat16 else cache.half()
        return cache

    def construct(self, max_seq_len, offset=0):
        return self.forward_impl(max_seq_len, self.dim, dtype=self.inv_freq.dtype)


class PrefixEncoder(nn.Cell):
    """
    The nn model to encode the prefix Input shape: (batch-size, prefix-length) Output shape: (batch-size,
    prefix-length, 2*layers*hidden)
    """

    def __init__(self, config: ChatGLMConfig):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            kv_size = config.num_layers * config.kv_channels * config.multi_query_group_num * 2
            self.embedding = nn.Embedding(config.pre_seq_len, kv_size)
            self.trans = nn.SequentialCell(
                nn.Dense(kv_size, config.hidden_size),
                nn.Tanh(),
                nn.Dense(config.hidden_size, kv_size),
            )
        else:
            self.embedding = nn.Embedding(
                config.pre_seq_len, config.num_layers * config.kv_channels * config.multi_query_group_num * 2
            )

    def construct(self, prefix: ms.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values


class ChatGLMModel(ChatGLMPreTrainedModel):
    def __init__(self, config: ChatGLMConfig, empty_init=True):
        super().__init__(config)
        init_method = default_init
        init_kwargs = {}
        self.embedding = init_method(Embedding, config, **init_kwargs)
        self.num_layers = config.num_layers
        self.multi_query_group_num = config.multi_query_group_num
        self.kv_channels = config.kv_channels
        self.output_hidden_states = config.output_hidden_states
        self.use_cache = config.use_cache
        self.use_return_dict = config.use_return_dict

        # Rotary positional embeddings
        self.seq_length = config.seq_length
        rotary_dim = (
            config.hidden_size // config.num_attention_heads if config.kv_channels is None else config.kv_channels
        )

        self.rotary_pos_emb = RotaryEmbedding(
            rotary_dim // 2,
            original_impl=config.original_rope,
        )
        self.encoder = init_method(GLMTransformer, config, **init_kwargs)
        self.output_layer = init_method(
            nn.Dense,
            config.hidden_size,
            config.padded_vocab_size,
            has_bias=False,
            **init_kwargs,
        )
        self.pre_seq_len = config.pre_seq_len
        self.prefix_projection = config.prefix_projection
        if self.pre_seq_len is not None:
            for _, param in self.parameters_and_names():
                param.requires_grad = False
            self.prefix_tokens = ops.arange(self.pre_seq_len).long()
            self.prefix_encoder = PrefixEncoder(config)
            self.dropout = nn.Dropout(p=0.1)
        self.post_init()

    def get_input_embeddings(self):
        return self.embedding.word_embeddings

    def get_prompt(self, batch_size, dtype=ms.float16):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).broadcast_to((batch_size, -1))
        past_key_values = self.prefix_encoder(prefix_tokens).type(dtype)
        past_key_values = past_key_values.view(
            batch_size, self.pre_seq_len, self.num_layers * 2, self.multi_query_group_num, self.kv_channels
        )
        # seq_len, b, nh, hidden_size
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 1, 0, 3, 4]).split(2)
        return past_key_values

    def construct(
        self,
        input_ids,
        position_ids: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        full_attention_mask: Optional[ms.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[ms.Tensor, ms.Tensor], ...]] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = False,
        attention_mask_all: Optional[bool] = True,
    ):
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.use_cache
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        batch_size, seq_length = input_ids.shape

        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)

        if self.pre_seq_len is not None:
            if past_key_values is None:
                past_key_values = self.get_prompt(batch_size=batch_size, dtype=inputs_embeds.dtype)
            if attention_mask is not None:
                attention_mask = ops.cat(
                    [attention_mask.new_ones((batch_size, self.pre_seq_len)), attention_mask], axis=-1
                )

        if full_attention_mask is None:
            if (attention_mask is not None and not attention_mask_all) or (past_key_values and seq_length != 1):
                full_attention_mask = self.get_masks(input_ids, past_key_values, padding_mask=attention_mask)

        # Rotary positional embeddings
        rotary_pos_emb = self.rotary_pos_emb(self.seq_length)
        if position_ids is not None:
            rotary_pos_emb = rotary_pos_emb[position_ids]
        else:
            rotary_pos_emb = rotary_pos_emb[None, :seq_length]
        rotary_pos_emb = rotary_pos_emb.swapaxes(0, 1)

        # Run encoder.
        hidden_states, presents, all_hidden_states, all_self_attentions = self.encoder(
            inputs_embeds,
            full_attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            kv_caches=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
        )

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
