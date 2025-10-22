# coding=utf-8
# Copyright 2024 The Seamless Authors and the HuggingFace Inc. team. All rights reserved.
#
# This code is adapted from https://github.com/huggingface/transformers
# with modifications to run transformers on mindspore.
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
"""Mindspore Wav2Vec2-BERT model."""

import math
from typing import Optional, Tuple, Union

import numpy as np
from transformers.models.wav2vec2_bert.configuration_wav2vec2_bert import Wav2Vec2BertConfig

import mindspore
from mindspore import nn
from mindspore.common.initializer import HeNormal, Uniform, initializer
from mindspore.mint.nn import CrossEntropyLoss

from ....models.utils import xavier_uniform_
from ...activations import ACT2FN
from ...mindspore_adapter import dtype_to_min
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import (
    BaseModelOutput,
    CausalLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    Wav2Vec2BaseModelOutput,
    XVectorOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import logging

logger = logging.get_logger(__name__)


_HIDDEN_STATES_START_POSITION = 2

# General docstring
_CONFIG_FOR_DOC = "Wav2Vec2BertConfig"

# Base docstring
_BASE_CHECKPOINT_FOR_DOC = "facebook/w2v-bert-2.0"
_PRETRAINED_CHECKPOINT_FOR_DOC = "hf-audio/wav2vec2-bert-CV16-en"
_EXPECTED_OUTPUT_SHAPE = [1, 146, 1024]

# CTC docstring
_CTC_EXPECTED_OUTPUT = "'mr quilter is the apostle of the middle classes and we are glad to welcome his gospel'"
_CTC_EXPECTED_LOSS = 17.04


# Copied from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2._compute_new_attention_mask
def _compute_new_attention_mask(hidden_states: mindspore.Tensor, seq_lens: mindspore.Tensor):
    """
    Computes an attention mask of the form `(batch, seq_len)` with an attention for each element in the batch that
    stops at the corresponding element in `seq_lens`.
    Args:
        hidden_states (`mindspore.tensor` of shape `(batch, seq_len, *)`):
            The sequences to mask, where `*` is any number of sequence-specific dimensions including none.
        seq_lens (`mindspore.tensor` of shape `(batch)`:
            Each element represents the length of the sequence at the same index in `hidden_states`
    Returns:
        `mindspore.tensor`: The float attention mask of shape `(batch, seq_len)`
    """
    batch_size, mask_seq_len = hidden_states.shape[:2]

    indices = mindspore.mint.arange(
        mask_seq_len,
    ).broadcast_to((batch_size, -1))

    bool_mask = indices >= seq_lens.unsqueeze(1).broadcast_to((-1, mask_seq_len))

    mask = hidden_states.new_ones((batch_size, mask_seq_len))

    mask = mask.masked_fill(bool_mask, 0)

    return mask


# Copied from transformers.models.wav2vec2.modeling_wav2vec2._compute_mask_indices
def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    attention_mask: Optional[mindspore.Tensor] = None,
    min_masks: int = 0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape. Used to implement [SpecAugment: A Simple Data Augmentation Method for
    ASR](https://arxiv.org/abs/1904.08779). Note that this method is not optimized to run on TPU and should be run on
    CPU as part of the preprocessing during training.

    Args:
        shape: The shape for which to compute masks. This should be of a tuple of size 2 where
               the first element is the batch size and the second element is the length of the axis to span.
        mask_prob:  The percentage of the whole axis (between 0 and 1) which will be masked. The number of
                    independently generated mask spans of length `mask_length` is computed by
                    `mask_prob*shape[1]/mask_length`. Note that due to overlaps, `mask_prob` is an upper bound and the
                    actual percentage will be smaller.
        mask_length: size of the mask
        min_masks: minimum number of masked spans
        attention_mask: A (right-padded) attention mask which independently shortens the feature axis of
                        each batch dimension.
    """
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
            f" and `sequence_length`: {sequence_length}`"
        )

    # epsilon is used for probabilistic rounding
    epsilon = np.random.rand(1).item()

    def compute_num_masked_span(input_length):
        """Given input length, compute how many spans should be masked"""
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
        num_masked_span = max(num_masked_span, min_masks)

        # make sure num masked span <= sequence_length
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length

        # make sure num_masked span is also <= input_length - (mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)

        return num_masked_span

    # compute number of masked spans in batch
    input_lengths = (
        attention_mask.sum(-1).detach().tolist()
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]
    )

    # SpecAugment mask to fill
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)
    spec_aug_mask_idxs = []

    max_num_masked_span = compute_num_masked_span(sequence_length)

    if max_num_masked_span == 0:
        return spec_aug_mask

    for input_length in input_lengths:
        # compute num of masked spans for this input
        num_masked_span = compute_num_masked_span(input_length)

        # get random indices to mask
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )

        # pick first sampled index that will serve as a dummy index to pad vector
        # to ensure same dimension for all batches due to probabilistic rounding
        # Picking first sample just pads those vectors twice.
        if len(spec_aug_mask_idx) == 0:
            # this case can only happen if `input_length` is strictly smaller then
            # `sequence_length` in which case the last token has to be a padding
            # token which we can use as a dummy mask id
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]

        spec_aug_mask_idx = np.concatenate(
            [spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    # expand masked indices to masked spans
    spec_aug_mask_idxs = np.expand(spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length))
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)

    # add offset to the starting indexes so that indexes now create a span
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.expand(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # ensure that we cannot have indices larger than sequence_length
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1

    # scatter indices to mask
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    return spec_aug_mask


# Copied from transformers.models.wav2vec2.modeling_wav2vec2._sample_negative_indices
def _sample_negative_indices(features_shape: Tuple, num_negatives: int, mask_time_indices: Optional[np.ndarray] = None):
    """
    Sample `num_negatives` vectors from feature vectors.
    """
    batch_size, sequence_length = features_shape

    # generate indices of the positive vectors themselves, repeat them `num_negatives` times
    sequence_length_range = np.arange(sequence_length)

    # get `num_negatives` random vector indices from the same utterance
    sampled_negative_indices = np.zeros(shape=(batch_size, sequence_length, num_negatives), dtype=np.int32)

    mask_time_indices = (
        mask_time_indices.astype(bool) if mask_time_indices is not None else np.ones(features_shape, dtype=bool)
    )

    for batch_idx in range(batch_size):
        high = mask_time_indices[batch_idx].sum() - 1
        mapped_masked_indices = sequence_length_range[mask_time_indices[batch_idx]]

        feature_indices = np.expand(np.arange(high + 1)[:, None], (high + 1, num_negatives))
        sampled_indices = np.random.randint(0, high, size=(high + 1, num_negatives))
        # avoid sampling the same positive vector, but keep the distribution uniform
        sampled_indices[sampled_indices >= feature_indices] += 1

        # remap to actual indices
        sampled_negative_indices[batch_idx][mask_time_indices[batch_idx]] = mapped_masked_indices[sampled_indices]

        # correct for batch size
        sampled_negative_indices[batch_idx] += batch_idx * sequence_length

    return sampled_negative_indices


# Copied from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerRotaryPositionalEmbedding with Wav2Vec2Conformer->Wav2Vec2Bert
class Wav2Vec2BertRotaryPositionalEmbedding(mindspore.nn.Cell):
    """Rotary positional embedding
    Reference : https://blog.eleuther.ai/rotary-embeddings/ Paper: https://arxiv.org/pdf/2104.09864.pdf
    """

    def __init__(self, config):
        super().__init__()
        dim = config.hidden_size // config.num_attention_heads
        base = config.rotary_embedding_base

        inv_freq = 1.0 / (base ** (mindspore.mint.arange(0, dim, 2, dtype=mindspore.int64).float() / dim))
        # Ignore copy
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.cached_sequence_length = None
        self.cached_rotary_positional_embedding = None

    def construct(self, hidden_states):
        sequence_length = hidden_states.shape[1]

        if sequence_length == self.cached_sequence_length and self.cached_rotary_positional_embedding is not None:
            return self.cached_rotary_positional_embedding

        self.cached_sequence_length = sequence_length
        # Embeddings are computed in the dtype of the inv_freq constant
        time_stamps = mindspore.mint.arange(sequence_length).type_as(self.inv_freq)
        freqs = mindspore.mint.einsum("i,j->ij", time_stamps, self.inv_freq)
        embeddings = mindspore.mint.cat((freqs, freqs), dim=-1)

        cos_embeddings = embeddings.cos()[:, None, None, :]
        sin_embeddings = embeddings.sin()[:, None, None, :]
        # Computed embeddings are cast to the dtype of the hidden state inputs
        self.cached_rotary_positional_embedding = mindspore.mint.stack([cos_embeddings, sin_embeddings]).type_as(
            hidden_states
        )
        return self.cached_rotary_positional_embedding


# Copied from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerRelPositionalEmbedding with Wav2Vec2Conformer->Wav2Vec2Bert
class Wav2Vec2BertRelPositionalEmbedding(mindspore.nn.Cell):
    """Relative positional encoding module."""

    def __init__(self, config):
        super().__init__()
        self.max_len = config.max_source_positions
        self.d_model = config.hidden_size
        self.pe = None
        self.extend_pe(mindspore.Tensor(0.0).broadcast_to((1, self.max_len)))

    def extend_pe(self, x):
        # Reset the positional encodings
        if self.pe is not None:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if self.pe.shape[1] >= x.shape[1] * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(
                        dtype=x.dtype,
                    )
                return
        # Suppose `i` is the position of query vector and `j` is the
        # position of key vector. We use positive relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = mindspore.mint.zeros((x.shape[1], self.d_model))
        pe_negative = mindspore.mint.zeros((x.shape[1], self.d_model))
        position = mindspore.mint.arange(0, x.shape[1], dtype=mindspore.int64).float().unsqueeze(1)
        div_term = mindspore.mint.exp(
            mindspore.mint.arange(0, self.d_model, 2, dtype=mindspore.int64).float()
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = mindspore.mint.sin(position * div_term)
        pe_positive[:, 1::2] = mindspore.mint.cos(position * div_term)
        pe_negative[:, 0::2] = mindspore.mint.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = mindspore.mint.cos(-1 * position * div_term)

        # Reverse the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in https://arxiv.org/abs/1901.02860
        pe_positive = mindspore.mint.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = mindspore.mint.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(dtype=x.dtype)

    def construct(self, hidden_states: mindspore.Tensor):
        self.extend_pe(hidden_states)
        start_idx = self.pe.shape[1] // 2 - hidden_states.shape[1] + 1
        end_idx = self.pe.shape[1] // 2 + hidden_states.shape[1]
        relative_position_embeddings = self.pe[:, start_idx:end_idx]

        return relative_position_embeddings


class Wav2Vec2BertFeatureProjection(mindspore.nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = mindspore.mint.nn.LayerNorm(config.feature_projection_input_dim, eps=config.layer_norm_eps)
        self.projection = mindspore.mint.nn.Linear(config.feature_projection_input_dim, config.hidden_size)
        self.dropout = mindspore.mint.nn.Dropout(config.feat_proj_dropout)

    def construct(self, hidden_states):
        # non-projected hidden states are needed for quantization
        norm_hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states, norm_hidden_states


class Wav2Vec2BertFeedForward(mindspore.nn.Cell):
    def __init__(self, config, act_fn=None, hidden_size=None):
        super().__init__()
        act_fn = act_fn if act_fn is not None else config.hidden_act
        hidden_size = hidden_size if hidden_size is not None else config.hidden_size
        self.intermediate_dropout = mindspore.mint.nn.Dropout(config.activation_dropout)

        self.intermediate_dense = mindspore.mint.nn.Linear(hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[act_fn] if isinstance(act_fn, str) else act_fn

        self.output_dense = mindspore.mint.nn.Linear(config.intermediate_size, hidden_size)
        self.output_dropout = mindspore.mint.nn.Dropout(config.hidden_dropout)

    # Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeedForward.forward
    def construct(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)

        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


class Wav2Vec2BertConvolutionModule(mindspore.nn.Cell):
    """Convolution block used in the conformer block"""

    def __init__(self, config):
        super().__init__()
        if (config.conv_depthwise_kernel_size - 1) % 2 == 1:
            raise ValueError("`config.conv_depthwise_kernel_size` should be a odd number for 'SAME' padding")
        self.layer_norm = mindspore.mint.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pointwise_conv1 = nn.Conv1d(
            config.hidden_size,
            2 * config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            has_bias=False,
            pad_mode="pad",
        )
        self.glu = mindspore.mint.nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            config.conv_depthwise_kernel_size,
            stride=1,
            padding=0,
            group=config.hidden_size,
            has_bias=False,
            pad_mode="pad",
        )

        self.depthwise_layer_norm = mindspore.mint.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.activation = ACT2FN[config.hidden_act]
        self.pointwise_conv2 = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            has_bias=False,
            pad_mode="pad",
        )
        self.dropout = mindspore.mint.nn.Dropout(config.conformer_conv_dropout)

    def construct(self, hidden_states, attention_mask=None):
        hidden_states = self.layer_norm(hidden_states)

        # Ensure that we do not leak padded positions in depthwise convolution if attention mask is passed.
        # Put 0 where necessary
        if attention_mask is not None:
            hidden_states = hidden_states.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)

        # exchange the temporal dimension and the feature dimension
        hidden_states = hidden_states.transpose(1, 2)

        # GLU mechanism
        # => (batch, 2*channel, dim)
        hidden_states = self.pointwise_conv1(hidden_states)
        # => (batch, channel, dim)
        hidden_states = self.glu(hidden_states)

        # Pad the sequence entirely on the left because of causal convolution.
        hidden_states = mindspore.mint.nn.functional.pad(hidden_states, (self.depthwise_conv.kernel_size[1] - 1, 0))

        # 1D Depthwise Conv
        hidden_states = self.depthwise_conv(hidden_states)

        hidden_states = self.depthwise_layer_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        hidden_states = self.activation(hidden_states)

        hidden_states = self.pointwise_conv2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


class Wav2Vec2BertSelfAttention(mindspore.nn.Cell):
    """Construct an Wav2Vec2BertSelfAttention object.
    Can be enhanced with rotary or relative position embeddings.
    """

    def __init__(self, config, is_adapter_attention=False):
        super().__init__()
        hidden_size = config.hidden_size if not is_adapter_attention else config.output_hidden_size

        self.head_size = hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.position_embeddings_type = config.position_embeddings_type if not is_adapter_attention else None

        self.linear_q = mindspore.mint.nn.Linear(hidden_size, hidden_size)
        self.linear_k = mindspore.mint.nn.Linear(hidden_size, hidden_size)
        self.linear_v = mindspore.mint.nn.Linear(hidden_size, hidden_size)
        self.linear_out = mindspore.mint.nn.Linear(hidden_size, hidden_size)

        self.dropout = mindspore.mint.nn.Dropout(p=config.attention_dropout)

        if self.position_embeddings_type == "relative":
            # linear transformation for positional encoding
            self.linear_pos = mindspore.mint.nn.Linear(hidden_size, hidden_size, bias=False)
            # these two learnable bias are used in matrix c and matrix d
            # as described in https://arxiv.org/abs/1901.02860 Section 3.3
            self.pos_bias_u = mindspore.Parameter(mindspore.mint.zeros((self.num_heads, self.head_size)))
            self.pos_bias_v = mindspore.Parameter(mindspore.mint.zeros((self.num_heads, self.head_size)))

        if self.position_embeddings_type == "relative_key":
            self.left_max_position_embeddings = config.left_max_position_embeddings
            self.right_max_position_embeddings = config.right_max_position_embeddings
            num_positions = self.left_max_position_embeddings + self.right_max_position_embeddings + 1
            self.distance_embedding = mindspore.mint.nn.Embedding(num_positions, self.head_size)

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        relative_position_embeddings: Optional[mindspore.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]:
        # self-attention mechanism
        batch_size, sequence_length, hidden_size = hidden_states.shape

        # make sure query/key states can be != value states
        query_key_states = hidden_states
        value_states = hidden_states

        if self.position_embeddings_type == "rotary":
            if relative_position_embeddings is None:
                raise ValueError(
                    "`relative_position_embeddings` has to be defined when `self.position_embeddings_type == 'rotary'"
                )
            query_key_states = self._apply_rotary_embedding(query_key_states, relative_position_embeddings)

        # project query_key_states and value_states
        query = self.linear_q(query_key_states).view(batch_size, -1, self.num_heads, self.head_size)
        key = self.linear_k(query_key_states).view(batch_size, -1, self.num_heads, self.head_size)
        value = self.linear_v(value_states).view(batch_size, -1, self.num_heads, self.head_size)

        # => (batch, head, time1, d_k)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        if self.position_embeddings_type == "relative":
            if relative_position_embeddings is None:
                raise ValueError(
                    "`relative_position_embeddings` has to be defined when `self.position_embeddings_type =="
                    " 'relative'"
                )
            # apply relative_position_embeddings to qk scores
            # as proposed in Transformer_XL: https://arxiv.org/abs/1901.02860
            scores = self._apply_relative_embeddings(
                query=query, key=key, relative_position_embeddings=relative_position_embeddings
            )
        else:
            scores = mindspore.mint.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_size)

        if self.position_embeddings_type == "relative_key":
            query_length, key_length = query.shape[2], key.shape[2]

            position_ids_l = mindspore.mint.arange(
                query_length,
                dtype=mindspore.int64,
            ).view(-1, 1)
            position_ids_r = mindspore.mint.arange(
                key_length,
                dtype=mindspore.int64,
            ).view(1, -1)
            distance = position_ids_r - position_ids_l
            distance = mindspore.mint.clamp(
                distance, -self.left_max_position_embeddings, self.right_max_position_embeddings
            )

            positional_embedding = self.distance_embedding(distance + self.left_max_position_embeddings)
            positional_embedding = positional_embedding.to(dtype=query.dtype)  # fp16 compatibility

            relative_position_attn_weights = mindspore.mint.einsum("bhld,lrd->bhlr", query, positional_embedding)
            scores = scores + (relative_position_attn_weights / math.sqrt(self.head_size))

        # apply attention_mask if necessary
        if attention_mask is not None:
            scores = scores + attention_mask

        # => (batch, head, time1, time2)
        probs = mindspore.mint.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        # => (batch, head, time1, d_k)
        hidden_states = mindspore.mint.matmul(probs, value)

        # => (batch, time1, hidden_size)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_size)
        hidden_states = self.linear_out(hidden_states)

        return hidden_states, probs

    # Copied from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerSelfAttention._apply_rotary_embedding
    def _apply_rotary_embedding(self, hidden_states, relative_position_embeddings):
        batch_size, sequence_length, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, sequence_length, self.num_heads, self.head_size)

        cos = relative_position_embeddings[0, :sequence_length, ...]
        sin = relative_position_embeddings[1, :sequence_length, ...]

        # rotate hidden_states with rotary embeddings
        hidden_states = hidden_states.transpose(0, 1)
        rotated_states_begin = hidden_states[..., : self.head_size // 2]
        rotated_states_end = hidden_states[..., self.head_size // 2 :]
        rotated_states = mindspore.mint.cat(
            (-rotated_states_end, rotated_states_begin), dim=rotated_states_begin.ndim - 1
        )
        hidden_states = (hidden_states * cos) + (rotated_states * sin)
        hidden_states = hidden_states.transpose(0, 1)

        hidden_states = hidden_states.view(batch_size, sequence_length, self.num_heads * self.head_size)

        return hidden_states

    # Copied from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerSelfAttention._apply_relative_embeddings
    def _apply_relative_embeddings(self, query, key, relative_position_embeddings):
        # 1. project positional embeddings
        # => (batch, head, 2*time1-1, d_k)
        proj_relative_position_embeddings = self.linear_pos(relative_position_embeddings)
        proj_relative_position_embeddings = proj_relative_position_embeddings.view(
            relative_position_embeddings.shape[0], -1, self.num_heads, self.head_size
        )
        proj_relative_position_embeddings = proj_relative_position_embeddings.transpose(1, 2)
        proj_relative_position_embeddings = proj_relative_position_embeddings.transpose(2, 3)

        # 2. Add bias to query
        # => (batch, head, time1, d_k)
        query = query.transpose(1, 2)
        q_with_bias_u = (query + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (query + self.pos_bias_v).transpose(1, 2)

        # 3. attention score: first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # => (batch, head, time1, time2)
        scores_ac = mindspore.mint.matmul(q_with_bias_u, key.transpose(-2, -1))

        # 4. then compute matrix b and matrix d
        # => (batch, head, time1, 2*time1-1)
        scores_bd = mindspore.mint.matmul(q_with_bias_v, proj_relative_position_embeddings)

        # 5. shift matrix b and matrix d
        zero_pad = mindspore.mint.zeros((*scores_bd.shape[:3], 1), dtype=scores_bd.dtype)
        scores_bd_padded = mindspore.mint.cat([zero_pad, scores_bd], dim=-1)
        scores_bd_padded_shape = scores_bd.shape[:2] + (scores_bd.shape[3] + 1, scores_bd.shape[2])
        scores_bd_padded = scores_bd_padded.view(*scores_bd_padded_shape)
        scores_bd = scores_bd_padded[:, :, 1:].view_as(scores_bd)
        scores_bd = scores_bd[:, :, :, : scores_bd.shape[-1] // 2 + 1]

        # 6. sum matrices
        # => (batch, head, time1, time2)
        scores = (scores_ac + scores_bd) / math.sqrt(self.head_size)

        return scores


class Wav2Vec2BertEncoderLayer(mindspore.nn.Cell):
    """Conformer block based on https://arxiv.org/abs/2005.08100."""

    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        dropout = config.attention_dropout

        # Feed-forward 1
        self.ffn1_layer_norm = mindspore.mint.nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.ffn1 = Wav2Vec2BertFeedForward(config)

        # Self-Attention
        self.self_attn_layer_norm = mindspore.mint.nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.self_attn_dropout = mindspore.mint.nn.Dropout(dropout)
        self.self_attn = Wav2Vec2BertSelfAttention(config)

        # Conformer Convolution
        self.conv_module = Wav2Vec2BertConvolutionModule(config)

        # Feed-forward 2
        self.ffn2_layer_norm = mindspore.mint.nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.ffn2 = Wav2Vec2BertFeedForward(config)
        self.final_layer_norm = mindspore.mint.nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def construct(
        self,
        hidden_states,
        attention_mask: Optional[mindspore.Tensor] = None,
        relative_position_embeddings: Optional[mindspore.Tensor] = None,
        output_attentions: bool = False,
        conv_attention_mask: Optional[mindspore.Tensor] = None,
    ):
        # 1. Feed-Forward 1 layer
        residual = hidden_states
        hidden_states = self.ffn1_layer_norm(hidden_states)
        hidden_states = self.ffn1(hidden_states)
        hidden_states = hidden_states * 0.5 + residual
        residual = hidden_states

        # 2. Self-Attention layer
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weigts = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            relative_position_embeddings=relative_position_embeddings,
            output_attentions=output_attentions,
        )
        hidden_states = self.self_attn_dropout(hidden_states)
        hidden_states = hidden_states + residual

        # 3. Convolutional Layer
        residual = hidden_states
        hidden_states = self.conv_module(hidden_states, attention_mask=conv_attention_mask)
        hidden_states = residual + hidden_states

        # 4. Feed-Forward 2 Layer
        residual = hidden_states
        hidden_states = self.ffn2_layer_norm(hidden_states)
        hidden_states = self.ffn2(hidden_states)
        hidden_states = hidden_states * 0.5 + residual
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states, attn_weigts


class Wav2Vec2BertEncoder(mindspore.nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if config.position_embeddings_type == "relative":
            self.embed_positions = Wav2Vec2BertRelPositionalEmbedding(config)
        elif config.position_embeddings_type == "rotary":
            self.embed_positions = Wav2Vec2BertRotaryPositionalEmbedding(config)
        else:
            self.embed_positions = None

        self.dropout = mindspore.mint.nn.Dropout(config.hidden_dropout)
        self.layers = mindspore.nn.CellList([Wav2Vec2BertEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def construct(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        conv_attention_mask = attention_mask
        if attention_mask is not None:
            # make sure padded tokens output 0
            hidden_states = hidden_states.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)

            # extend attention_mask
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask * dtype_to_min(hidden_states.dtype)
            attention_mask = attention_mask.broadcast_to(
                (attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1])
            )

        hidden_states = self.dropout(hidden_states)

        if self.embed_positions is not None:
            relative_position_embeddings = self.embed_positions(hidden_states)
        else:
            relative_position_embeddings = None

        synced_gpus = False  # is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = mindspore.mint.rand([])

            skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False
            if not skip_the_layer or synced_gpus:
                # under fsdp or deepspeed zero3 all gpus must run in sync
                if self.gradient_checkpointing and self.training:
                    raise NotImplementedError
                else:
                    layer_outputs = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        relative_position_embeddings=relative_position_embeddings,
                        output_attentions=output_attentions,
                        conv_attention_mask=conv_attention_mask,
                    )
                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class Wav2Vec2BertAdapter(mindspore.nn.Cell):
    def __init__(self, config):
        super().__init__()
        # feature dim might need to be down-projected
        if config.output_hidden_size != config.hidden_size:
            self.proj = mindspore.mint.nn.Linear(config.hidden_size, config.output_hidden_size)
            self.proj_layer_norm = mindspore.mint.nn.LayerNorm(config.output_hidden_size, eps=config.layer_norm_eps)
        else:
            self.proj = self.proj_layer_norm = None
        self.layers = mindspore.nn.CellList(
            [Wav2Vec2BertAdapterLayer(config) for _ in range(config.num_adapter_layers)]
        )
        self.layerdrop = config.layerdrop

        self.kernel_size = config.adapter_kernel_size
        self.stride = config.adapter_stride

    def _compute_sub_sample_lengths_from_attention_mask(self, seq_lens):
        if seq_lens is None:
            return seq_lens
        pad = self.kernel_size // 2
        seq_lens = ((seq_lens + 2 * pad - self.kernel_size) / self.stride) + 1
        return seq_lens.floor()

    def construct(self, hidden_states, attention_mask=None):
        # down project hidden_states if necessary
        if self.proj is not None and self.proj_layer_norm is not None:
            hidden_states = self.proj(hidden_states)
            hidden_states = self.proj_layer_norm(hidden_states)

        sub_sampled_lengths = None
        if attention_mask is not None:
            sub_sampled_lengths = attention_mask.shape[1] - (1 - attention_mask.int()).sum(1)

        for layer in self.layers:
            layerdrop_prob = mindspore.mint.rand([])
            sub_sampled_lengths = self._compute_sub_sample_lengths_from_attention_mask(sub_sampled_lengths)
            if not self.training or (layerdrop_prob > self.layerdrop):
                hidden_states = layer(
                    hidden_states, attention_mask=attention_mask, sub_sampled_lengths=sub_sampled_lengths
                )

        return hidden_states


class Wav2Vec2BertAdapterLayer(mindspore.nn.Cell):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.output_hidden_size
        dropout = config.conformer_conv_dropout

        self.kernel_size = config.adapter_kernel_size
        self.stride = config.adapter_stride

        # 1. residual convolution
        self.residual_layer_norm = mindspore.mint.nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.residual_conv = nn.Conv1d(
            embed_dim,
            2 * embed_dim,
            self.kernel_size,
            stride=self.stride,
            padding=self.stride // 2,
            pad_mode="pad",
        )
        self.activation = mindspore.mint.nn.GLU(dim=1)

        # Self-Attention
        self.self_attn_layer_norm = mindspore.mint.nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.self_attn_conv = nn.Conv1d(
            embed_dim,
            2 * embed_dim,
            self.kernel_size,
            stride=self.stride,
            padding=self.stride // 2,
            pad_mode="pad",
        )
        self.self_attn = Wav2Vec2BertSelfAttention(config, is_adapter_attention=True)
        self.self_attn_dropout = mindspore.mint.nn.Dropout(dropout)

        # Feed-forward
        self.ffn_layer_norm = mindspore.mint.nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.ffn = Wav2Vec2BertFeedForward(config, act_fn=config.adapter_act, hidden_size=embed_dim)

    def construct(
        self,
        hidden_states,
        attention_mask: Optional[mindspore.Tensor] = None,
        output_attentions: bool = False,
        sub_sampled_lengths: Optional[mindspore.Tensor] = None,
    ):
        residual = self.residual_layer_norm(hidden_states)

        # Apply pooling to the residual to match the sequence length of the
        # multi-head attention output.
        # (batch, seq_len, feature_dim) -> (batch, feature_dim, seq_len)
        residual = residual.transpose(1, 2)
        residual = self.residual_conv(residual)
        residual = self.activation(residual)
        # (batch, feature_dim, seq_len) -> (batch, seq_len, feature_dim)
        residual = residual.transpose(1, 2)

        hidden_states = self.self_attn_layer_norm(hidden_states)
        # Apply pooling before feeding to the multihead-attention layer.
        # (batch, seq_len, feature_dim) -> (batch, feature_dim, seq_len)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.self_attn_conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        # (batch, feature_dim, seq_len) -> (batch, seq_len, feature_dim)
        hidden_states = hidden_states.transpose(1, 2)

        if attention_mask is not None:
            attention_mask = _compute_new_attention_mask(hidden_states=hidden_states, seq_lens=sub_sampled_lengths)
            attention_mask = _prepare_4d_attention_mask(
                attention_mask,
                hidden_states.dtype,
            )

        # The rest of the computation is identical to a vanilla Transformer
        # encoder layer.
        hidden_states, attn_weigths = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.self_attn_dropout(hidden_states)
        hidden_states = hidden_states + residual

        residual = hidden_states

        hidden_states = self.ffn_layer_norm(hidden_states)
        hidden_states = self.ffn(hidden_states) + residual

        return hidden_states


# Copied from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerPreTrainedModel with
# Wav2Vec2Conformer->Wav2Vec2Bert,wav2vec2_conformer->wav2vec2_bert, input_values->input_features
class Wav2Vec2BertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Wav2Vec2BertConfig
    base_model_prefix = "wav2vec2_bert"
    main_input_name = "input_features"
    supports_gradient_checkpointing = True

    # Ignore copy
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, Wav2Vec2BertSelfAttention):
            if hasattr(module, "pos_bias_u"):
                xavier_uniform_(module.pos_bias_u)
            if hasattr(module, "pos_bias_v"):
                xavier_uniform_(module.pos_bias_v)
        elif isinstance(module, Wav2Vec2BertFeatureProjection):
            k = math.sqrt(1 / module.projection.in_features)
            module.projection.weight.set_data(
                initializer(
                    Uniform(scale=k), shape=module.projection.weight.shape, dtype=module.projection.weight.dtype
                )
            )
            module.projection.bias.set_data(
                initializer(Uniform(scale=k), shape=module.projection.bias.shape, dtype=module.projection.bias.dtype)
            )
        elif isinstance(module, mindspore.mint.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (mindspore.mint.nn.LayerNorm, mindspore.mint.nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            module.weight.set_data(initializer(HeNormal(), shape=module.weight.shape, dtype=module.weight.dtype))
            if module.bias is not None:
                k = math.sqrt(module.group / (module.in_channels * module.kernel_size[0]))
                module.bias.set_data(initializer(Uniform(scale=k), shape=module.bias.shape, dtype=module.bias.dtype))

    # Ignore copy
    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[mindspore.Tensor, int], add_adapter: Optional[bool] = None
    ):
        """
        Computes the output length of the convolutional layers
        """

        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length, kernel_size, stride, padding):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return mindspore.mint.div(input_length + 2 * padding - kernel_size, stride, rounding_mode="floor") + 1

        if add_adapter:
            padding = self.config.adapter_kernel_size // 2
            for _ in range(self.config.num_adapter_layers):
                input_lengths = _conv_out_length(
                    input_lengths, self.config.adapter_kernel_size, self.config.adapter_stride, padding
                )

        return input_lengths

    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: mindspore.Tensor, add_adapter=None
    ):
        # Effectively attention_mask.sum(-1), but not inplace to be able to run
        # on inference mode.
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths, add_adapter=add_adapter)
        output_lengths = output_lengths.to(mindspore.int64)

        batch_size = attention_mask.shape[0]

        attention_mask = mindspore.mint.zeros(
            (batch_size, feature_vector_length),
            dtype=attention_mask.dtype,
        )
        # these two operations makes sure that all values before the output lengths idxs are attended to
        attention_mask[
            (
                mindspore.mint.arange(
                    attention_mask.shape[0],
                ),
                output_lengths - 1,
            )
        ] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask


class Wav2Vec2BertModel(Wav2Vec2BertPreTrainedModel):
    def __init__(self, config: Wav2Vec2BertConfig):
        super().__init__(config)
        self.config = config
        self.feature_projection = Wav2Vec2BertFeatureProjection(config)

        # model only needs masking vector if mask prob is > 0.0
        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed = mindspore.Parameter(mindspore.mint.empty(config.hidden_size).uniform_())

        self.encoder = Wav2Vec2BertEncoder(config)

        self.adapter = Wav2Vec2BertAdapter(config) if config.add_adapter else None

        self.intermediate_ffn = None
        if config.use_intermediate_ffn_before_adapter:
            self.intermediate_ffn = Wav2Vec2BertFeedForward(config, act_fn="relu")

        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Model._mask_hidden_states
    def _mask_hidden_states(
        self,
        hidden_states: mindspore.Tensor,
        mask_time_indices: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
    ):
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """

        # `config.apply_spec_augment` can set masking to False
        if not getattr(self.config, "apply_spec_augment", True):
            return hidden_states

        # generate indices & apply SpecAugment along time axis
        batch_size, sequence_length, hidden_size = hidden_states.shape

        if mask_time_indices is not None:
            # apply SpecAugment along time axis with given mask_time_indices
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        elif self.config.mask_time_prob > 0 and self.training:
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.config.mask_time_prob,
                mask_length=self.config.mask_time_length,
                attention_mask=attention_mask,
                min_masks=self.config.mask_time_min_masks,
            )
            mask_time_indices = mindspore.Tensor(mask_time_indices, dtype=mindspore.bool_)
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)

        if self.config.mask_feature_prob > 0 and self.training:
            # generate indices & apply SpecAugment along feature axis
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length,
                min_masks=self.config.mask_feature_min_masks,
            )
            mask_feature_indices = mindspore.Tensor(mask_feature_indices, dtype=mindspore.bool_)
            mask_feature_indices = mask_feature_indices[:, None].broadcast_to((-1, sequence_length, -1))
            hidden_states[mask_feature_indices] = 0

        return hidden_states

    def construct(
        self,
        input_features: Optional[mindspore.Tensor],
        attention_mask: Optional[mindspore.Tensor] = None,
        mask_time_indices: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Wav2Vec2BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states, extract_features = self.feature_projection(input_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.intermediate_ffn:
            expanded_hidden_states = self.intermediate_ffn(hidden_states)
            hidden_states = hidden_states + 0.5 * expanded_hidden_states

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states, attention_mask=attention_mask)

        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]

        return Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class Wav2Vec2BertForCTC(Wav2Vec2BertPreTrainedModel):
    # Copied from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerForCTC.__init__ with
    # Wav2Vec2Conformer->Wav2Vec2Bert,WAV2VEC2_CONFORMER->WAV2VEC2_BERT,wav2vec2_conformer->wav2vec2_bert
    def __init__(self, config, target_lang: Optional[str] = None):
        super().__init__(config)

        self.wav2vec2_bert = Wav2Vec2BertModel(config)
        self.dropout = mindspore.mint.nn.Dropout(config.final_dropout)

        self.target_lang = target_lang

        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `Wav2Vec2BertForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )
        output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )
        self.lm_head = mindspore.mint.nn.Linear(output_hidden_size, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_features: Optional[mindspore.Tensor],
        attention_mask: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[mindspore.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        r"""
        labels (`mindspore.tensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """
        if labels is not None and labels.max() >= self.config.vocab_size:
            raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.wav2vec2_bert(
            input_features,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask
                if attention_mask is not None
                else mindspore.mint.ones(input_features.shape[:2], dtype=mindspore.int64)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum([-1])).to(mindspore.int64)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = mindspore.mint.nn.functional.log_softmax(logits, dim=-1, dtype=mindspore.float32).transpose(
                0, 1
            )

            # mindspore ctc_loss doesn't support 1-dim
            if flattened_targets.ndim == 1:
                flattened_targets = flattened_targets.unsqueeze(0)

            loss = mindspore.ops.ctc_loss(
                log_probs,
                flattened_targets,
                input_lengths,
                target_lengths,
                blank=self.config.pad_token_id,
                reduction=self.config.ctc_loss_reduction,
                zero_infinity=self.config.ctc_zero_infinity,
            )

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )


class Wav2Vec2BertForSequenceClassification(Wav2Vec2BertPreTrainedModel):
    # Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForSequenceClassification.__init__ with Wav2Vec2->Wav2Vec2Bert,wav2vec2->wav2vec2_bert
    def __init__(self, config):
        super().__init__(config)

        if hasattr(config, "add_adapter") and config.add_adapter:
            raise ValueError(
                "Sequence classification does not support the use of Wav2Vec2Bert adapters (config.add_adapter=True)"
            )
        self.wav2vec2_bert = Wav2Vec2BertModel(config)
        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        if config.use_weighted_layer_sum:
            self.layer_weights = mindspore.Parameter(mindspore.mint.ones(num_layers) / num_layers)
        self.projector = mindspore.mint.nn.Linear(config.hidden_size, config.classifier_proj_size)
        self.classifier = mindspore.mint.nn.Linear(config.classifier_proj_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.wav2vec2_bert.parameters():
            param.requires_grad = False

    # Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForSequenceClassification.forward with
    # Wav2Vec2->Wav2Vec2Bert,wav2vec2->wav2vec2_bert,WAV_2_VEC_2->WAV2VEC2_BERT, input_values->input_features
    def construct(
        self,
        input_features: Optional[mindspore.Tensor],
        attention_mask: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[mindspore.Tensor] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`mindspore.tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        outputs = self.wav2vec2_bert(
            input_features,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = mindspore.mint.stack(hidden_states, dim=1)
            norm_weights = mindspore.mint.nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = outputs[0]

        hidden_states = self.projector(hidden_states)
        if attention_mask is None:
            pooled_output = hidden_states.mean(dim=1)
        else:
            padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
            expand_padding_mask = padding_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_padding_mask] = 0.0
            pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Wav2Vec2BertForAudioFrameClassification(Wav2Vec2BertPreTrainedModel):
    # Copied from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerForAudioFrameClassification.__init__
    # with Wav2Vec2Conformer->Wav2Vec2Bert,WAV2VEC2_CONFORMER->WAV2VEC2_BERT,wav2vec2_conformer->wav2vec2_bert
    def __init__(self, config):
        super().__init__(config)

        if hasattr(config, "add_adapter") and config.add_adapter:
            raise ValueError(
                "Audio frame classification does not support the use of Wav2Vec2Bert adapters (config.add_adapter=True)"
            )
        self.wav2vec2_bert = Wav2Vec2BertModel(config)
        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        if config.use_weighted_layer_sum:
            self.layer_weights = mindspore.Parameter(mindspore.mint.ones(num_layers) / num_layers)
        self.classifier = mindspore.mint.nn.Linear(config.hidden_size, config.num_labels)
        self.num_labels = config.num_labels

        self.init_weights()

    # Copied from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerForAudioFrameClassification.freeze_base_model
    # with wav2vec2_conformer->wav2vec2_bert
    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.wav2vec2_bert.parameters():
            param.requires_grad = False

    # Copied from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerForAudioFrameClassification.forward
    # with wav2vec2_conformer->wav2vec2_bert, input_values->input_features
    def construct(
        self,
        input_features: Optional[mindspore.Tensor],
        attention_mask: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`mindspore.tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        outputs = self.wav2vec2_bert(
            input_features,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = mindspore.mint.stack(hidden_states, dim=1)
            norm_weights = mindspore.mint.nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = outputs[0]

        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.num_labels), mindspore.mint.argmax(labels.view(-1, self.num_labels), axis=1)
            )

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.AMSoftmaxLoss
class AMSoftmaxLoss(mindspore.nn.Cell):
    def __init__(self, input_dim, num_labels, scale=30.0, margin=0.4):
        super(AMSoftmaxLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.num_labels = num_labels
        self.weight = mindspore.Parameter(mindspore.mint.randn(input_dim, num_labels), requires_grad=True)
        self.loss = mindspore.mint.nn.CrossEntropyLoss()

    def construct(self, hidden_states, labels):
        labels = labels.flatten()
        weight = mindspore.mint.nn.functional.normalize(self.weight, dim=0)
        hidden_states = mindspore.mint.nn.functional.normalize(hidden_states, dim=1)
        cos_theta = mindspore.mint.mm(hidden_states, weight)
        psi = cos_theta - self.margin

        onehot = mindspore.mint.nn.functional.one_hot(labels, self.num_labels)
        logits = self.scale * mindspore.mint.where(onehot.bool(), psi, cos_theta)
        loss = self.loss(logits, labels)

        return loss


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.TDNNLayer
class TDNNLayer(mindspore.nn.Cell):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.tdnn_dim[layer_id - 1] if layer_id > 0 else config.tdnn_dim[layer_id]
        self.out_conv_dim = config.tdnn_dim[layer_id]
        self.kernel_size = config.tdnn_kernel[layer_id]
        self.dilation = config.tdnn_dilation[layer_id]

        self.kernel = mindspore.mint.nn.Linear(self.in_conv_dim * self.kernel_size, self.out_conv_dim)
        self.activation = mindspore.mint.nn.ReLU()

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        # for backward compatibility, we keep nn.Linear but call F.conv1d for speed up
        hidden_states = hidden_states.transpose(1, 2)
        weight = self.kernel.weight.view(self.out_conv_dim, self.kernel_size, self.in_conv_dim).transpose(1, 2)
        hidden_states = mindspore.mint.nn.functional.conv1d(
            hidden_states, weight, self.kernel.bias, dilation=self.dilation
        )
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = self.activation(hidden_states)
        return hidden_states


class Wav2Vec2BertForXVector(Wav2Vec2BertPreTrainedModel):
    # Copied from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerForXVector.__init__
    # with Wav2Vec2Conformer->Wav2Vec2Bert,WAV2VEC2_CONFORMER->WAV2VEC2_BERT,wav2vec2_conformer->wav2vec2_bert
    def __init__(self, config):
        super().__init__(config)

        self.wav2vec2_bert = Wav2Vec2BertModel(config)
        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        if config.use_weighted_layer_sum:
            self.layer_weights = mindspore.Parameter(mindspore.mint.ones(num_layers) / num_layers)
        self.projector = mindspore.mint.nn.Linear(config.hidden_size, config.tdnn_dim[0])

        tdnn_layers = [TDNNLayer(config, i) for i in range(len(config.tdnn_dim))]
        self.tdnn = mindspore.nn.CellList(tdnn_layers)

        self.feature_extractor = mindspore.mint.nn.Linear(config.tdnn_dim[-1] * 2, config.xvector_output_dim)
        self.classifier = mindspore.mint.nn.Linear(config.xvector_output_dim, config.xvector_output_dim)

        self.objective = AMSoftmaxLoss(config.xvector_output_dim, config.num_labels)

        self.init_weights()

    # Copied from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerForXVector.freeze_base_model
    # with wav2vec2_conformer->wav2vec2_bert
    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.wav2vec2_bert.parameters():
            param.requires_grad = False

    # Copied from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerForXVector._get_tdnn_output_lengths
    def _get_tdnn_output_lengths(self, input_lengths: Union[mindspore.Tensor, int]):
        """
        Computes the output length of the TDNN layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1

        for kernel_size in self.config.tdnn_kernel:
            input_lengths = _conv_out_length(input_lengths, kernel_size, 1)

        return input_lengths

    # Copied from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerForXVector.forward with
    # wav2vec2_conformer->wav2vec2_bert, input_values->input_features
    def construct(
        self,
        input_features: Optional[mindspore.Tensor],
        attention_mask: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[mindspore.Tensor] = None,
    ) -> Union[Tuple, XVectorOutput]:
        r"""
        labels (`mindspore.tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        outputs = self.wav2vec2_bert(
            input_features,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = mindspore.mint.stack(hidden_states, dim=1)
            norm_weights = mindspore.mint.nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = outputs[0]

        hidden_states = self.projector(hidden_states)

        for tdnn_layer in self.tdnn:
            hidden_states = tdnn_layer(hidden_states)

        # Statistic Pooling
        if attention_mask is None:
            mean_features = hidden_states.mean(dim=1)
            std_features = hidden_states.std(dim=1)
        else:
            feat_extract_output_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(dim=1))
            tdnn_output_lengths = self._get_tdnn_output_lengths(feat_extract_output_lengths)
            mean_features = []
            std_features = []
            for i, length in enumerate(tdnn_output_lengths):
                mean_features.append(hidden_states[i, :length].mean(dim=0))
                std_features.append(hidden_states[i, :length].std(dim=0))
            mean_features = mindspore.mint.stack(mean_features)
            std_features = mindspore.mint.stack(std_features)
        statistic_pooling = mindspore.mint.cat([mean_features, std_features], dim=-1)

        output_embeddings = self.feature_extractor(statistic_pooling)
        logits = self.classifier(output_embeddings)

        loss = None
        if labels is not None:
            loss = self.objective(logits, labels)

        if not return_dict:
            output = (logits, output_embeddings) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return XVectorOutput(
            loss=loss,
            logits=logits,
            embeddings=output_embeddings,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "Wav2Vec2BertForAudioFrameClassification",
    "Wav2Vec2BertForCTC",
    "Wav2Vec2BertForSequenceClassification",
    "Wav2Vec2BertForXVector",
    "Wav2Vec2BertModel",
    "Wav2Vec2BertPreTrainedModel",
]
