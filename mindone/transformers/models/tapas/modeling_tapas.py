# coding=utf-8
# Copyright 2020 Google Research and The HuggingFace Inc. team.
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
"""MindSpore TAPAS model."""

import enum
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

from transformers.models.tapas.configuration_tapas import TapasConfig
from transformers.utils import ModelOutput

import mindspore as ms
from mindspore import mint, nn, ops
from mindspore.common.initializer import Initializer, Normal
from mindspore.mint.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...mindspore_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, MaskedLMOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import logging

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "TapasConfig"
_CHECKPOINT_FOR_DOC = "google/tapas-base"


EPSILON_ZERO_DIVISION = 1e-10
CLOSE_ENOUGH_TO_LOG_ZERO = -10000.0


def scatter_reduce(input, dim, index, src, reduce, *, include_self=True, out=None):
    """
    Reimplementation of ms.scatter_reduce without using the native operator.
    Supports: 'sum', 'mean', 'amax', 'amin'.
    """
    # Broadcast index to match src shape
    if index.shape != src.shape:
        index = index.expand_as(src)

    # Initialize output tensor
    output = input.clone() if include_self else mint.zeros_like(input)

    # Process different reduce operations
    if reduce == "sum":
        for i in range(src.shape[dim]):
            src_slice = src.select(dim, i)
            idx_slice = index.select(dim, i)
            output.scatter_add_(dim, idx_slice.unsqueeze(0), src_slice.unsqueeze(0))

    elif reduce == "mean":
        count = mint.ones_like(input) if include_self else mint.zeros_like(input)
        sum_tensor = input.clone() if include_self else mint.zeros_like(input)

        for i in range(src.shape[dim]):
            src_slice = src.select(dim, i)
            idx_slice = index.select(dim, i)
            sum_tensor.scatter_add_(dim, idx_slice.unsqueeze(0), src_slice.unsqueeze(0))
            count.scatter_add_(dim, idx_slice.unsqueeze(0), mint.ones_like(src_slice).unsqueeze(0))

        # Avoid division by zero
        count = count.masked_fill(count == 0, 1)
        output = sum_tensor / count

    elif reduce == "amax":
        if not include_self:
            output = mint.full_like(input, -ms.tensor(float("inf")))
            updated_mask = mint.zeros_like(input, dtype=ms.bool_)

        for i in range(src.shape[dim]):
            src_slice = src.select(dim, i)
            idx_slice = index.select(dim, i)
            temp = mint.full_like(output, -ms.tensor(float("inf")))
            temp = ops.tensor_scatter_elements(temp, indices=idx_slice.unsqueeze(0), updates=src_slice.unsqueeze(0))
            output = mint.max(output, temp)

            if not include_self:
                mask_update = mint.zeros_like(updated_mask)
                mask_update = ops.tensor_scatter_elements(
                    mask_update.float(),
                    indices=idx_slice.unsqueeze(0),
                    updates=mint.ones_like(src_slice, dtype=ms.float32).unsqueeze(0),
                ).bool()
                updated_mask |= mask_update

        if not include_self:
            output = mint.where(updated_mask, output, input)

    elif reduce == "amin":
        if not include_self:
            output = mint.full_like(input, ms.tensor(float("inf")))
            updated_mask = mint.zeros_like(input, dtype=ms.bool_)

        for i in range(src.shape[dim]):
            src_slice = src.select(dim, i)
            idx_slice = index.select(dim, i)
            temp = mint.full_like(output, ms.tensor(float("inf")))
            temp = ops.tensor_scatter_elements(temp, indices=idx_slice.unsqueeze(0), updates=src_slice.unsqueeze(0))
            output = mint.min(output, temp)

            if not include_self:
                mask_update = mint.zeros_like(updated_mask)
                mask_update = ops.tensor_scatter_elements(
                    mask_update.float(),
                    indices=idx_slice.unsqueeze(0),
                    updates=mint.ones_like(src_slice, dtype=ms.float32).unsqueeze(0),
                ).bool()
                updated_mask |= mask_update

        if not include_self:
            output = mint.where(updated_mask, output, input)

    else:
        raise NotImplementedError(f"reduce='{reduce}' not supported")

    # Handle out parameter
    if out is not None:
        out.copy_(output)
        return out
    return output


@dataclass
class TableQuestionAnsweringOutput(ModelOutput):
    """
    Output type of [`TapasForQuestionAnswering`].

    Args:
        loss (`ms.Tensor` of shape `(1,)`, *optional*, returned when `labels` (and possibly `answer`, `aggregation_labels`, `numeric_values` and `numeric_values_scale` are provided)):  # noqa: E501
            Total loss as the sum of the hierarchical cell selection log-likelihood loss and (optionally) the
            semi-supervised regression loss and (optionally) supervised loss for aggregations.
        logits (`ms.Tensor` of shape `(batch_size, sequence_length)`):
            Prediction scores of the cell selection head, for every token.
        logits_aggregation (`ms.Tensor`, *optional*, of shape `(batch_size, num_aggregation_labels)`):
            Prediction scores of the aggregation head, for every aggregation operator.
        hidden_states (`tuple(ms.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `ms.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(ms.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `ms.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    loss: Optional[ms.Tensor] = None
    logits: Optional[ms.Tensor] = None
    logits_aggregation: Optional[ms.Tensor] = None
    hidden_states: Optional[Tuple[ms.Tensor]] = None
    attentions: Optional[Tuple[ms.Tensor]] = None


class TapasEmbeddings(nn.Cell):
    """
    Construct the embeddings from word, position and token_type embeddings. Same as BertEmbeddings but with a number of
    additional token type embeddings to encode tabular structure.
    """

    def __init__(self, config):
        super().__init__()
        # we do not include config.disabled_features and config.disable_position_embeddings from the original implementation
        # word embeddings
        self.word_embeddings = mint.nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # position embeddings
        self.position_embeddings = mint.nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # token type embeddings
        for i, type_vocab_sizes in enumerate(config.type_vocab_sizes):
            name = f"token_type_embeddings_{i}"
            setattr(self, name, mint.nn.Embedding(type_vocab_sizes, config.hidden_size))

        self.number_of_token_type_embeddings = len(config.type_vocab_sizes)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = mint.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = mint.nn.Dropout(config.hidden_dropout_prob)

        self.config = config

    def construct(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            # create absolute position embeddings
            position_ids = mint.arange(seq_length, dtype=ms.int64)
            position_ids = position_ids.unsqueeze(0).broadcast_to((input_shape))
            # when self.config.reset_position_index_per_cell is set to True, create relative position embeddings
            if self.config.reset_position_index_per_cell:
                # shape (batch_size, seq_len)
                col_index = IndexMap(token_type_ids[:, :, 1], self.config.type_vocab_sizes[1], batch_dims=1)
                # shape (batch_size, seq_len)
                row_index = IndexMap(token_type_ids[:, :, 2], self.config.type_vocab_sizes[2], batch_dims=1)
                # shape (batch_size, seq_len)
                full_index = ProductIndexMap(col_index, row_index)
                # shape (max_rows * max_columns,). First absolute position for every cell
                first_position_per_segment = reduce_min(position_ids, full_index)[0]
                # ? shape (batch_size, seq_len). First absolute position of the cell for every token
                first_position = gather(first_position_per_segment, full_index)
                # shape (1, seq_len)
                position = mint.arange(seq_length, dtype=ms.int64).unsqueeze(0)
                position_ids = mint.min(ms.tensor(self.config.max_position_embeddings - 1), position - first_position)

        if token_type_ids is None:
            token_type_ids = mint.zeros((input_shape + self.number_of_token_type_embeddings), dtype=ms.int64)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        position_embeddings = self.position_embeddings(position_ids)

        embeddings = inputs_embeds + position_embeddings

        for i in range(self.number_of_token_type_embeddings):
            name = f"token_type_embeddings_{i}"
            embeddings += getattr(self, name)(token_type_ids[:, :, i])

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class TapasSelfAttention(nn.Cell):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = mint.nn.Linear(config.hidden_size, self.all_head_size)
        self.key = mint.nn.Linear(config.hidden_size, self.all_head_size)
        self.value = mint.nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = mint.nn.Dropout(config.attention_probs_dropout_prob)
        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x):
        new_x_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def construct(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = mint.cat([past_key_value[0], key_layer], dim=2)
            value_layer = mint.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = mint.matmul(query_layer, key_layer.swapaxes(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in TapasModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = mint.nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = mint.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput
class TapasSelfOutput(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.dense = mint.nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = mint.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = mint.nn.Dropout(config.hidden_dropout_prob)

    def construct(self, hidden_states: ms.Tensor, input_tensor: ms.Tensor) -> ms.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TapasAttention(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.self = TapasSelfAttention(config)
        self.output = TapasSelfOutput(config)
        self.pruned_heads = set()

    # Copied from transformers.models.bert.modeling_bert.BertAttention.prune_heads
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # Copied from transformers.models.bert.modeling_bert.BertAttention.forward
    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        encoder_attention_mask: Optional[ms.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[ms.Tensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[ms.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate
class TapasIntermediate(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.dense = mint.nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput
class TapasOutput(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.dense = mint.nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = mint.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = mint.nn.Dropout(config.hidden_dropout_prob)

    def construct(self, hidden_states: ms.Tensor, input_tensor: ms.Tensor) -> ms.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TapasLayer(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = TapasAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = TapasAttention(config)
        self.intermediate = TapasIntermediate(config)
        self.output = TapasOutput(config)

    # Copied from transformers.models.bert.modeling_bert.BertLayer.forward
    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        encoder_attention_mask: Optional[ms.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[ms.Tensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[ms.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    # Copied from transformers.models.bert.modeling_bert.BertLayer.feed_forward_chunk
    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class TapasEncoder(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.CellList([TapasLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def construct(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_values,
                output_attentions,
            )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


# Copied from transformers.models.bert.modeling_bert.BertPooler
class TapasPooler(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.dense = mint.nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = mint.nn.Tanh()

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


# Copied from transformers.models.bert.modeling_bert.BertPredictionHeadTransform with Bert->Tapas
class TapasPredictionHeadTransform(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.dense = mint.nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = mint.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertLMPredictionHead with Bert->Tapas
class TapasLMPredictionHead(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.transform = TapasPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = mint.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = ms.Parameter(mint.zeros(config.vocab_size), name="bias")

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def _tie_weights(self):
        self.decoder.bias = self.bias

    def construct(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOnlyMLMHead with Bert->Tapas
class TapasOnlyMLMHead(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.predictions = TapasLMPredictionHead(config)

    def construct(self, sequence_output: ms.Tensor) -> ms.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class TapasPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = TapasConfig
    base_model_prefix = "tapas"
    supports_gradient_checkpointing = True
    _supports_param_buffer_assignment = False

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights with Bert->Tapas
    def _init_weights(self, module):
        """Initialize the weights"""
        pass


TAPAS_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its models (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a MindSpore
    [ms.nn.Cell](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Cell.html) subclass.
    Use it as a regular MindSpore Cell and refer to the MindSpore documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`TapasConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

TAPAS_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`ms.Tensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary. Indices can be obtained using [`AutoTokenizer`]. See
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`ms.Tensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`ms.Tensor` of shape `({0}, 7)`, *optional*):
            Token indices that encode tabular structure. Indices can be obtained using [`AutoTokenizer`]. See this
            class for more info.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`ms.Tensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. If
            `reset_position_index_per_cell` of [`TapasConfig`] is set to `True`, relative position embeddings will be
            used. Selected in the range `[0, config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`ms.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`: - 1
            indicates the head is **not masked**, - 0 indicates the head is **masked**.
        inputs_embeds (`ms.Tensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class TapasModel(TapasPreTrainedModel):
    """
    This class is a small change compared to [`BertModel`], taking into account the additional token type ids.

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = TapasEmbeddings(config)
        self.encoder = TapasEncoder(config)

        self.pooler = TapasPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        encoder_attention_mask: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer
        >>> from mindone.transformers import TapasModel
        >>> import pandas as pd
        >>> import mindspore as ms

        >>> tokenizer = AutoTokenizer.from_pretrained("google/tapas-base", revision="refs/pr/1")
        >>> model = TapasModel.from_pretrained("google/tapas-base", revision="refs/pr/1")

        >>> data = {
        ...     "Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
        ...     "Age": ["56", "45", "59"],
        ...     "Number of movies": ["87", "53", "69"],
        ... }
        >>> table = pd.DataFrame.from_dict(data)
        >>> queries = ["How many movies has George Clooney played in?", "How old is Brad Pitt?"]

        >>> inputs = tokenizer(table=table, queries=queries, padding="max_length", return_tensors="np")
        >>> inputs = {k: ms.tensor(v) for k, v in inputs.items()}
        >>> outputs = model(**inputs)

        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = mint.ones(input_shape)
        if token_type_ids is None:
            token_type_ids = mint.zeros((*input_shape, len(self.config.type_vocab_sizes)), dtype=ms.int64)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: ms.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.shape
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = mint.ones(encoder_hidden_shape)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class TapasForMaskedLM(TapasPreTrainedModel):
    _tied_weights_keys = ["cls.predictions.decoder.weight", "cls.predictions.decoder.bias"]
    config_class = TapasConfig
    base_model_prefix = "tapas"

    def __init__(self, config):
        super().__init__(config)

        self.tapas = TapasModel(config, add_pooling_layer=False)
        self.cls = TapasOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
        self.cls.predictions.bias = new_embeddings.bias

    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        encoder_attention_mask: Optional[ms.Tensor] = None,
        labels: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, MaskedLMOutput]:
        r"""
        labels (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer
        >>> from mindone.transformers import TapasForMaskedLM
        >>> import pandas as pd
        >>> import mindspore as ms

        >>> tokenizer = AutoTokenizer.from_pretrained("google/tapas-base")
        >>> model = TapasForMaskedLM.from_pretrained("google/tapas-base")

        >>> data = {
        ...     "Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
        ...     "Age": ["56", "45", "59"],
        ...     "Number of movies": ["87", "53", "69"],
        ... }
        >>> table = pd.DataFrame.from_dict(data)

        >>> inputs = tokenizer(
        ...     table=table, queries="How many [MASK] has George [MASK] played in?", return_tensors="np"
        ... )
        >>> labels = tokenizer(
        ...     table=table, queries="How many movies has George Clooney played in?", return_tensors="np"
        ... )["input_ids"]
        >>> inputs = {k: ms.tensor(v) for k, v in inputs.items()}
        >>> labels = ms.tensor(labels)

        >>> outputs = model(**inputs, labels=labels)
        >>> logits = outputs.logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.tapas(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class TapasForQuestionAnswering(TapasPreTrainedModel):
    def __init__(self, config: TapasConfig):
        super().__init__(config)

        # base model
        self.tapas = TapasModel(config)

        # dropout (only used when training)
        self.dropout = mint.nn.Dropout(config.hidden_dropout_prob)

        # cell selection heads
        if config.init_cell_selection_weights_to_zero:
            # init_cell_selection_weights_to_zero: Whether the initial weights should be
            # set to 0. This ensures that all tokens have the same prior probability.
            self.output_weights = ms.Parameter(mint.zeros(config.hidden_size), name="output_weights")
            self.column_output_weights = ms.Parameter(mint.zeros(config.hidden_size), name="column_output_weights")
        else:
            self.output_weights = ms.Parameter(mint.empty(config.hidden_size), name="output_weights")
            self.output_weights.set_data(
                Initializer(Normal(sigma=config.initializer_range, mean=0.0), shape=self.output_weights.shape)
            )  # here, a truncated normal is used in the original implementation
            self.column_output_weights = ms.Parameter(mint.empty(config.hidden_size), name="column_output_weights")
            self.column_output_weights.set_data(
                Initializer(Normal(sigma=config.initializer_range, mean=0.0), shape=self.column_output_weights.shape)
            )  # here, a truncated normal is used in the original implementation
        self.output_bias = ms.Parameter(mint.zeros([]), name="output_bias")
        self.column_output_bias = ms.Parameter(mint.zeros([]), name="column_output_bias")

        # aggregation head
        if config.num_aggregation_labels > 0:
            self.aggregation_classifier = mint.nn.Linear(config.hidden_size, config.num_aggregation_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        table_mask: Optional[ms.Tensor] = None,
        labels: Optional[ms.Tensor] = None,
        aggregation_labels: Optional[ms.Tensor] = None,
        float_answer: Optional[ms.Tensor] = None,
        numeric_values: Optional[ms.Tensor] = None,
        numeric_values_scale: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TableQuestionAnsweringOutput]:
        r"""
        table_mask (`ms.Tensor` of shape `(batch_size, seq_length)`, *optional*):
            Mask for the table. Indicates which tokens belong to the table (1). Question tokens, table headers and
            padding are 0.
        labels (`ms.Tensor` of shape `(batch_size, seq_length)`, *optional*):
            Labels per token for computing the hierarchical cell selection loss. This encodes the positions of the
            answer appearing in the table. Can be obtained using [`AutoTokenizer`].

            - 1 for tokens that are **part of the answer**,
            - 0 for tokens that are **not part of the answer**.

        aggregation_labels (`ms.Tensor` of shape `(batch_size, )`, *optional*):
            Aggregation function index for every example in the batch for computing the aggregation loss. Indices
            should be in `[0, ..., config.num_aggregation_labels - 1]`. Only required in case of strong supervision for
            aggregation (WikiSQL-supervised).
        float_answer (`ms.Tensor` of shape `(batch_size, )`, *optional*):
            Float answer for every example in the batch. Set to *float('nan')* for cell selection questions. Only
            required in case of weak supervision (WTQ) to calculate the aggregate mask and regression loss.
        numeric_values (`ms.Tensor` of shape `(batch_size, seq_length)`, *optional*):
            Numeric values of every token, NaN for tokens which are not numeric values. Can be obtained using
            [`AutoTokenizer`]. Only required in case of weak supervision for aggregation (WTQ) to calculate the
            regression loss.
        numeric_values_scale (`ms.Tensor` of shape `(batch_size, seq_length)`, *optional*):
            Scale of the numeric values of every token. Can be obtained using [`AutoTokenizer`]. Only required in case
            of weak supervision for aggregation (WTQ) to calculate the regression loss.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer
        >>> from mindone.transformers import TapasForQuestionAnswering
        >>> import pandas as pd
        >>> import mindspore as ms

        >>> tokenizer = AutoTokenizer.from_pretrained("google/tapas-base-finetuned-wtq", revision="refs/pr/2")
        >>> model = TapasForQuestionAnswering.from_pretrained("google/tapas-base-finetuned-wtq", revision="refs/pr/2")

        >>> data = {
        ...     "Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
        ...     "Age": ["56", "45", "59"],
        ...     "Number of movies": ["87", "53", "69"],
        ... }
        >>> table = pd.DataFrame.from_dict(data)
        >>> queries = ["How many movies has George Clooney played in?", "How old is Brad Pitt?"]

        >>> inputs = tokenizer(table=table, queries=queries, padding="max_length", return_tensors="np")
        >>> inputs = {k: ms.tensor(v) for k, v in inputs.items()}
        >>> outputs = model(**inputs)

        >>> logits = outputs.logits
        >>> logits_aggregation = outputs.logits_aggregation
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.tapas(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        sequence_output = self.dropout(sequence_output)

        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        # Construct indices for the table.
        if token_type_ids is None:
            token_type_ids = mint.zeros((*input_shape, len(self.config.type_vocab_sizes)), dtype=ms.int64)

        token_types = [
            "segment_ids",
            "column_ids",
            "row_ids",
            "prev_labels",
            "column_ranks",
            "inv_column_ranks",
            "numeric_relations",
        ]

        row_ids = token_type_ids[:, :, token_types.index("row_ids")]
        column_ids = token_type_ids[:, :, token_types.index("column_ids")]

        row_index = IndexMap(
            indices=mint.min(row_ids, ms.tensor(self.config.max_num_rows - 1)),
            num_segments=self.config.max_num_rows,
            batch_dims=1,
        )
        col_index = IndexMap(
            indices=mint.min(column_ids, ms.tensor(self.config.max_num_columns - 1)),
            num_segments=self.config.max_num_columns,
            batch_dims=1,
        )
        cell_index = ProductIndexMap(row_index, col_index)

        # Masks.
        input_shape = input_ids.shape if input_ids is not None else inputs_embeds.shape[:-1]
        if attention_mask is None:
            attention_mask = mint.ones(input_shape)
        # Table cells only, without question tokens and table headers.
        if table_mask is None:
            table_mask = mint.where(row_ids > 0, mint.ones_like(row_ids), mint.zeros_like(row_ids))
        # ms.Tensor[batch_size, seq_length]
        input_mask_float = attention_mask.to(dtype=ms.float32)
        table_mask_float = table_mask.to(dtype=ms.float32)
        # Mask for cells that exist in the table (i.e. that are not padding).
        cell_mask, _ = reduce_mean(input_mask_float, cell_index)

        # Compute logits per token. These are used to select individual cells.
        logits = compute_token_logits(sequence_output, self.config.temperature, self.output_weights, self.output_bias)

        # Compute logits per column. These are used to select a column.
        column_logits = None
        if self.config.select_one_column:
            column_logits = compute_column_logits(
                sequence_output,
                self.column_output_weights,
                self.column_output_bias,
                cell_index,
                cell_mask,
                self.config.allow_empty_column_selection,
            )

        # Aggregation logits
        logits_aggregation = None
        if self.config.num_aggregation_labels > 0:
            logits_aggregation = self.aggregation_classifier(pooled_output)

        # Total loss calculation
        total_loss = 0.0
        calculate_loss = False
        if labels is not None:
            calculate_loss = True
            is_supervised = not self.config.num_aggregation_labels > 0 or not self.config.use_answer_as_supervision

            # Semi-supervised cell selection in case of no aggregation:
            # If the answer (the denotation) appears directly in the table we might
            # select the answer without applying any aggregation function. There are
            # some ambiguous cases, see utils._calculate_aggregate_mask for more info.
            # `aggregate_mask` is 1 for examples where we chose to aggregate and 0
            #  for examples where we chose to select the answer directly.
            # `labels` encodes the positions of the answer appearing in the table.
            if is_supervised:
                aggregate_mask = None
            else:
                if float_answer is not None:
                    assert (
                        labels.shape[0] == float_answer.shape[0]
                    ), "Make sure the answers are a FloatTensor of shape (batch_size,)"
                    # <float32>[batch_size]
                    aggregate_mask = _calculate_aggregate_mask(
                        float_answer,
                        pooled_output,
                        self.config.cell_selection_preference,
                        labels,
                        self.aggregation_classifier,
                    )
                else:
                    raise ValueError("You have to specify float answers in order to calculate the aggregate mask")

            # Cell selection log-likelihood
            if self.config.average_logits_per_cell:
                logits_per_cell, _ = reduce_mean(logits, cell_index)
                logits = gather(logits_per_cell, cell_index)
            # dist_per_token = torch.distributions.Bernoulli(logits=logits)
            dist_per_token_logits = logits

            # Compute cell selection loss per example.
            selection_loss_per_example = None
            if not self.config.select_one_column:
                weight = mint.where(
                    labels == 0,
                    mint.ones_like(labels, dtype=ms.float32),
                    self.config.positive_label_weight * mint.ones_like(labels, dtype=ms.float32),
                )
                dist_per_token_log_prob = -mint.nn.functional.binary_cross_entropy_with_logits(
                    dist_per_token_logits, labels, reduction="none"
                )
                selection_loss_per_token = -dist_per_token_log_prob * weight
                selection_loss_per_example = mint.sum(selection_loss_per_token * input_mask_float, dim=1) / (
                    mint.sum(input_mask_float, dim=1) + EPSILON_ZERO_DIVISION
                )
            else:
                selection_loss_per_example, logits = _single_column_cell_selection_loss(
                    logits, column_logits, labels, cell_index, col_index, cell_mask
                )
                dist_per_token_logits = logits

            # Supervised cell selection
            if self.config.disable_per_token_loss:
                pass
            elif is_supervised:
                total_loss += mint.mean(selection_loss_per_example)
            else:
                # For the not supervised case, do not assign loss for cell selection
                total_loss += mint.mean(selection_loss_per_example * (1.0 - aggregate_mask))

            # Semi-supervised regression loss and supervised loss for aggregations
            if self.config.num_aggregation_labels > 0:
                if is_supervised:
                    # Note that `aggregate_mask` is None if the setting is supervised.
                    if aggregation_labels is not None:
                        assert (
                            labels.shape[0] == aggregation_labels.shape[0]
                        ), "Make sure the aggregation labels are a LongTensor of shape (batch_size,)"
                        per_example_additional_loss = _calculate_aggregation_loss(
                            logits_aggregation,
                            aggregate_mask,
                            aggregation_labels,
                            self.config.use_answer_as_supervision,
                            self.config.num_aggregation_labels,
                            self.config.aggregation_loss_weight,
                        )
                    else:
                        raise ValueError(
                            "You have to specify aggregation labels in order to calculate the aggregation loss"
                        )
                else:
                    # Set aggregation labels to zeros
                    aggregation_labels = mint.zeros(labels.shape[0], dtype=ms.int64)
                    per_example_additional_loss = _calculate_aggregation_loss(
                        logits_aggregation,
                        aggregate_mask,
                        aggregation_labels,
                        self.config.use_answer_as_supervision,
                        self.config.num_aggregation_labels,
                        self.config.aggregation_loss_weight,
                    )

                if self.config.use_answer_as_supervision:
                    if numeric_values is not None and numeric_values_scale is not None:
                        assert numeric_values.shape == numeric_values_scale.shape
                        # Add regression loss for numeric answers which require aggregation.
                        answer_loss, large_answer_loss_mask = _calculate_regression_loss(
                            float_answer,
                            aggregate_mask,
                            dist_per_token_logits,
                            numeric_values,
                            numeric_values_scale,
                            table_mask_float,
                            logits_aggregation,
                            self.config,
                        )
                        per_example_additional_loss += answer_loss
                        # Zero loss for examples with answer_loss > cutoff.
                        per_example_additional_loss *= large_answer_loss_mask
                    else:
                        raise ValueError(
                            "You have to specify numeric values and numeric values scale in order to calculate the"
                            " regression loss"
                        )

                total_loss += mint.mean(per_example_additional_loss)

        else:
            # if no label ids are provided, set them to zeros in order to properly compute logits
            labels = mint.zeros_like(logits)
            _, logits = _single_column_cell_selection_loss(
                logits, column_logits, labels, cell_index, col_index, cell_mask
            )
        if not return_dict:
            output = (logits, logits_aggregation) + outputs[2:]
            return ((total_loss,) + output) if calculate_loss else output

        return TableQuestionAnsweringOutput(
            loss=total_loss if calculate_loss else None,
            logits=logits,
            logits_aggregation=logits_aggregation,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class TapasForSequenceClassification(TapasPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.tapas = TapasModel(config)
        self.dropout = mint.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = mint.nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        labels: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[ms.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`ms.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy). Note: this is called
            "classification_class_index" in the original implementation.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer
        >>> from mindone.transformers import TapasForSequenceClassification
        >>> import mindspore as ms
        >>> import pandas as pd

        >>> tokenizer = AutoTokenizer.from_pretrained("google/tapas-base-finetuned-tabfact", revision="refs/pr/1")
        >>> model = TapasForSequenceClassification.from_pretrained("google/tapas-base-finetuned-tabfact", revision="refs/pr/1")

        >>> data = {
        ...     "Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
        ...     "Age": ["56", "45", "59"],
        ...     "Number of movies": ["87", "53", "69"],
        ... }
        >>> table = pd.DataFrame.from_dict(data)
        >>> queries = [
        ...     "There is only one actor who is 45 years old",
        ...     "There are 3 actors which played in more than 60 movies",
        ... ]

        >>> inputs = tokenizer(table=table, queries=queries, padding="max_length", return_tensors="np")
        >>> inputs = {k: ms.tensor(v) for k, v in inputs.items()}
        >>> labels = ms.tensor([1, 0])  # 1 means entailed, 0 means refuted

        >>> outputs = model(**inputs, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.tapas(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == ms.int64 or labels.dtype == ms.int32):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


""" TAPAS utilities."""


class AverageApproximationFunction(str, enum.Enum):
    RATIO = "ratio"
    FIRST_ORDER = "first_order"
    SECOND_ORDER = "second_order"


# Beginning of everything related to segmented tensors


class IndexMap:
    """Index grouping entries within a tensor."""

    def __init__(self, indices, num_segments, batch_dims=0):
        """
        Creates an index

        Args:
            indices (`ms.Tensor`, same shape as a *values* Tensor to which the indices refer):
                Tensor containing the indices.
            num_segments (`ms.Tensor`):
                Scalar tensor, the number of segments. All elements in a batched segmented tensor must have the same
                number of segments (although many segments can be empty).
            batch_dims (`int`, *optional*, defaults to 0):
                The number of batch dimensions. The first *batch_dims* dimensions of a SegmentedTensor are treated as
                batch dimensions. Segments in different batch elements are always distinct even if they have the same
                index.
        """
        self.indices = ms.tensor(indices)
        self.num_segments = ms.tensor(num_segments)
        self.batch_dims = batch_dims

    def batch_shape(self):
        return self.indices.shape[: self.batch_dims]  # returns a shape object


class ProductIndexMap(IndexMap):
    """The product of two indices."""

    def __init__(self, outer_index, inner_index):
        """
        Combines indices i and j into pairs (i, j). The result is an index where each segment (i, j) is the
        intersection of segments i and j. For example if the inputs represent table cells indexed by respectively rows
        and columns the output will be a table indexed by (row, column) pairs, i.e. by cell. The implementation
        combines indices {0, .., n - 1} and {0, .., m - 1} into {0, .., nm - 1}. The output has *num_segments* equal to
        *outer_index.num_segments* * *inner_index.num_segments*

        Args:
            outer_index (`IndexMap`):
                IndexMap.
            inner_index (`IndexMap`):
                IndexMap, must have the same shape as *outer_index*.
        """
        if outer_index.batch_dims != inner_index.batch_dims:
            raise ValueError("outer_index.batch_dims and inner_index.batch_dims must be the same.")

        super().__init__(
            indices=(inner_index.indices + outer_index.indices * inner_index.num_segments),
            num_segments=inner_index.num_segments * outer_index.num_segments,
            batch_dims=inner_index.batch_dims,
        )
        self.outer_index = outer_index
        self.inner_index = inner_index

    def project_outer(self, index):
        """Projects an index with the same index set onto the outer components."""
        indices = mint.div(index.indices, self.inner_index.num_segments, rounding_mode="floor").type(ms.int64)
        return IndexMap(indices=indices, num_segments=self.outer_index.num_segments, batch_dims=index.batch_dims)

    def project_inner(self, index):
        """Projects an index with the same index set onto the inner components."""
        return IndexMap(
            indices=mint.fmod(index.indices, self.inner_index.num_segments).type(ms.int32).floor().type(ms.int64),
            num_segments=self.inner_index.num_segments,
            batch_dims=index.batch_dims,
        )


def gather(values, index, name="segmented_gather"):
    """
    Gathers from *values* using the index map. For each element in the domain of the index map this operation looks up
    a value for that index in *values*. Two elements from the same segment always get assigned the same value.

    Args:
        values (`ms.Tensor` of shape (B1, ..., Bn, num_segments, V1, ...)):
            Tensor with segment values.
        index (`IndexMap` of shape (B1, ..., Bn, I1, ..., Ik)):
            IndexMap.
        name (`str`, *optional*, defaults to 'segmented_gather'):
            Name for the operation. Currently not used

    Returns:
        `tuple(ms.Tensor)`: Tensor of shape (B1, ..., Bn, I1, ..., Ik, V1, ...) with the gathered values.
    """
    indices = index.indices
    # first, check whether the indices of the index represent scalar values (i.e. not vectorized)
    if len(values.shape[index.batch_dims :]) < 2:
        return mint.gather(
            values,
            index.batch_dims,
            indices.view(
                values.shape[0], -1
            ),  # mint.gather expects index to have the same number of dimensions as values
        ).view(indices.shape)
    else:
        # this means we have a vectorized version
        # we have to adjust the index
        indices = indices.unsqueeze(-1).broadcast_to(values.shape)
        return mint.gather(values, index.batch_dims, indices)


def flatten(index, name="segmented_flatten"):
    """
    Flattens a batched index map (which is typically of shape batch_size, seq_length) to a 1d index map. This operation
    relabels the segments to keep batch elements distinct. The k-th batch element will have indices shifted by
    *num_segments* * (k - 1). The result is a tensor with *num_segments* multiplied by the number of elements in the
    batch.

    Args:
        index (`IndexMap`):
            IndexMap to flatten.
        name (`str`, *optional*, defaults to 'segmented_flatten'):
            Name for the operation. Currently not used

    Returns:
        (`IndexMap`): The flattened IndexMap.
    """
    # first, get batch_size as scalar tensor
    batch_size = mint.prod(ms.tensor(list(index.batch_shape())))
    # next, create offset as 1-D tensor of length batch_size,
    # and multiply element-wise by num segments (to offset different elements in the batch) e.g. if batch size is 2: [0, 64]
    offset = mint.arange(start=0, end=batch_size.item()) * index.num_segments
    offset = offset.view(index.batch_shape())
    for _ in range(index.batch_dims, len(index.indices.shape)):  # typically range(1,2)
        offset = offset.unsqueeze(-1)

    indices = offset + index.indices
    return IndexMap(indices=indices.view(-1), num_segments=index.num_segments * batch_size, batch_dims=0)


def range_index_map(batch_shape, num_segments, name="range_index_map"):
    """
    Constructs an index map equal to range(num_segments).

    Args:
        batch_shape (`list`):
            Batch shape
        num_segments (`int`):
            Number of segments
        name (`str`, *optional*, defaults to 'range_index_map'):
            Name for the operation. Currently not used

    Returns:
        (`IndexMap`): IndexMap of shape batch_shape with elements equal to range(num_segments).
    """
    batch_shape = ms.tensor(
        batch_shape, dtype=ms.int64
    )  # create a rank 1 tensor vector containing batch_shape (e.g. [2])
    assert len(batch_shape.shape) == 1
    num_segments = ms.tensor(num_segments)  # create a rank 0 tensor (scalar) containing num_segments (e.g. 64)
    assert len(num_segments.shape) == 0

    indices = mint.arange(start=0, end=num_segments.item())  # create a rank 1 vector with num_segments elements
    new_tensor = mint.cat([mint.ones_like(batch_shape, dtype=ms.int64), num_segments.unsqueeze(dim=0)], dim=0)
    # new_tensor is just a vector of [1 64] for example (assuming only 1 batch dimension)
    new_shape = [int(x) for x in new_tensor.tolist()]
    indices = indices.view(tuple(new_shape))

    multiples = mint.cat([batch_shape, ms.tensor([1])], dim=0)
    indices = indices.tile(tuple(multiples.tolist()))
    # equivalent (in Numpy:)
    # indices = ms.as_tensor(np.tile(indices.numpy(), multiples.tolist()))

    return IndexMap(indices=indices, num_segments=num_segments, batch_dims=list(batch_shape.shape)[0])


def _segment_reduce(values, index, segment_reduce_fn, name):
    """
    Applies a segment reduction segment-wise.

    Args:
        values (`ms.Tensor`):
            Tensor with segment values.
        index (`IndexMap`):
            IndexMap.
        segment_reduce_fn (`str`):
            Name for the reduce operation. One of "sum", "mean", "max" or "min".
        name (`str`):
            Name for the operation. Currently not used

    Returns:
        (`IndexMap`): IndexMap of shape batch_shape with elements equal to range(num_segments).
    """
    # Flatten the batch dimensions, as segments ops (scatter) do not support batching.
    # However if `values` has extra dimensions to the right keep them
    # unflattened. Segmented ops support vector-valued operations.
    flat_index = flatten(index)
    vector_shape = values.shape[len(index.indices.shape) :]  # shape object
    flattened_shape = mint.cat([ms.tensor([-1], dtype=ms.int64), ms.tensor(vector_shape, dtype=ms.int64)], dim=0)
    # changed "view" by "reshape" in the following line
    flat_values = values.reshape(flattened_shape.tolist())

    out = mint.zeros(int(flat_index.num_segments), dtype=ms.float32)
    segment_means = scatter_reduce(
        out,
        dim=0,
        index=flat_index.indices.long(),
        src=flat_values.float(),
        reduce=segment_reduce_fn,
        include_self=False,
    )

    # Unflatten the values.
    new_shape = mint.cat(
        [
            ms.tensor(index.batch_shape(), dtype=ms.int64),
            ms.tensor([index.num_segments], dtype=ms.int64),
            ms.tensor(vector_shape, dtype=ms.int64),
        ],
        dim=0,
    )

    output_values = segment_means.clone().view(tuple(new_shape.tolist())).to(values.dtype)
    output_index = range_index_map(index.batch_shape(), index.num_segments)
    return output_values, output_index


def reduce_sum(values, index, name="segmented_reduce_sum"):
    """
    Sums a tensor over its segments.

    Outputs 0 for empty segments.

    This operations computes the sum over segments, with support for:

        - Batching using the first dimensions [B1, B2, ..., Bn]. Each element in a batch can have different indices.
        - Vectorization using the last dimension [V1, V2, ...]. If they are present, the output will be a sum of
          vectors rather than scalars. Only the middle dimensions [I1, ..., Ik] are reduced by the operation.

    Args:
        values (`ms.Tensor` of shape [B1, B2, ..., Bn, I1, .., Ik, V1, V2, ..]):
            Tensor containing the values of which the sum must be taken segment-wise.
        index (`IndexMap`, indices are of shape [B1, B2, ..., Bn, I1, .., Ik].):
            Index defining the segments.
        name (`str`, *optional*, defaults to 'segmented_reduce_sum'):
            Name for the operation. Currently not used

    Returns:
        output_values (`ms.Tensor`of shape [B1, B2, ..., Bn, num_segments, V1, V2, ..]): Tensor containing the
        output values. output_index (`IndexMap`): IndexMap with shape [B1, B2, ..., Bn, num_segments]. .
    """
    return _segment_reduce(values, index, "sum", name)


def reduce_mean(values, index, name="segmented_reduce_mean"):
    """
    Averages a tensor over its segments.

    Outputs 0 for empty segments.

    This operations computes the mean over segments, with support for:

        - Batching using the first dimensions [B1, B2, ..., Bn]. Each element in a batch can have different indices.
        - Vectorization using the last dimension [V1, V2, ...]. If they are present, the output will be a mean of
          vectors rather than scalars.

    Only the middle dimensions [I1, ..., Ik] are reduced by the operation.

    Args:
        values (`ms.Tensor` of shape [B1, B2, ..., Bn, I1, .., Ik, V1, V2, ..]):
            Tensor containing the values of which the mean must be taken segment-wise.
        index (`IndexMap`, indices are of shape [B1, B2, ..., Bn, I1, .., Ik].):
            Index defining the segments.
        name (`str`, *optional*, defaults to 'segmented_reduce_sum'):
            Name for the operation. Currently not used

    Returns:
        output_values (`ms.Tensor`of shape [B1, B2, ..., Bn, num_segments, V1, V2, ..]): Tensor containing the
        output values. output_index (`IndexMap`): IndexMap with shape [B1, B2, ..., Bn, num_segments].
    """
    return _segment_reduce(values, index, "mean", name)


def reduce_max(values, index, name="segmented_reduce_max"):
    """
    Computes the maximum over segments.

    This operation computes the maximum over segments, with support for:

        - Batching using the first dimensions [B1, B2, ..., Bn]. Each element in a batch can have different indices.
        - Vectorization using the last dimension [V1, V2, ...]. If they are present, the output will be an element-wise
          maximum of vectors rather than scalars.

    Only the middle dimensions [I1, ..., Ik] are reduced by the operation.

    Args:
        values (`ms.Tensor` of shape [B1, B2, ..., Bn, I1, .., Ik, V1, V2, ..]):
            Tensor containing the values of which the max must be taken segment-wise.
        index (`IndexMap`, indices are of shape [B1, B2, ..., Bn, I1, .., Ik].):
            Index defining the segments.
        name (`str`, *optional*, defaults to 'segmented_reduce_sum'):
            Name for the operation. Currently not used

    Returns:
        output_values (`ms.Tensor`of shape [B1, B2, ..., Bn, num_segments, V1, V2, ..]): Tensor containing the
        output values. output_index (`IndexMap`): IndexMap with shape [B1, B2, ..., Bn, num_segments].
    """
    return _segment_reduce(values, index, "amax", name)


def reduce_min(values, index, name="segmented_reduce_min"):
    """
    Computes the minimum over segments.

    This operations computes the minimum over segments, with support for:

        - Batching using the first dimensions [B1, B2, ..., Bn]. Each element in a batch can have different indices.
        - Vectorization using the last dimension [V1, V2, ...]. If they are present, the output will be an element-wise
          minimum of vectors rather than scalars.

    Only the middle dimensions [I1, ..., Ik] are reduced by the operation.

    Args:
        values (`ms.Tensor` of shape [B1, B2, ..., Bn, I1, .., Ik, V1, V2, ..]):
            Tensor containing the values of which the min must be taken segment-wise.
        index (`IndexMap`, indices are of shape [B1, B2, ..., Bn, I1, .., Ik].):
            Index defining the segments.
        name (`str`, *optional*, defaults to 'segmented_reduce_sum'):
            Name for the operation. Currently not used

    Returns:
        output_values (`ms.Tensor`of shape [B1, B2, ..., Bn, num_segments, V1, V2, ..]): Tensor containing the
        output values. output_index (`IndexMap`): IndexMap with shape [B1, B2, ..., Bn, num_segments].
    """
    return _segment_reduce(values, index, "amin", name)


# End of everything related to segmented tensors


def compute_column_logits(
    sequence_output, column_output_weights, column_output_bias, cell_index, cell_mask, allow_empty_column_selection
):
    """
    Computes the column logits.

    Args:
        sequence_output (`ms.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Also known as last_hidden_state. Sequence of hidden-states at the output of the last layer of the model.
        column_output_weights (`ms.Tensor` of shape `(hidden_size)`):
            Weights of the linear layer for column selection.
        column_output_bias (`ms.Tensor` of shape `()`):
            Bias of the linear layer for column selection.
        cell_index (`ProductIndexMap`):
            Index that groups tokens into cells.
        cell_mask (`ms.Tensor` of shape `(batch_size, max_num_rows * max_num_cols)`):
            Mask for cells that exist in the table (i.e. that are not padding).
        allow_empty_column_selection (`bool`):
            Whether to allow not to select any column

    Returns:
        column_logits (`ms.Tensor`of shape `(batch_size, max_num_cols)`): Tensor containing the column logits
        for every example in the batch.
    """

    # First, compute the token logits (batch_size, seq_len) - without temperature
    token_logits = mint.einsum("bsj,j->bs", sequence_output, column_output_weights) + column_output_bias

    # Next, average the logits per cell (batch_size, max_num_cols*max_num_rows)
    cell_logits, cell_logits_index = reduce_mean(token_logits, cell_index)

    # Finally, average the logits per column (batch_size, max_num_cols)
    column_index = cell_index.project_inner(cell_logits_index)
    column_logits, out_index = reduce_sum(cell_logits * cell_mask, column_index)

    cell_count, _ = reduce_sum(cell_mask, column_index)
    column_logits /= cell_count + EPSILON_ZERO_DIVISION

    # Mask columns that do not appear in the example.
    is_padding = mint.logical_and(cell_count < 0.5, ~mint.eq(out_index.indices, 0))
    column_logits += CLOSE_ENOUGH_TO_LOG_ZERO * ms.tensor(is_padding, dtype=ms.float32)

    if not allow_empty_column_selection:
        column_logits += CLOSE_ENOUGH_TO_LOG_ZERO * ms.tensor(mint.eq(out_index.indices, 0), dtype=ms.float32)

    return column_logits


def _single_column_cell_selection_loss(token_logits, column_logits, labels, cell_index, col_index, cell_mask):
    """
    Computes the loss for cell selection constrained to a single column. The loss is a hierarchical log-likelihood. The
    model first predicts a column and then selects cells within that column (conditioned on the column). Cells outside
    the selected column are never selected.

    Args:
        token_logits (`ms.Tensor` of shape `(batch_size, sequence_length)`):
            Tensor containing the logits per token.
        column_logits (`ms.Tensor` of shape `(batch_size, max_num_cols)`):
            Tensor containing the logits per column.
        labels (`ms.Tensor` of shape `(batch_size, sequence_length)`):
            Labels per token.
        cell_index (`ProductIndexMap`):
            Index that groups tokens into cells.
        col_index (`IndexMap`):
            Index that groups tokens into columns.
        cell_mask (`ms.Tensor` of shape `(batch_size, max_num_rows * max_num_cols)`):
            Mask for cells that exist in the table (i.e. that are not padding).

    Returns:
        selection_loss_per_example (`ms.Tensor` of shape `(batch_size,)`): Loss for each example. logits
        (`ms.Tensor` of shape `(batch_size, sequence_length)`): New logits which are only allowed to select
        cells in a single column. Logits outside of the most likely column according to *column_logits* will be set to
        a very low value (such that the probabilities are 0).
    """
    # Part 1: column loss

    # First find the column we should select. We use the column with maximum number of selected cells.
    labels_per_column, _ = reduce_sum(ms.tensor(labels, dtype=ms.float32), col_index)
    # shape of labels_per_column is (batch_size, max_num_cols). It contains the number of label ids for every column, for every example
    column_label = mint.argmax(labels_per_column, dim=-1)  # shape (batch_size,)
    # Check if there are no selected cells in the column. In that case the model
    # should predict the special column id 0, which means "select nothing".
    no_cell_selected = mint.eq(
        mint.max(labels_per_column, dim=-1)[0], 0
    )  # no_cell_selected is of shape (batch_size,) and equals True
    # if an example of the batch has no cells selected (i.e. if there are no labels set to 1 for that example)
    column_label = mint.where(no_cell_selected.view(column_label.shape), mint.zeros_like(column_label), column_label)

    log_probs = mint.nn.functional.log_softmax(column_logits, dim=-1)
    column_loss_per_example = -log_probs.gather(1, column_label.unsqueeze(1)).squeeze(1)

    # Part 2: cell loss

    # Reduce the labels and logits to per-cell from per-token.
    # logits_per_cell: shape (batch_size, max_num_rows*max_num_cols) i.e. (batch_size, 64*32)
    logits_per_cell, _ = reduce_mean(token_logits, cell_index)
    # labels_per_cell: shape (batch_size, 64*32), indicating whether each cell should be selected (1) or not (0)
    labels_per_cell, labels_index = reduce_max(ms.tensor(labels, dtype=ms.int64), cell_index)

    # Mask for the selected column.
    # column_id_for_cells: shape (batch_size, 64*32), indicating to which column each cell belongs
    column_id_for_cells = cell_index.project_inner(labels_index).indices
    # column_mask: shape (batch_size, 64*32), equal to 1 if cell belongs to column to be selected
    column_mask = ms.tensor(mint.eq(column_id_for_cells, mint.unsqueeze(column_label, dim=-1)), dtype=ms.float32)

    # Compute the log-likelihood for cells, but only for the selected column.
    cell_log_prob = -mint.nn.functional.binary_cross_entropy_with_logits(
        logits_per_cell, labels_per_cell.type(ms.float32), reduction="none"
    )  # shape(batch_size, 64*32)

    cell_loss = -mint.sum(cell_log_prob * column_mask * cell_mask, dim=1)

    # We need to normalize the loss by the number of cells in the column.
    cell_loss /= mint.sum(column_mask * cell_mask, dim=1) + EPSILON_ZERO_DIVISION

    selection_loss_per_example = column_loss_per_example
    selection_loss_per_example += mint.where(
        no_cell_selected.view(selection_loss_per_example.shape),
        mint.zeros_like(selection_loss_per_example),
        cell_loss,
    )

    # Set the probs outside the selected column (selected by the *model*)
    # to 0. This ensures backwards compatibility with models that select
    # cells from multiple columns.
    selected_column_id = ms.tensor(mint.argmax(column_logits, dim=-1), dtype=ms.int64)  # shape (batch_size,)

    # selected_column_mask: shape (batch_size, 64*32), equal to 1 if cell belongs to column selected by the model
    selected_column_mask = ms.tensor(
        mint.eq(column_id_for_cells, mint.unsqueeze(selected_column_id, dim=-1)), dtype=ms.float32
    )

    # Never select cells with the special column id 0.
    selected_column_mask = mint.where(
        mint.eq(column_id_for_cells, 0).view(selected_column_mask.shape),
        mint.zeros_like(selected_column_mask),
        selected_column_mask,
    )
    new_logits_per_cell = logits_per_cell + CLOSE_ENOUGH_TO_LOG_ZERO * (1.0 - cell_mask * selected_column_mask)
    logits = gather(new_logits_per_cell, cell_index)

    return selection_loss_per_example, logits


def compute_token_logits(sequence_output, temperature, output_weights, output_bias):
    """
    Computes logits per token

    Args:
        sequence_output (`ms.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Also known as last_hidden_state. Sequence of hidden-states at the output of the last layer of the model.
        temperature (`float`):
            Temperature for the Bernoulli distribution.
        output_weights (`ms.Tensor` of shape `(hidden_size,)`):
            Weights of the linear layer for cell selection.
        output_bias (`ms.Tensor` of shape `()`):
            Bias of the linear layer for cell selection

    Returns:
        logits (`ms.Tensor` of shape `(batch_size, sequence_length)`): Logits per token.
    """
    logits = (mint.einsum("bsj,j->bs", sequence_output, output_weights) + output_bias) / temperature

    return logits


def _calculate_aggregate_mask(answer, pooled_output, cell_selection_preference, labels, aggregation_classifier):
    """
    Finds examples where the model should select cells with no aggregation.

    Returns a mask that determines for which examples should the model select answers directly from the table, without
    any aggregation function. If the answer is a piece of text the case is unambiguous as aggregation functions only
    apply to numbers. If the answer is a number but does not appear in the table then we must use some aggregation
    case. The ambiguous case is when the answer is a number that also appears in the table. In this case we use the
    aggregation function probabilities predicted by the model to decide whether to select or aggregate. The threshold
    for this is a hyperparameter *cell_selection_preference*

    Args:
        answer (`ms.Tensor` of shape `(batch_size, )`):
            Answer for every example in the batch. Nan if there is no scalar answer.
        pooled_output (`ms.Tensor` of shape `(batch_size, hidden_size)`):
            Output of the pooler (BertPooler) on top of the encoder layer.
        cell_selection_preference (`float`):
            Preference for cell selection in ambiguous cases.
        labels (`ms.Tensor` of shape `(batch_size, sequence_length)`):
            Labels per token. aggregation_classifier (`mint.nn.Linear`): Aggregation head

    Returns:
        aggregate_mask (`ms.Tensor` of shape `(batch_size,)`): A mask set to 1 for examples that should use
        aggregation functions.
    """
    # ms.Tensor(batch_size,)
    aggregate_mask_init = mint.logical_not(mint.isnan(answer)).type(ms.float32)
    logits_aggregation = aggregation_classifier(pooled_output)
    dist_probs = mint.softmax(logits_aggregation, dim=-1)[:, 1:]
    # Index 0 corresponds to "no aggregation".
    aggregation_ops_total_mass = mint.sum(dist_probs, dim=1)

    # Cell selection examples according to current model.
    is_pred_cell_selection = aggregation_ops_total_mass <= cell_selection_preference

    # Examples with non-empty cell selection supervision.
    is_cell_supervision_available = mint.sum(labels, dim=1) > 0

    # mint.where is not equivalent to tf.where (in tensorflow 1)
    # hence the added .view on the condition to match the shape of the first tensor
    aggregate_mask = mint.where(
        mint.logical_and(is_pred_cell_selection, is_cell_supervision_available).view(aggregate_mask_init.shape),
        mint.zeros_like(aggregate_mask_init, dtype=ms.float32),
        aggregate_mask_init,
    )

    aggregate_mask = ops.stop_gradient(aggregate_mask)

    return aggregate_mask


def _calculate_aggregation_loss_known(
    logits_aggregation, aggregate_mask, aggregation_labels, use_answer_as_supervision, num_aggregation_labels
):
    """
    Calculates aggregation loss when its type is known during training.

    In the weakly supervised setting, the only known information is that for cell selection examples, "no aggregation"
    should be predicted. For other examples (those that require aggregation), no loss is accumulated. In the setting
    where aggregation type is always known, standard cross entropy loss is accumulated for all examples

    Args:
        logits_aggregation (`ms.Tensor` of shape `(batch_size, num_aggregation_labels)`):
            Logits per aggregation operation.
        aggregate_mask (`ms.Tensor` of shape `(batch_size, )`):
            A mask set to 1 for examples that should use aggregation functions.
        aggregation_labels (`ms.Tensor` of shape `(batch_size, )`):
            Aggregation function id for every example in the batch.
        use_answer_as_supervision (`bool`, *optional*):
            Whether to use the answer as the only supervision for aggregation examples.
        num_aggregation_labels (`int`, *optional*, defaults to 0):
            The number of aggregation operators to predict.

    Returns:
        aggregation_loss_known (`ms.Tensor` of shape `(batch_size,)`): Aggregation loss (when its type is known
        during training) per example.
    """
    if use_answer_as_supervision:
        # Prepare "no aggregation" targets for cell selection examples.
        target_aggregation = mint.zeros_like(aggregate_mask, dtype=ms.int64)
    else:
        # Use aggregation supervision as the target.
        target_aggregation = aggregation_labels

    one_hot_labels = mint.nn.functional.one_hot(target_aggregation, num_classes=num_aggregation_labels).type(ms.float32)
    log_probs = mint.nn.functional.log_softmax(logits_aggregation, dim=-1)

    # ms.Tensor[batch_size]
    per_example_aggregation_intermediate = -mint.sum(one_hot_labels * log_probs, dim=-1)
    if use_answer_as_supervision:
        # Accumulate loss only for examples requiring cell selection
        # (no aggregation).
        return per_example_aggregation_intermediate * (1 - aggregate_mask)
    else:
        return per_example_aggregation_intermediate


def _calculate_aggregation_loss_unknown(logits_aggregation, aggregate_mask):
    """
    Calculates aggregation loss in the case of answer supervision.

    Args:
        logits_aggregation (`ms.Tensor` of shape `(batch_size, num_aggregation_labels)`):
            Logits per aggregation operation.
        aggregate_mask (`ms.Tensor` of shape `(batch_size, )`):
            A mask set to 1 for examples that should use aggregation functions

    Returns:
        aggregation_loss_unknown (`ms.Tensor` of shape `(batch_size,)`): Aggregation loss (in case of answer
        supervision) per example.
    """
    dist_probs = mint.softmax(logits_aggregation, dim=-1)[:, 1:]
    # Index 0 corresponds to "no aggregation".
    aggregation_ops_total_mass = mint.sum(dist_probs, dim=1)
    # Predict some aggregation in case of an answer that needs aggregation.
    # This increases the probability of all aggregation functions, in a way
    # similar to MML, but without considering whether the function gives the
    # correct answer.
    return -mint.log(aggregation_ops_total_mass) * aggregate_mask


def _calculate_aggregation_loss(
    logits_aggregation,
    aggregate_mask,
    aggregation_labels,
    use_answer_as_supervision,
    num_aggregation_labels,
    aggregation_loss_weight,
):
    """
    Calculates the aggregation loss per example.

    Args:
        logits_aggregation (`ms.Tensor` of shape `(batch_size, num_aggregation_labels)`):
            Logits per aggregation operation.
        aggregate_mask (`ms.Tensor` of shape `(batch_size, )`):
            A mask set to 1 for examples that should use aggregation functions.
        aggregation_labels (`ms.Tensor` of shape `(batch_size, )`):
            Aggregation function id for every example in the batch.
        use_answer_as_supervision (`bool`, *optional*):
            Whether to use the answer as the only supervision for aggregation examples.
        num_aggregation_labels (`int`, *optional*, defaults to 0):
            The number of aggregation operators to predict.
        aggregation_loss_weight (`float`, *optional*, defaults to 1.0):
            Importance weight for the aggregation loss.

    Returns:
        aggregation_loss (`ms.Tensor` of shape `(batch_size,)`): Aggregation loss per example.
    """
    per_example_aggregation_loss = _calculate_aggregation_loss_known(
        logits_aggregation, aggregate_mask, aggregation_labels, use_answer_as_supervision, num_aggregation_labels
    )

    if use_answer_as_supervision:
        # Add aggregation loss for numeric answers that need aggregation.
        per_example_aggregation_loss += _calculate_aggregation_loss_unknown(logits_aggregation, aggregate_mask)
    return aggregation_loss_weight * per_example_aggregation_loss


def _calculate_expected_result(
    dist_per_cell_logits, numeric_values, numeric_values_scale, input_mask_float, logits_aggregation, config
):
    """
    Calculates the expected result given cell and aggregation probabilities.

    Args:
        dist_per_cell (`torch.distributions.Bernoulli`):
            Cell selection distribution for each cell.
        numeric_values (`ms.Tensor` of shape `(batch_size, seq_length)`):
            Numeric values of every token. Nan for tokens which are not numeric values.
        numeric_values_scale (`ms.Tensor` of shape `(batch_size, seq_length)`):
            Scale of the numeric values of every token.
        input_mask_float (`ms.Tensor` of shape `(batch_size, seq_length)`):
            Mask for the table, without question tokens and table headers.
        logits_aggregation (`ms.Tensor` of shape `(batch_size, num_aggregation_labels)`):
            Logits per aggregation operation.
        config ([`TapasConfig`]):
            Model configuration class with all the hyperparameters of the model

    Returns:
        expected_result (`ms.Tensor` of shape `(batch_size,)`): The expected result per example.
    """
    if config.use_gumbel_for_cells:
        u = mint.rand_like(dist_per_cell_logits)
        g = -mint.log(-mint.log(u + 1e-10))

        scaled_probability_per_cell = mint.sigmoid((dist_per_cell_logits * config.temperature + g) / config.temperature)
    else:
        scaled_probability_per_cell = mint.sigmoid(dist_per_cell_logits)

    # <float32>[batch_size, seq_length]
    scaled_probability_per_cell = (scaled_probability_per_cell / numeric_values_scale) * input_mask_float
    count_result = mint.sum(scaled_probability_per_cell, dim=1)
    numeric_values_masked = mint.where(
        mint.isnan(numeric_values), mint.zeros_like(numeric_values), numeric_values
    )  # Mask non-numeric table values to zero.
    sum_result = mint.sum(scaled_probability_per_cell * numeric_values_masked, dim=1)
    avg_approximation = config.average_approximation_function
    if avg_approximation == AverageApproximationFunction.RATIO:
        average_result = sum_result / (count_result + EPSILON_ZERO_DIVISION)
    elif avg_approximation == AverageApproximationFunction.FIRST_ORDER:
        # The sum of all probabilities except that correspond to other cells
        # Ex here stands for expectation, more explicitly the expectation of the sum of N-1 Bernoulli random variables plus
        # the constant 1, which is computed as adding all N expected values and subtracting the extra one. It corresponds to X_c
        # in Appendix D of the original TAPAS paper which is trying to approximate the average of a random set.
        ex = mint.sum(scaled_probability_per_cell, dim=1, keepdim=True) - scaled_probability_per_cell + 1
        average_result = mint.sum(numeric_values_masked * scaled_probability_per_cell / ex, dim=1)
    elif avg_approximation == AverageApproximationFunction.SECOND_ORDER:
        # The sum of all probabilities except that correspond to other cells
        ex = mint.sum(scaled_probability_per_cell, dim=1, keepdim=True) - scaled_probability_per_cell + 1
        pointwise_var = scaled_probability_per_cell * (1 - scaled_probability_per_cell)
        var = mint.sum(pointwise_var, dim=1, keepdim=True) - pointwise_var

        multiplier = (var / mint.square(ex) + 1) / ex
        average_result = mint.sum(numeric_values_masked * scaled_probability_per_cell * multiplier, dim=1)
    else:
        raise ValueError(f"Invalid average_approximation_function: {config.average_approximation_function}")

    if config.use_gumbel_for_aggregation:
        u = mint.empty_like(logits_aggregation[:, 1:]).uniform_(1e-6, 1 - 1e-6)
        g = -mint.log(-mint.log(u))

        noisy_logits = (logits_aggregation[:, 1:] + g) / config.aggregation_temperature
        # <float32>[batch_size, num_aggregation_labels - 1]
        aggregation_op_only_probs = mint.softmax(noisy_logits, dim=-1)
    else:
        # <float32>[batch_size, num_aggregation_labels - 1]
        aggregation_op_only_probs = mint.nn.functional.softmax(
            logits_aggregation[:, 1:] / config.aggregation_temperature, dim=-1
        )

    all_results = mint.cat(
        [
            mint.unsqueeze(sum_result, dim=1),
            mint.unsqueeze(average_result, dim=1),
            mint.unsqueeze(count_result, dim=1),
        ],
        dim=1,
    )

    expected_result = mint.sum(all_results * aggregation_op_only_probs, dim=1)
    return expected_result


# MindSpore does not currently support Huber loss with custom delta so we define it ourself
def huber_loss(input, target, delta: float = 1.0):
    errors = mint.abs(input - target)  # shape (batch_size,)
    return mint.where(errors < delta, 0.5 * errors**2, errors * delta - (0.5 * delta**2))


def _calculate_regression_loss(
    answer,
    aggregate_mask,
    dist_per_cell_logits,
    numeric_values,
    numeric_values_scale,
    input_mask_float,
    logits_aggregation,
    config,
):
    """
    Calculates the regression loss per example.

    Args:
        answer (`ms.Tensor` of shape `(batch_size,)`):
            Answer for every example in the batch. Nan if there is no scalar answer.
        aggregate_mask (`ms.Tensor` of shape `(batch_size,)`):
            A mask set to 1 for examples that should use aggregation functions.
        dist_per_cell_logits (`ms.Tensor`):
            Cell selection distribution logits for each cell.
        numeric_values (`ms.Tensor` of shape `(batch_size, seq_length)`):
            Numeric values of every token. Nan for tokens which are not numeric values.
        numeric_values_scale (`ms.Tensor` of shape `(batch_size, seq_length)`):
            Scale of the numeric values of every token.
        input_mask_float (`ms.Tensor` of shape `(batch_size, seq_length)`):
            Mask for the table, without question tokens and table headers.
        logits_aggregation (`ms.Tensor` of shape `(batch_size, num_aggregation_labels)`):
            Logits per aggregation operation.
        config ([`TapasConfig`]):
            Model configuration class with all the parameters of the model

    Returns:
        per_example_answer_loss_scaled (`ms.Tensor` of shape `(batch_size,)`): Scales answer loss for each
        example in the batch. large_answer_loss_mask (`ms.Tensor` of shape `(batch_size,)`): A mask which is 1
        for examples for which their answer loss is larger than the answer_loss_cutoff.
    """
    # float32 (batch_size,)
    expected_result = _calculate_expected_result(
        dist_per_cell_logits, numeric_values, numeric_values_scale, input_mask_float, logits_aggregation, config
    )

    # float32 (batch_size,)
    answer_masked = mint.where(mint.isnan(answer), mint.zeros_like(answer), answer)

    if config.use_normalized_answer_loss:
        normalizer = ops.stop_gradient(
            mint.max(mint.abs(expected_result), mint.abs(answer_masked)) + EPSILON_ZERO_DIVISION
        )

        normalized_answer_masked = answer_masked / normalizer
        normalized_expected_result = expected_result / normalizer
        per_example_answer_loss = huber_loss(
            normalized_expected_result * aggregate_mask, normalized_answer_masked * aggregate_mask
        )
    else:
        per_example_answer_loss = huber_loss(
            expected_result * aggregate_mask, answer_masked * aggregate_mask, delta=config.huber_loss_delta
        )

    if config.answer_loss_cutoff is None:
        large_answer_loss_mask = mint.ones_like(per_example_answer_loss, dtype=ms.float32)

    else:
        large_answer_loss_mask = mint.where(
            per_example_answer_loss > config.answer_loss_cutoff,
            mint.zeros_like(per_example_answer_loss, dtype=ms.float32),
            mint.ones_like(per_example_answer_loss, dtype=ms.float32),
        )
    per_example_answer_loss_scaled = config.answer_loss_importance * (per_example_answer_loss * aggregate_mask)

    return per_example_answer_loss_scaled, large_answer_loss_mask


__all__ = [
    "TapasForMaskedLM",
    "TapasForQuestionAnswering",
    "TapasForSequenceClassification",
    "TapasModel",
    "TapasPreTrainedModel",
]
