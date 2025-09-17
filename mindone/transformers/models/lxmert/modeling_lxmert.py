# coding=utf-8
# Copyright 2018 Hao Tan, Mohit Bansal, and the HuggingFace team
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
"""MindSpore LXMERT model."""

import math
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

from transformers.modeling_outputs import ModelOutput
from transformers.models.lxmert.configuration_lxmert import LxmertConfig

import mindspore as ms
import mindspore.mint as mint
from mindspore import nn
from mindspore.mint.nn import CrossEntropyLoss, SmoothL1Loss

from ...activations import ACT2FN, gelu
from ...mindspore_adapter import dtype_to_min
from ...modeling_utils import PreTrainedModel
from ...utils import logging

logger = logging.get_logger(__name__)


class GeLU(nn.Cell):
    def __init__(self):
        super().__init__()

    def construct(self, x):
        return gelu(x)


@dataclass
class LxmertModelOutput(ModelOutput):
    """
    Lxmert's outputs that contain the last hidden states, pooled outputs, and attention probabilities for the language,
    visual, and, cross-modality encoders. (note: the visual encoder in Lxmert is referred to as the "relation-ship"
    encoder")


    Args:
        language_output (`ms.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the language encoder.
        vision_output (`ms.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the visual encoder.
        pooled_output (`ms.Tensor` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification, CLS, token) further processed
            by a Linear layer and a Tanh activation function. The Linear
        language_hidden_states (`tuple(ms.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `ms.Tensor` (one for input features + one for the output of each cross-modality layer) of
            shape `(batch_size, sequence_length, hidden_size)`.
        vision_hidden_states (`tuple(ms.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `ms.Tensor` (one for input features + one for the output of each cross-modality layer) of
            shape `(batch_size, sequence_length, hidden_size)`.
        language_attentions (`tuple(ms.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `ms.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        vision_attentions (`tuple(ms.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `ms.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        cross_encoder_attentions (`tuple(ms.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `ms.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    language_output: Optional[ms.Tensor] = None
    vision_output: Optional[ms.Tensor] = None
    pooled_output: Optional[ms.Tensor] = None
    language_hidden_states: Optional[Tuple[ms.Tensor]] = None
    vision_hidden_states: Optional[Tuple[ms.Tensor]] = None
    language_attentions: Optional[Tuple[ms.Tensor]] = None
    vision_attentions: Optional[Tuple[ms.Tensor]] = None
    cross_encoder_attentions: Optional[Tuple[ms.Tensor]] = None


@dataclass
class LxmertForQuestionAnsweringOutput(ModelOutput):
    """
    Output type of [`LxmertForQuestionAnswering`].

    Args:
        loss (*optional*, returned when `labels` is provided, `ms.Tensor` of shape `(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.k.
        question_answering_score (`ms.Tensor` of shape `(batch_size, n_qa_answers)`, *optional*):
            Prediction scores of question answering objective (classification).
        language_hidden_states (`tuple(ms.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `ms.Tensor` (one for input features + one for the output of each cross-modality layer) of
            shape `(batch_size, sequence_length, hidden_size)`.
        vision_hidden_states (`tuple(ms.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `ms.Tensor` (one for input features + one for the output of each cross-modality layer) of
            shape `(batch_size, sequence_length, hidden_size)`.
        language_attentions (`tuple(ms.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `ms.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        vision_attentions (`tuple(ms.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `ms.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        cross_encoder_attentions (`tuple(ms.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `ms.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    loss: Optional[ms.Tensor] = None
    question_answering_score: Optional[ms.Tensor] = None
    language_hidden_states: Optional[Tuple[ms.Tensor]] = None
    vision_hidden_states: Optional[Tuple[ms.Tensor]] = None
    language_attentions: Optional[Tuple[ms.Tensor]] = None
    vision_attentions: Optional[Tuple[ms.Tensor]] = None
    cross_encoder_attentions: Optional[Tuple[ms.Tensor]] = None


@dataclass
class LxmertForPreTrainingOutput(ModelOutput):
    """
    Output type of [`LxmertForPreTraining`].

    Args:
        loss (*optional*, returned when `labels` is provided, `ms.Tensor` of shape `(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (`ms.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        cross_relationship_score (`ms.Tensor` of shape `(batch_size, 2)`):
            Prediction scores of the textual matching objective (classification) head (scores of True/False
            continuation before SoftMax).
        question_answering_score (`ms.Tensor` of shape `(batch_size, n_qa_answers)`):
            Prediction scores of question answering objective (classification).
        language_hidden_states (`tuple(ms.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `ms.Tensor` (one for input features + one for the output of each cross-modality layer) of
            shape `(batch_size, sequence_length, hidden_size)`.
        vision_hidden_states (`tuple(ms.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `ms.Tensor` (one for input features + one for the output of each cross-modality layer) of
            shape `(batch_size, sequence_length, hidden_size)`.
        language_attentions (`tuple(ms.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `ms.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        vision_attentions (`tuple(ms.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `ms.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        cross_encoder_attentions (`tuple(ms.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `ms.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.

    """

    loss: Optional[ms.Tensor] = None
    prediction_logits: Optional[ms.Tensor] = None
    cross_relationship_score: Optional[ms.Tensor] = None
    question_answering_score: Optional[ms.Tensor] = None
    language_hidden_states: Optional[Tuple[ms.Tensor]] = None
    vision_hidden_states: Optional[Tuple[ms.Tensor]] = None
    language_attentions: Optional[Tuple[ms.Tensor]] = None
    vision_attentions: Optional[Tuple[ms.Tensor]] = None
    cross_encoder_attentions: Optional[Tuple[ms.Tensor]] = None


class LxmertEmbeddings(nn.Cell):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = mint.nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = mint.nn.Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=0)
        self.token_type_embeddings = mint.nn.Embedding(config.type_vocab_size, config.hidden_size, padding_idx=0)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = mint.nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = mint.nn.Dropout(config.hidden_dropout_prob)

    def construct(self, input_ids, token_type_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]
        seq_length = input_shape[1]

        position_ids = mint.arange(seq_length, dtype=ms.int64)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)

        if token_type_ids is None:
            token_type_ids = mint.zeros(input_shape, dtype=ms.int64)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class LxmertAttention(nn.Cell):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.head_size = self.num_attention_heads * self.attention_head_size

        # visual_dim = 2048
        if ctx_dim is None:
            ctx_dim = config.hidden_size
        self.query = mint.nn.Linear(config.hidden_size, self.head_size)
        self.key = mint.nn.Linear(ctx_dim, self.head_size)
        self.value = mint.nn.Linear(ctx_dim, self.head_size)

        self.dropout = mint.nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.shape[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def construct(self, hidden_states, context, attention_mask=None, output_attentions=False):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = mint.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel construct() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = mint.nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = mint.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.shape[:-2] + (self.head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class LxmertAttentionOutput(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.dense = mint.nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = mint.nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = mint.nn.Dropout(config.hidden_dropout_prob)

    def construct(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LxmertCrossAttentionLayer(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.att = LxmertAttention(config)
        self.output = LxmertAttentionOutput(config)

    def construct(self, input_tensor, ctx_tensor, ctx_att_mask=None, output_attentions=False):
        output = self.att(input_tensor, ctx_tensor, ctx_att_mask, output_attentions=output_attentions)
        if output_attentions:
            attention_probs = output[1]
        attention_output = self.output(output[0], input_tensor)
        outputs = (attention_output, attention_probs) if output_attentions else (attention_output,)
        return outputs


class LxmertSelfAttentionLayer(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.self = LxmertAttention(config)
        self.output = LxmertAttentionOutput(config)

    def construct(self, input_tensor, attention_mask, output_attentions=False):
        # Self attention attends to itself, thus keys and queries are the same (input_tensor).
        output = self.self(
            input_tensor,
            input_tensor,
            attention_mask,
            output_attentions=output_attentions,
        )
        if output_attentions:
            attention_probs = output[1]
        attention_output = self.output(output[0], input_tensor)
        outputs = (attention_output, attention_probs) if output_attentions else (attention_output,)
        return outputs


class LxmertIntermediate(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.dense = mint.nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act]

    def construct(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class LxmertOutput(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.dense = mint.nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = mint.nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = mint.nn.Dropout(config.hidden_dropout_prob)

    def construct(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LxmertLayer(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.attention = LxmertSelfAttentionLayer(config)
        self.intermediate = LxmertIntermediate(config)
        self.output = LxmertOutput(config)

    def construct(self, hidden_states, attention_mask=None, output_attentions=False):
        outputs = self.attention(hidden_states, attention_mask, output_attentions=output_attentions)
        attention_output = outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs[1:]  # add attentions if we output them
        return outputs


class LxmertXLayer(nn.Cell):
    def __init__(self, config):
        super().__init__()
        # The cross-attention Layer
        self.visual_attention = LxmertCrossAttentionLayer(config)

        # Self-attention Layers
        self.lang_self_att = LxmertSelfAttentionLayer(config)
        self.visn_self_att = LxmertSelfAttentionLayer(config)

        # Intermediate and Output Layers (FFNs)
        self.lang_inter = LxmertIntermediate(config)
        self.lang_output = LxmertOutput(config)
        self.visn_inter = LxmertIntermediate(config)
        self.visn_output = LxmertOutput(config)

    def cross_att(
        self,
        lang_input,
        lang_attention_mask,
        visual_input,
        visual_attention_mask,
        output_x_attentions=False,
    ):
        # Cross Attention
        lang_att_output = self.visual_attention(
            lang_input,
            visual_input,
            ctx_att_mask=visual_attention_mask,
            output_attentions=output_x_attentions,
        )
        visual_att_output = self.visual_attention(
            visual_input,
            lang_input,
            ctx_att_mask=lang_attention_mask,
            output_attentions=False,
        )
        return lang_att_output, visual_att_output

    def self_att(self, lang_input, lang_attention_mask, visual_input, visual_attention_mask):
        # Self Attention
        lang_att_output = self.lang_self_att(lang_input, lang_attention_mask, output_attentions=False)
        visual_att_output = self.visn_self_att(visual_input, visual_attention_mask, output_attentions=False)
        return lang_att_output[0], visual_att_output[0]

    def output_fc(self, lang_input, visual_input):
        # FC layers
        lang_inter_output = self.lang_inter(lang_input)
        visual_inter_output = self.visn_inter(visual_input)

        # Layer output
        lang_output = self.lang_output(lang_inter_output, lang_input)
        visual_output = self.visn_output(visual_inter_output, visual_input)

        return lang_output, visual_output

    def construct(
        self,
        lang_feats,
        lang_attention_mask,
        visual_feats,
        visual_attention_mask,
        output_attentions=False,
    ):
        lang_att_output, visual_att_output = self.cross_att(
            lang_input=lang_feats,
            lang_attention_mask=lang_attention_mask,
            visual_input=visual_feats,
            visual_attention_mask=visual_attention_mask,
            output_x_attentions=output_attentions,
        )
        attention_probs = lang_att_output[1:]
        lang_att_output, visual_att_output = self.self_att(
            lang_att_output[0],
            lang_attention_mask,
            visual_att_output[0],
            visual_attention_mask,
        )

        lang_output, visual_output = self.output_fc(lang_att_output, visual_att_output)
        return (
            (
                lang_output,
                visual_output,
                attention_probs[0],
            )
            if output_attentions
            else (lang_output, visual_output)
        )


class LxmertVisualFeatureEncoder(nn.Cell):
    def __init__(self, config):
        super().__init__()
        feat_dim = config.visual_feat_dim
        pos_dim = config.visual_pos_dim

        # Object feature encoding
        self.visn_fc = mint.nn.Linear(feat_dim, config.hidden_size)
        self.visn_layer_norm = mint.nn.LayerNorm(config.hidden_size, eps=1e-12)

        # Box position encoding
        self.box_fc = mint.nn.Linear(pos_dim, config.hidden_size)
        self.box_layer_norm = mint.nn.LayerNorm(config.hidden_size, eps=1e-12)

        self.dropout = mint.nn.Dropout(config.hidden_dropout_prob)

    def construct(self, visual_feats, visual_pos):
        x = self.visn_fc(visual_feats)
        x = self.visn_layer_norm(x)
        y = self.box_fc(visual_pos)
        y = self.box_layer_norm(y)
        output = (x + y) / 2

        output = self.dropout(output)
        return output


class LxmertEncoder(nn.Cell):
    def __init__(self, config):
        super().__init__()

        # Obj-level image embedding layer
        self.visn_fc = LxmertVisualFeatureEncoder(config)
        self.config = config

        # Number of layers
        self.num_l_layers = config.l_layers
        self.num_x_layers = config.x_layers
        self.num_r_layers = config.r_layers

        # Layers
        # Using self.layer instead of self.l_layer to support loading BERT weights.
        self.layer = nn.CellList([LxmertLayer(config) for _ in range(self.num_l_layers)])
        self.x_layers = nn.CellList([LxmertXLayer(config) for _ in range(self.num_x_layers)])
        self.r_layers = nn.CellList([LxmertLayer(config) for _ in range(self.num_r_layers)])

    def construct(
        self,
        lang_feats,
        lang_attention_mask,
        visual_feats,
        visual_pos,
        visual_attention_mask=None,
        output_attentions=None,
    ):
        vision_hidden_states = ()
        language_hidden_states = ()
        vision_attentions = () if output_attentions or self.config.output_attentions else None
        language_attentions = () if output_attentions or self.config.output_attentions else None
        cross_encoder_attentions = () if output_attentions or self.config.output_attentions else None

        visual_feats = self.visn_fc(visual_feats, visual_pos)

        # Run language layers
        for layer_module in self.layer:
            l_outputs = layer_module(lang_feats, lang_attention_mask, output_attentions=output_attentions)
            lang_feats = l_outputs[0]
            language_hidden_states = language_hidden_states + (lang_feats,)
            if language_attentions is not None:
                language_attentions = language_attentions + (l_outputs[1],)

        # Run relational layers
        for layer_module in self.r_layers:
            v_outputs = layer_module(visual_feats, visual_attention_mask, output_attentions=output_attentions)
            visual_feats = v_outputs[0]
            vision_hidden_states = vision_hidden_states + (visual_feats,)
            if vision_attentions is not None:
                vision_attentions = vision_attentions + (v_outputs[1],)

        # Run cross-modality layers
        for layer_module in self.x_layers:
            x_outputs = layer_module(
                lang_feats,
                lang_attention_mask,
                visual_feats,
                visual_attention_mask,
                output_attentions=output_attentions,
            )
            lang_feats, visual_feats = x_outputs[:2]
            vision_hidden_states = vision_hidden_states + (visual_feats,)
            language_hidden_states = language_hidden_states + (lang_feats,)
            if cross_encoder_attentions is not None:
                cross_encoder_attentions = cross_encoder_attentions + (x_outputs[2],)
        visual_encoder_outputs = (
            vision_hidden_states,
            vision_attentions if output_attentions else None,
        )
        lang_encoder_outputs = (
            language_hidden_states,
            language_attentions if output_attentions else None,
        )
        return (
            visual_encoder_outputs,
            lang_encoder_outputs,
            cross_encoder_attentions if output_attentions else None,
        )


class LxmertPooler(nn.Cell):
    def __init__(self, config):
        super(LxmertPooler, self).__init__()
        self.dense = mint.nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = mint.nn.Tanh()

    def construct(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class LxmertPredictionHeadTransform(nn.Cell):
    def __init__(self, config):
        super(LxmertPredictionHeadTransform, self).__init__()
        self.dense = mint.nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act]
        self.LayerNorm = mint.nn.LayerNorm(config.hidden_size, eps=1e-12)

    def construct(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class LxmertLMPredictionHead(nn.Cell):
    def __init__(self, config, lxmert_model_embedding_weights):
        super(LxmertLMPredictionHead, self).__init__()
        self.transform = LxmertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = mint.nn.Linear(
            lxmert_model_embedding_weights.shape[1],
            lxmert_model_embedding_weights.shape[0],
            bias=False,
        )
        self.decoder.weight = lxmert_model_embedding_weights
        self.bias = ms.Parameter(mint.zeros(lxmert_model_embedding_weights.shape[0]))

    def construct(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class LxmertVisualAnswerHead(nn.Cell):
    def __init__(self, config, num_labels):
        super().__init__()
        hid_dim = config.hidden_size
        self.logit_fc = nn.SequentialCell(
            mint.nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            mint.nn.LayerNorm(hid_dim * 2, eps=1e-12),
            mint.nn.Linear(hid_dim * 2, num_labels),
        )

    def construct(self, hidden_states):
        return self.logit_fc(hidden_states)


class LxmertVisualObjHead(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.transform = LxmertPredictionHeadTransform(config)
        # Decide the use of visual losses
        visual_losses = {}
        if config.visual_obj_loss:
            visual_losses["obj"] = {"shape": (-1,), "num": config.num_object_labels}
        if config.visual_attr_loss:
            visual_losses["attr"] = {"shape": (-1,), "num": config.num_attr_labels}
        if config.visual_feat_loss:
            visual_losses["feat"] = {
                "shape": (-1, config.visual_feat_dim),
                "num": config.visual_feat_dim,
            }
        self.visual_losses = visual_losses

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder_dict = nn.CellDict(
            {key: mint.nn.Linear(config.hidden_size, self.visual_losses[key]["num"]) for key in self.visual_losses}
        )

    def construct(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        output = {}
        for key in self.visual_losses:
            output[key] = self.decoder_dict[key](hidden_states)
        return output


class LxmertPreTrainingHeads(nn.Cell):
    def __init__(self, config, lxmert_model_embedding_weights):
        super(LxmertPreTrainingHeads, self).__init__()
        self.predictions = LxmertLMPredictionHead(config, lxmert_model_embedding_weights)
        self.seq_relationship = mint.nn.Linear(config.hidden_size, 2)

    def construct(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class LxmertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LxmertConfig
    base_model_prefix = "lxmert"
    _supports_param_buffer_assignment = False

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, mint.nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/mindspore/mindspore/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, mint.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, mint.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class LxmertModel(LxmertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = LxmertEmbeddings(config)
        self.encoder = LxmertEncoder(config)
        self.pooler = LxmertPooler(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        visual_feats: Optional[ms.Tensor] = None,
        visual_pos: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        visual_attention_mask: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[LxmertModelOutput, Tuple[ms.Tensor]]:
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

        if visual_feats is None:
            raise ValueError("`visual_feats` cannot be `None`")
        if visual_pos is None:
            raise ValueError("`visual_pos` cannot be `None`")

        if attention_mask is None:
            attention_mask = mint.ones(input_shape)
        if token_type_ids is None:
            token_type_ids = mint.zeros(input_shape, dtype=ms.int64)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * dtype_to_min(self.dtype)

        # Process the visual attention mask
        if visual_attention_mask is not None:
            extended_visual_attention_mask = visual_attention_mask.unsqueeze(1).unsqueeze(2)
            extended_visual_attention_mask = extended_visual_attention_mask.to(dtype=self.dtype)
            extended_visual_attention_mask = (1.0 - extended_visual_attention_mask) * dtype_to_min(self.dtype)
        else:
            extended_visual_attention_mask = None

        # Positional Word Embeddings
        embedding_output = self.embeddings(input_ids, token_type_ids, inputs_embeds)

        # Run Lxmert encoder
        encoder_outputs = self.encoder(
            embedding_output,
            extended_attention_mask,
            visual_feats=visual_feats,
            visual_pos=visual_pos,
            visual_attention_mask=extended_visual_attention_mask,
            output_attentions=output_attentions,
        )

        visual_encoder_outputs, lang_encoder_outputs = encoder_outputs[:2]
        vision_hidden_states = visual_encoder_outputs[0]
        language_hidden_states = lang_encoder_outputs[0]

        all_attentions = ()
        if output_attentions:
            language_attentions = lang_encoder_outputs[1]
            vision_attentions = visual_encoder_outputs[1]
            cross_encoder_attentions = encoder_outputs[2]
            all_attentions = (
                language_attentions,
                vision_attentions,
                cross_encoder_attentions,
            )

        hidden_states = (language_hidden_states, vision_hidden_states) if output_hidden_states else ()

        visual_output = vision_hidden_states[-1]
        lang_output = language_hidden_states[-1]
        pooled_output = self.pooler(lang_output)

        if not return_dict:
            return (lang_output, visual_output, pooled_output) + hidden_states + all_attentions

        return LxmertModelOutput(
            pooled_output=pooled_output,
            language_output=lang_output,
            vision_output=visual_output,
            language_hidden_states=language_hidden_states if output_hidden_states else None,
            vision_hidden_states=vision_hidden_states if output_hidden_states else None,
            language_attentions=language_attentions if output_attentions else None,
            vision_attentions=vision_attentions if output_attentions else None,
            cross_encoder_attentions=cross_encoder_attentions if output_attentions else None,
        )


class LxmertForPreTraining(LxmertPreTrainedModel):
    _tied_weights_keys = ["cls.predictions.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)
        # Configuration
        self.config = config
        self.num_qa_labels = config.num_qa_labels
        self.visual_loss_normalizer = config.visual_loss_normalizer

        # Use of pretraining tasks
        self.task_mask_lm = config.task_mask_lm
        self.task_obj_predict = config.task_obj_predict
        self.task_matched = config.task_matched
        self.task_qa = config.task_qa

        # Lxmert backbone
        self.lxmert = LxmertModel(config)

        # Pre-training heads
        self.cls = LxmertPreTrainingHeads(config, self.lxmert.embeddings.word_embeddings.weight)
        if self.task_obj_predict:
            self.obj_predict_head = LxmertVisualObjHead(config)
        if self.task_qa:
            self.answer_head = LxmertVisualAnswerHead(config, self.num_qa_labels)

        # Weight initialization
        # Initialize weights and apply final processing
        self.post_init()

        # Loss functions
        self.loss_fcts = {
            "l2": SmoothL1Loss(reduction="none"),
            "visual_ce": CrossEntropyLoss(reduction="none"),
            "ce": CrossEntropyLoss(),
        }

        visual_losses = {}
        if config.visual_obj_loss:
            visual_losses["obj"] = {
                "shape": (-1,),
                "num": config.num_object_labels,
                "loss": "visual_ce",
            }
        if config.visual_attr_loss:
            visual_losses["attr"] = {
                "shape": (-1,),
                "num": config.num_attr_labels,
                "loss": "visual_ce",
            }
        if config.visual_feat_loss:
            visual_losses["feat"] = {
                "shape": (-1, config.visual_feat_dim),
                "num": config.visual_feat_dim,
                "loss": "l2",
            }
        self.visual_losses = visual_losses

    def resize_token_embeddings(
        self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = None, mean_resizing: bool = True
    ) -> mint.nn.Embedding:
        # Adding the following steps to resize bias to match the shape of resized embeddings
        new_embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)
        self.cls.predictions.bias = self._resize_bias(self.cls.predictions.bias, new_num_tokens)
        return new_embeddings

    def _resize_bias(self, bias, new_num_tokens: int):
        old_num_tokens = bias.shape[0]
        if new_num_tokens <= old_num_tokens:
            new_bias = bias[:new_num_tokens]
        else:
            extra_bias = mint.zeros(new_num_tokens - old_num_tokens)
            new_bias = mint.cat([bias, extra_bias])
        new_bias = ms.Parameter(new_bias)
        return new_bias

    def resize_num_qa_labels(self, num_labels):
        """
        Build a resized question answering linear layer Module from a provided new linear layer. Increasing the size
        will add newly initialized weights. Reducing the size will remove weights from the end

        Args:
            num_labels (`int`, *optional*):
                New number of labels in the linear layer weight matrix. Increasing the size will add newly initialized
                weights at the end. Reducing the size will remove weights from the end. If not provided or `None`, just
                returns a pointer to the qa labels ``mint.mint.nn.Linear``` module of the model without doing anything.

        Return:
            `mint.mint.nn.Linear`: Pointer to the resized Linear layer or the old Linear layer
        """

        cur_qa_logit_layer = self.get_qa_logit_layer()
        if num_labels is None or cur_qa_logit_layer is None:
            return
        new_qa_logit_layer = self._resize_qa_labels(num_labels)
        self.config.num_qa_labels = num_labels
        self.num_qa_labels = num_labels

        return new_qa_logit_layer

    def _resize_qa_labels(self, num_labels):
        cur_qa_logit_layer = self.get_qa_logit_layer()
        new_qa_logit_layer = self._get_resized_qa_labels(cur_qa_logit_layer, num_labels)
        self._set_qa_logit_layer(new_qa_logit_layer)
        return self.get_qa_logit_layer()

    def get_qa_logit_layer(self) -> nn.Cell:
        """
        Returns the linear layer that produces question answering logits.

        Returns:
            `nn.Cell`: A mindspore module mapping the question answering prediction hidden states or `None` if LXMERT
            does not have a visual answering head.
        """
        if hasattr(self, "answer_head"):
            return self.answer_head.logit_fc[-1]

    def _set_qa_logit_layer(self, qa_logit_layer):
        self.answer_head.logit_fc[-1] = qa_logit_layer

    def _get_resized_qa_labels(self, cur_qa_logit_layer, num_labels):
        if num_labels is None:
            return cur_qa_logit_layer

        cur_qa_labels, hidden_dim = cur_qa_logit_layer.weight.shape
        if cur_qa_labels == num_labels:
            return cur_qa_logit_layer

        # Build new linear output
        if getattr(cur_qa_logit_layer, "bias", None) is not None:
            new_qa_logit_layer = mint.nn.Linear(hidden_dim, num_labels)
        else:
            new_qa_logit_layer = mint.nn.Linear(hidden_dim, num_labels, bias=False)

        # initialize all new labels
        self._init_weights(new_qa_logit_layer)

        # Copy labels from the previous weights
        num_labels_to_copy = min(cur_qa_labels, num_labels)
        new_qa_logit_layer.weight.data[:num_labels_to_copy, :] = cur_qa_logit_layer.weight.data[:num_labels_to_copy, :]
        if getattr(cur_qa_logit_layer, "bias", None) is not None:
            new_qa_logit_layer.bias.data[:num_labels_to_copy] = cur_qa_logit_layer.bias.data[:num_labels_to_copy]

        return new_qa_logit_layer

    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        visual_feats: Optional[ms.Tensor] = None,
        visual_pos: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        visual_attention_mask: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        labels: Optional[ms.Tensor] = None,
        obj_labels: Optional[Dict[str, Tuple[ms.Tensor, ms.Tensor]]] = None,
        matched_label: Optional[ms.Tensor] = None,
        ans: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[LxmertForPreTrainingOutput, Tuple[ms.Tensor]]:
        r"""
        labels (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        obj_labels (`Dict[Str: Tuple[ms.Tensor, ms.Tensor]]`, *optional*):
            each key is named after each one of the visual losses and each element of the tuple is of the shape
            `(batch_size, num_features)` and `(batch_size, num_features, visual_feature_dim)` for each the label id and
            the label score respectively
        matched_label (`ms.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the whether or not the text input matches the image (classification) loss. Input
            should be a sequence pair (see `input_ids` docstring) Indices should be in `[0, 1]`:

            - 0 indicates that the sentence does not match the image,
            - 1 indicates that the sentence does match the image.
        ans (`ms.Tensor` of shape `(batch_size)`, *optional*):
            a one hot representation hof the correct answer *optional*

        Returns:
        """

        if "masked_lm_labels" in kwargs:
            warnings.warn(
                "The `masked_lm_labels` argument is deprecated and will be removed in a future version, use `labels`"
                " instead.",
                FutureWarning,
            )
            labels = kwargs.pop("masked_lm_labels")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        lxmert_output = self.lxmert(
            input_ids=input_ids,
            visual_feats=visual_feats,
            visual_pos=visual_pos,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            visual_attention_mask=visual_attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        lang_output, visual_output, pooled_output = (
            lxmert_output[0],
            lxmert_output[1],
            lxmert_output[2],
        )
        lang_prediction_scores, cross_relationship_score = self.cls(lang_output, pooled_output)
        if self.task_qa:
            answer_score = self.answer_head(pooled_output)
        else:
            answer_score = pooled_output[0][0]

        total_loss = (
            None
            if (labels is None and matched_label is None and obj_labels is None and ans is None)
            else ms.tensor(0.0)
        )
        if labels is not None and self.task_mask_lm:
            masked_lm_loss = self.loss_fcts["ce"](
                lang_prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1),
            )
            total_loss += masked_lm_loss
        if matched_label is not None and self.task_matched:
            matched_loss = self.loss_fcts["ce"](cross_relationship_score.view(-1, 2), matched_label.view(-1))
            total_loss += matched_loss
        if obj_labels is not None and self.task_obj_predict:
            total_visual_loss = ms.tensor(0.0)
            visual_prediction_scores_dict = self.obj_predict_head(visual_output)
            for key, key_info in self.visual_losses.items():
                label, mask_conf = obj_labels[key]
                output_dim = key_info["num"]
                loss_fct_name = key_info["loss"]
                label_shape = key_info["shape"]
                weight = self.visual_loss_normalizer
                visual_loss_fct = self.loss_fcts[loss_fct_name]
                visual_prediction_scores = visual_prediction_scores_dict[key]
                visual_loss = visual_loss_fct(
                    visual_prediction_scores.view(-1, output_dim),
                    label.view(label_shape),
                )
                if visual_loss.dim() > 1:  # Regression Losses
                    visual_loss = visual_loss.mean(1)
                visual_loss = (visual_loss * mask_conf.view(-1)).mean() * weight
                total_visual_loss += visual_loss
            total_loss += total_visual_loss
        if ans is not None and self.task_qa:
            answer_loss = self.loss_fcts["ce"](answer_score.view(-1, self.num_qa_labels), ans.view(-1))
            total_loss += answer_loss

        if not return_dict:
            output = (
                lang_prediction_scores,
                cross_relationship_score,
                answer_score,
            ) + lxmert_output[3:]
            return ((total_loss,) + output) if total_loss is not None else output

        return LxmertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=lang_prediction_scores,
            cross_relationship_score=cross_relationship_score,
            question_answering_score=answer_score,
            language_hidden_states=lxmert_output.language_hidden_states,
            vision_hidden_states=lxmert_output.vision_hidden_states,
            language_attentions=lxmert_output.language_attentions,
            vision_attentions=lxmert_output.vision_attentions,
            cross_encoder_attentions=lxmert_output.cross_encoder_attentions,
        )


class LxmertForQuestionAnswering(LxmertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # Configuration
        self.config = config
        self.num_qa_labels = config.num_qa_labels
        self.visual_loss_normalizer = config.visual_loss_normalizer

        # Lxmert backbone
        self.lxmert = LxmertModel(config)

        self.answer_head = LxmertVisualAnswerHead(config, self.num_qa_labels)

        # Weight initialization
        # Initialize weights and apply final processing
        self.post_init()

        # Loss function
        self.loss = CrossEntropyLoss()

    def resize_num_qa_labels(self, num_labels):
        """
        Build a resized question answering linear layer Module from a provided new linear layer. Increasing the size
        will add newly initialized weights. Reducing the size will remove weights from the end

        Args:
            num_labels (`int`, *optional*):
                New number of labels in the linear layer weight matrix. Increasing the size will add newly initialized
                weights at the end. Reducing the size will remove weights from the end. If not provided or `None`, just
                returns a pointer to the qa labels ``mint.mint.nn.Linear``` module of the model without doing anything.

        Return:
            `mint.mint.nn.Linear`: Pointer to the resized Linear layer or the old Linear layer
        """

        cur_qa_logit_layer = self.get_qa_logit_layer()
        if num_labels is None or cur_qa_logit_layer is None:
            return
        new_qa_logit_layer = self._resize_qa_labels(num_labels)
        self.config.num_qa_labels = num_labels
        self.num_qa_labels = num_labels

        return new_qa_logit_layer

    def _resize_qa_labels(self, num_labels):
        cur_qa_logit_layer = self.get_qa_logit_layer()
        new_qa_logit_layer = self._get_resized_qa_labels(cur_qa_logit_layer, num_labels)
        self._set_qa_logit_layer(new_qa_logit_layer)
        return self.get_qa_logit_layer()

    def get_qa_logit_layer(self) -> nn.Cell:
        """
        Returns the linear layer that produces question answering logits

        Returns:
            `nn.Cell`: A mindspore module mapping the question answering prediction hidden states. `None`: A NoneType
            object if Lxmert does not have the visual answering head.
        """

        if hasattr(self, "answer_head"):
            return self.answer_head.logit_fc[-1]

    def _set_qa_logit_layer(self, qa_logit_layer):
        self.answer_head.logit_fc[-1] = qa_logit_layer

    def _get_resized_qa_labels(self, cur_qa_logit_layer, num_labels):
        if num_labels is None:
            return cur_qa_logit_layer

        cur_qa_labels, hidden_dim = cur_qa_logit_layer.weight.shape
        if cur_qa_labels == num_labels:
            return cur_qa_logit_layer

        # Build new linear output
        if getattr(cur_qa_logit_layer, "bias", None) is not None:
            new_qa_logit_layer = mint.nn.Linear(hidden_dim, num_labels)
        else:
            new_qa_logit_layer = mint.nn.Linear(hidden_dim, num_labels, bias=False)

        # initialize all new labels
        self._init_weights(new_qa_logit_layer)

        # Copy labels from the previous weights
        num_labels_to_copy = min(cur_qa_labels, num_labels)
        new_qa_logit_layer.weight.data[:num_labels_to_copy, :] = cur_qa_logit_layer.weight.data[:num_labels_to_copy, :]
        if getattr(cur_qa_logit_layer, "bias", None) is not None:
            new_qa_logit_layer.bias.data[:num_labels_to_copy] = cur_qa_logit_layer.bias.data[:num_labels_to_copy]

        return new_qa_logit_layer

    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        visual_feats: Optional[ms.Tensor] = None,
        visual_pos: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        visual_attention_mask: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        labels: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[LxmertForQuestionAnsweringOutput, Tuple[ms.Tensor]]:
        r"""
        labels (`ms.Tensor` of shape `(batch_size)`, *optional*):
            A one-hot representation of the correct answer

        Example:

        ```python
        >>> from transformers import AutoTokenizer
        >>> from mindone.transformers import LxmertForQuestionAnswering
        >>> import mindspore as ms

        >>> tokenizer = AutoTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased", revision="refs/pr/3")
        >>> model = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-base-uncased", revision="refs/pr/3")

        >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

        >>> inputs = tokenizer(question, text, return_tensors="np")
        >>> for k, v in inputs.items():
        ...     inputs[k] = ms.tensor(v)

        >>> outputs = model(**inputs)

        >>> answer_start_index = outputs.start_logits.argmax()
        >>> answer_end_index = outputs.end_logits.argmax()

        >>> predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
        >>> tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)

        >>> # target is "nice puppet"
        >>> target_start_index = ms.tensor([14])
        >>> target_end_index = ms.tensor([15])

        >>> outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
        >>> loss = outputs.loss
        >>> round(loss.item(), 2)
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        lxmert_output = self.lxmert(
            input_ids=input_ids,
            visual_feats=visual_feats,
            visual_pos=visual_pos,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            visual_attention_mask=visual_attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        pooled_output = lxmert_output[2]
        answer_score = self.answer_head(pooled_output)
        loss = None
        if labels is not None:
            loss = self.loss(answer_score.view(-1, self.num_qa_labels), labels.view(-1))

        if not return_dict:
            output = (answer_score,) + lxmert_output[3:]
            return (loss,) + output if loss is not None else output

        return LxmertForQuestionAnsweringOutput(
            loss=loss,
            question_answering_score=answer_score,
            language_hidden_states=lxmert_output.language_hidden_states,
            vision_hidden_states=lxmert_output.vision_hidden_states,
            language_attentions=lxmert_output.language_attentions,
            vision_attentions=lxmert_output.vision_attentions,
            cross_encoder_attentions=lxmert_output.cross_encoder_attentions,
        )


__all__ = [
    "LxmertEncoder",
    "LxmertForPreTraining",
    "LxmertForQuestionAnswering",
    "LxmertModel",
    "LxmertPreTrainedModel",
    "LxmertVisualFeatureEncoder",
    "LxmertXLayer",
]
