# coding=utf-8
# Copyright 2021 The HuggingFace Team The HuggingFace Inc. team. All rights reserved.
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
"""MindSpore RemBERT model."""

from typing import Optional, Tuple, Union

from transformers.models.rembert.configuration_rembert import RemBertConfig
from transformers.utils import logging

import mindspore as ms
from mindspore import mint, nn
from mindspore.common.initializer import Normal, One, Zero, initializer

from ...activations import ACT2FN
from ...generation import GenerationMixin
from ...mindspore_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import MSPreTrainedModel

logger = logging.get_logger(__name__)


class RemBertEmbeddings(nn.Cell):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.input_embedding_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.input_embedding_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.input_embedding_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = mint.nn.LayerNorm(config.input_embedding_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_ids = mint.arange(config.max_position_embeddings).broadcast_to((1, -1))

    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        past_key_values_length: int = 0,
    ) -> ms.Tensor:
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if token_type_ids is None:
            token_type_ids = mint.zeros(input_shape, dtype=ms.int32)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


# Copied from transformers.models.bert.modeling_bert.BertPooler with Bert->RemBert
class RemBertPooler(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.dense = mint.nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class RemBertSelfAttention(nn.Cell):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = mint.nn.Linear(config.hidden_size, self.all_head_size)
        self.key = mint.nn.Linear(config.hidden_size, self.all_head_size)
        self.value = mint.nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(p=config.attention_probs_dropout_prob)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: ms.Tensor) -> ms.Tensor:
        new_x_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        encoder_attention_mask: Optional[ms.Tensor] = None,
        past_key_value: Tuple[Tuple[ms.Tensor]] = None,
        output_attentions: bool = False,
    ) -> Tuple:
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
            # if cross_attention save Tuple(ms.Tensor, ms.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(ms.Tensor, ms.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = mint.matmul(query_layer, key_layer.swapaxes(-1, -2))

        attention_scores = attention_scores / mint.sqrt(
            ms.tensor(self.attention_head_size, dtype=attention_scores.dtype)
        )
        if attention_mask is not None:
            # Apply the attention mask is (precomput
            # ed for all layers in RemBertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = mint.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = mint.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3)
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput with Bert->RemBert
class RemBertSelfOutput(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.dense = mint.nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = mint.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states: ms.Tensor, input_tensor: ms.Tensor) -> ms.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class RemBertAttention(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.self = RemBertSelfAttention(config)
        self.output = RemBertSelfOutput(config)
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


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->RemBert
class RemBertIntermediate(nn.Cell):
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


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->RemBert
class RemBertOutput(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.dense = mint.nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = mint.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states: ms.Tensor, input_tensor: ms.Tensor) -> ms.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class RemBertLayer(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = RemBertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = RemBertAttention(config)
        self.intermediate = RemBertIntermediate(config)
        self.output = RemBertOutput(config)

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

        layer_output = self.feed_forward_chunk(attention_output)
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


class RemBertEncoder(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embedding_hidden_mapping_in = mint.nn.Linear(config.input_embedding_size, config.hidden_size)
        self.layer = nn.CellList([RemBertLayer(config) for _ in range(config.num_hidden_layers)])
        self.add_cross_attention = config.add_cross_attention
        self.gradient_checkpointing = False

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        encoder_attention_mask: Optional[ms.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[ms.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                raise NotImplementedError("Gradient checkpoint is not yet supported.")
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


# Copied from transformers.models.bert.modeling_bert.BertPredictionHeadTransform with Bert->RemBert
class RemBertPredictionHeadTransform(nn.Cell):
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


class RemBertLMPredictionHead(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.dense = mint.nn.Linear(config.hidden_size, config.output_embedding_size)
        self.decoder = nn.Linear(config.output_embedding_size, config.vocab_size)
        self.activation = ACT2FN[config.hidden_act]
        self.LayerNorm = mint.nn.LayerNorm(config.output_embedding_size, eps=config.layer_norm_eps)

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOnlyMLMHead with Bert->RemBert
class RemBertOnlyMLMHead(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.predictions = RemBertLMPredictionHead(config)

    def construct(self, sequence_output: ms.Tensor) -> ms.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class RemBertPreTrainedModel(MSPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = RemBertConfig
    base_model_prefix = "rembert"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, mint.nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.set_data(
                initializer(
                    Normal(sigma=self.config.initializer_range, mean=0.0), module.weight.shape, module.weight.dtype
                )
            )
            if module.bias is not None:
                module.bias.set_data(initializer(Zero(), module.bias.shape, module.bias.dtype))
        elif isinstance(module, nn.Embedding):
            module.embedding_table.set_data(
                initializer(
                    Normal(sigma=self.config.initializer_range, mean=0.0),
                    module.embedding_table.shape,
                    module.embedding_table.dtype,
                )
            )
            if module.padding_idx is not None:
                module.embedding_table[module.padding_idx] = 0
        elif isinstance(module, mint.nn.LayerNorm):
            module.bias.set_data(initializer(Zero(), module.bias.shape, module.bias.dtype))
            module.weight.set_data(initializer(One(), module.weight.shape, module.weight.dtype))


class RemBertModel(RemBertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    _no_split_modules = ["BertEmbeddings", "BertLayer"]

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = RemBertEmbeddings(config)
        self.encoder = RemBertEncoder(config)

        self.pooler = RemBertPooler(config) if add_pooling_layer else None

        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.use_return_dict = config.use_return_dict
        self.is_decoder = config.is_decoder
        self.use_cache = config.use_cache
        self.num_hidden_layers = config.num_hidden_layers

        self.token_type_ids = self.embeddings.token_type_ids if hasattr(self.embeddings, "token_type_ids") else None

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
        input_ids: ms.Tensor = None,
        attention_mask: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        encoder_attention_mask: Optional[ms.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[ms.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = False,
    ) -> Union[Tuple, BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`ms.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(ms.Tensor))` of length `config.n_layers` with each tuple having 4 tensors of shape
            `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        if self.is_decoder:
            use_cache = use_cache if use_cache is not None else self.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = mint.ones((batch_size, seq_length + past_key_values_length))
        if token_type_ids is None:
            token_type_ids = mint.zeros(input_shape, dtype=ms.int32)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
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
        head_mask = self.get_head_mask(head_mask, self.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class RemBertForMaskedLM(RemBertPreTrainedModel):
    _tied_weights_keys = ["cls.predictions.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `RemBertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.rembert = RemBertModel(config, add_pooling_layer=False)
        self.cls = RemBertOnlyMLMHead(config)
        self.use_return_dict = config.use_return_dict
        self.vocab_size = config.vocab_size
        self.pad_token_id = config.pad_token_id

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def construct(
        self,
        input_ids: ms.Tensor = None,
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
        return_dict: Optional[bool] = False,
    ) -> Union[Tuple, MaskedLMOutput]:
        r"""
        labels (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        outputs = self.rembert(
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
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.vocab_size), labels.view(-1).int())

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        assert self.pad_token_id is not None, "The PAD token should be defined for generation"

        attention_mask = mint.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = mint.full((effective_batch_size, 1), self.pad_token_id, dtype=ms.int32)
        input_ids = mint.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}


class RemBertForCausalLM(RemBertPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["cls.predictions.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)

        if not config.is_decoder:
            logger.warning("If you want to use `RemBertForCausalLM` as a standalone, add `is_decoder=True.`")

        self.rembert = RemBertModel(config, add_pooling_layer=False)
        self.cls = RemBertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

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
        past_key_values: Optional[Tuple[Tuple[ms.Tensor]]] = None,
        labels: Optional[ms.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, RemBertConfig
        >>> from mindone.transformers import RemBertForCausalLM
        >>> import mindspore as ms

        >>> tokenizer = AutoTokenizer.from_pretrained("google/rembert")
        >>> config = RemBertConfig.from_pretrained("google/rembert")
        >>> config.is_decoder = True
        >>> model = RemBertForCausalLM.from_pretrained("google/rembert", config=config)

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="np")
        >>> outputs = model(**inputs)

        >>> prediction_logits = outputs.logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.rembert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        lm_loss = None
        if labels is not None:
            lm_loss = self.loss_function(
                prediction_scores,
                labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past


class RemBertForSequenceClassification(RemBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.problem_type = config.problem_type

        self.rembert = RemBertModel(config)
        self.dropout = nn.Dropout(p=config.classifier_dropout_prob)
        self.classifier = mint.nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: ms.Tensor = None,
        attention_mask: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        labels: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = False,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`ms.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        outputs = self.rembert(
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
            if self.problem_type is None:
                if self.num_labels == 1:
                    problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == ms.int64 or labels.dtype == ms.int32):
                    problem_type = "single_label_classification"
                else:
                    problem_type = "multi_label_classification"
            else:
                problem_type = self.problem_type

            if problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1).int())
            elif problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
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


class RemBertForMultipleChoice(RemBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.rembert = RemBertModel(config)
        self.dropout = nn.Dropout(p=config.classifier_dropout_prob)
        self.classifier = mint.nn.Linear(config.hidden_size, 1)
        self.use_return_dict = config.use_return_dict

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: ms.Tensor = None,
        attention_mask: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        labels: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = False,
    ) -> Union[Tuple, MultipleChoiceModelOutput]:
        r"""
        labels (`ms.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.shape[-1]) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1]) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1]) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.shape[-1]) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.shape[-2], inputs_embeds.shape[-1])
            if inputs_embeds is not None
            else None
        )

        outputs = self.rembert(
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
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels.int())

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RemBertForTokenClassification(RemBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.rembert = RemBertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(p=config.classifier_dropout_prob)
        self.classifier = mint.nn.Linear(config.hidden_size, config.num_labels)
        self.use_return_dict = config.use_return_dict

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: ms.Tensor = None,
        attention_mask: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        labels: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = False,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.rembert(
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

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1).int())

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RemBertForQuestionAnswering(RemBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels

        self.rembert = RemBertModel(config, add_pooling_layer=False)
        self.qa_outputs = mint.nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: ms.Tensor = None,
        attention_mask: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        start_positions: Optional[ms.Tensor] = None,
        end_positions: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = False,
    ) -> Union[Tuple[ms.Tensor], QuestionAnsweringModelOutput]:
        r"""
        start_positions (`ms.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`ms.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.rembert(
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

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, axis=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.shape) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.shape) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.shape[1]
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions.int())
            end_loss = loss_fct(end_logits, end_positions.int())
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
