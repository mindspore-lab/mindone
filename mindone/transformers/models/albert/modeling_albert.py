# coding=utf-8
# Copyright 2018 Google AI, Google Brain and the HuggingFace Inc. team.
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
"""MindSpore ALBERT model."""

import numbers
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from transformers.models.albert.configuration_albert import AlbertConfig
from transformers.utils import ModelOutput, logging

import mindspore as ms
from mindspore import nn, ops
from mindspore.common.initializer import Normal, One, Zero, initializer

from ...activations import ACT2FN
from ...modeling_attn_mask_utils import dtype_to_min
from ...mindspore_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import MSPreTrainedModel

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "albert/albert-base-v2"
_CONFIG_FOR_DOC = "AlbertConfig"

class AlbertEmbeddings(nn.Cell):
    """
    Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config: AlbertConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm((config.embedding_size,), epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.position_ids = ops.arange(config.max_position_embeddings).broadcast_to((1, -1))
        self.token_type_ids = ops.zeros(self.position_ids.shape, dtype=ms.int32)

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

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.broadcast_to((input_shape[0], seq_length))
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = ops.zeros(input_shape, dtype=ms.int32)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids.int())
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class AlbertAttention(nn.Cell):
    def __init__(self, config: AlbertConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads}"
            )
    
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Dense(config.hidden_size, self.all_head_size)
        self.key = nn.Dense(config.hidden_size, self.all_head_size)
        self.value = nn.Dense(config.hidden_size, self.all_head_size)

        self.attention_dropout = nn.Dropout(p=config.attention_probs_dropout_prob)
        self.output_dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm((config.hidden_size,), epsilon=config.layer_norm_eps)
        self.pruned_heads = set()

        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

    # Copied from transformers.models.bert.modeling_bert.BertSelfAttention.transpose_for_scores
    def transpose_for_scores(self, x: ms.Tensor) -> ms.Tensor:
        new_x_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def prune_heads(self, heads) -> None:
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.num_attention_heads, self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.query = prune_linear_layer(self.query, index)
        self.key = prune_linear_layer(self.key, index)
        self.value = prune_linear_layer(self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.num_attention_heads = self.num_attention_heads - len(heads)
        self.all_head_size = self.attention_head_size * self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)
    
    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[ms.Tensor], Tuple[ms.Tensor, ms.Tensor]]:
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = ops.matmul(query_layer, key_layer.swapaxes(-1, -2))
        attention_scores = attention_scores / ops.sqrt(
            ms.tensor(self.attention_head_size, dtype=attention_scores.dtype)
        )

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask
        
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.shape[1]
            position_ids_l = ops.arange(seq_length, dtype=ms.int64).view((-1, 1))
            position_ids_r = ops.arange(seq_length, dtype=ms.int64).view((1, -1))
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = ops.matmul(
                    query_layer.unsqueeze(3), positional_embedding.permute(0, 2, 1)
                ).squeeze(
                    3
                )  # "bhld,lrd->bhlr"
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = ops.matmul(
                    query_layer.unsqueeze(3), positional_embedding.permute(0, 2, 1)
                ).squeeze(
                    3
                )  # "bhld,lrd->bhlr"
                relative_position_scores_key = (
                    key_layer[:, :, None, :, :] * positional_embedding[None, None, :, :, :]
                ).sum(
                    -1
                )  # "bhrd,lrd->bhlr"
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key
            
        # Normalize the attention scores to probabilities.
        attention_probs = ops.softmax(attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = ops.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).flatten(2)

        projected_context_layer = self.dense(context_layer)
        projected_context_layer_dropout = self.output_dropout(projected_context_layer)
        layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
        return (layernormed_context_layer, attention_probs) if output_attentions else (layernormed_context_layer,)
    


class AlbertSdpaAttention(AlbertAttention):
    def __init__(self, config):
        super().__init__(config)
        self.dropout_prob = config.attention_probs_dropout_prob
    
    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[ms.Tensor], Tuple[ms.Tensor, ms.Tensor]]:
        if self.position_embedding_type != "absolute" or output_attentions or head_mask is not None:
            logger.warning(
                "AlbertSdpaAttention is used but `scaled_dot_product_attention` does not support "
                "non-absolute `position_embedding_type` or `output_attentions=True` or `head_mask`. Falling back to "
                "the eager attention implementation, but specifying the eager implementation will be required from "
                "Transformers version v5.0.0 onwards. This warning can be removed using the argument "
                '`attn_implementation="eager"` when loading the model.'
            )
            return super().construct(hidden_states, attention_mask, head_mask, output_attentions)
    
        batch_size, seq_len, _ = hidden_states.shape
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_output = scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attention_mask,
        )
        attention_output = attention_output.swapaxes(1, 2)
        attention_output = attention_output.reshape(batch_size, seq_len, self.all_head_size)

        projected_context_layer = self.dense(attention_output)
        projected_context_layer_dropout = self.output_dropout(projected_context_layer)
        layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
        return (layernormed_context_layer,)
    
ALBERT_ATTENTION_CLASSES = {
    "eager": AlbertAttention,
    "sdpa": AlbertSdpaAttention,
}


class AlbertLayer(nn.Cell):
    def __init__(self, config: AlbertConfig):
        super().__init__()

        self.config = config
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.full_layer_layer_norm = nn.LayerNorm((config.hidden_size,), epsilon=config.layer_norm_eps)
        self.attention = ALBERT_ATTENTION_CLASSES[config._attn_implementation](config)
        self.ffn = nn.Dense(config.hidden_size, config.intermediate_size)
        self.ffn_output = nn.Dense(config.intermediate_size, config.hidden_size)
        self.activation = ACT2FN[config.hidden_act]
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
    
    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Tuple[ms.Tensor, ms.Tensor]:
        attention_output = self.attention(
            hidden_states, 
            attention_mask, 
            head_mask, 
            output_attentions=output_attentions,
        )
        
        ffn_output = self.ff_chunk(attention_output[0])
        hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])

        return (hidden_states,) + attention_output[1:]  # add attentions if we output them

    def ff_chunk(self, attention_output: ms.Tensor) -> ms.Tensor:
        ffn_output = self.ffn(attention_output)
        ffn_output = self.activation(ffn_output)
        ffn_output = self.ffn_output(ffn_output)
        return ffn_output


class AlbertLayerGroup(nn.Cell):
    def __init__(self, config: AlbertConfig):
        super().__init__()

        self.albert_layers = nn.CellList([AlbertLayer(config) for _ in range(config.inner_group_num)])
    
    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Tuple[Union[ms.Tensor, Tuple[ms.Tensor]], ...]:
        layer_hidden_states = ()
        layer_attentions = ()

        for layer_index, albert_layer in enumerate(self.albert_layers):
            layer_output = albert_layer(hidden_states, attention_mask, head_mask[layer_index], output_attentions)
            hidden_states = layer_output[0]

            if output_attentions:
                layer_attentions = layer_attentions + (layer_output[1],)

            if output_hidden_states:
                layer_hidden_states = layer_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if output_hidden_states:
            outputs = outputs + (layer_hidden_states,)
        if output_attentions:
            outputs = outputs + (layer_attentions,)
        return outputs  # last-layer hidden state, (layer hidden states), (layer attentions)
    

class AlbertTransformer(nn.Cell):
    def __init__(self, config: AlbertConfig):
        super().__init__()

        self.config = config
        self.embedding_hidden_mapping_in = nn.Dense(config.embedding_size, config.hidden_size)
        self.albert_layer_groups = nn.CellList([AlbertLayerGroup(config) for _ in range(config.num_hidden_groups)])

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[BaseModelOutput, Tuple]:
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)

        all_hidden_states = (hidden_states,) if output_hidden_states else None
        all_attentions = () if output_attentions else None

        head_mask = [None] * self.config.num_hidden_layers if head_mask is None else head_mask

        for i in range(self.config.num_hidden_layers):
            # Number of layers in a hidden group
            layers_per_group = int(self.config.num_hidden_layers / self.config.num_hidden_groups)

            # Index of the hidden group
            group_idx = int(i / (self.config.num_hidden_layers / self.config.num_hidden_groups))

            layer_group_output = self.albert_layer_groups[group_idx](
                hidden_states,
                attention_mask,
                head_mask[group_idx * layers_per_group : (group_idx + 1) * layers_per_group],
                output_attentions,
                output_hidden_states,
            )
            hidden_states = layer_group_output[0]

            if output_attentions:
                all_attentions = all_attentions + layer_group_output[-1]

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


class AlbertPreTrainedModel(MSPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = AlbertConfig
    base_model_prefix = "albert"
    supports_gradient_checkpointing = True
    _supports_sdpa = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Dense):
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
        elif isinstance(module, LayerNorm):
            module.bias.set_data(initializer(Zero(), module.bias.shape, module.bias.dtype))
            module.weight.set_data(initializer(One(), module.weight.shape, module.weight.dtype))


@dataclass
class AlbertForPreTrainingOutput(ModelOutput):
    """
    Output type of [`AlbertForPreTraining`].

    Args:
        loss (*optional*, returned when `labels` is provided, `ms.Tensor` of shape `(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (`ms.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        sop_logits (`ms.Tensor` of shape `(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (`tuple(ms.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `ms.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(ms.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `ms.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[ms.Tensor] = None
    prediction_logits: ms.Tensor = None
    sop_logits: ms.Tensor = None
    hidden_states: Optional[Tuple[ms.Tensor]] = None
    attentions: Optional[Tuple[ms.Tensor]] = None

class AlbertModel(AlbertPreTrainedModel):
    def __init__(self, config: AlbertConfig, add_pooling_layer = True):
        super().__init__(config)

        self.config = config
        self.embeddings = AlbertEmbeddings(config)
        self.encoder = AlbertTransformer(config)

        if add_pooling_layer:
            self.pooler = nn.Dense(config.hidden_size, config.hidden_size)
            self.pooler_activation = nn.Tanh()
        else:
            self.pooler = None
            self.pooler_activation = None
        if self.pooler is None:
            logger.warning(
                "Pooler is None, not returning `pooled_output`. "
                "Model output will be (`last_hidden_stat`, `past_key_values`, `hidden_states`, `attentions`, `cross_attentions`)"
            )

        self.attn_implementation = config._attn_implementation
        self.position_embedding_type = config.position_embedding_type

        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.use_return_dict = config.use_return_dict
        self.num_hidden_layers = config.num_hidden_layers

        self.token_type_ids = self.embeddings.token_type_ids if hasattr(self.embeddings, "token_type_ids") else None

        # Initialize weights and apply final processing
        self.post_init()
    
    def get_input_embeddings(self) -> nn.Embedding:
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value) -> None:
        self.embeddings.word_embeddings = value
    
    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} ALBERT has
        a different architecture in that its layers are shared across groups, which then has inner groups. If an ALBERT
        model has 12 hidden layers and 2 hidden groups, with two inner groups, there is a total of 4 different layers.

        These layers are flattened: the indices [0,1] correspond to the two inner groups of the first hidden layer,
        while [2,3] correspond to the two inner groups of the second hidden layer.

        Any layer with in index other than [0,1,2,3] will result in an error. See base class PreTrainedModel for more
        information about head pruning
        """
        for layer, heads in heads_to_prune.items():
            group_idx = int(layer / self.config.inner_group_num)
            inner_group_idx = int(layer - group_idx * self.config.inner_group_num)
            self.encoder.albert_layer_groups[group_idx].albert_layers[inner_group_idx].attention.prune_heads(heads)

    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = False,
    ) -> Union[BaseModelOutputWithPooling, Tuple]:
        output_attentions = output_attentions if output_attentions is not None else self.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        if attention_mask is None:
            attention_mask = ops.ones(input_shape)
        if token_type_ids is None:
            if self.token_type_ids is not None:
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.broadcast_to((batch_size, seq_length))
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = ops.zeros(input_shape, dtype=ms.int32)
        
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        use_sdpa_attention_mask = (
            self.attn_implementation == "sdpa"
            and self.position_embedding_type == "absolute"
            and head_mask is None
            and not output_attentions
        )
        if use_sdpa_attention_mask:
            extended_attention_mask = _prepare_4d_attention_mask(
                attention_mask, embedding_output.dtype, tgt_len=seq_length
            )
        else:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * dtype_to_min(self.dtype)

        head_mask = self.get_head_mask(head_mask, self.num_hidden_layers)
        encoder_outputs = self.encoder(
            embedding_output,
            extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]

        pooled_output = self.pooler_activation(self.pooler(sequence_output[:, 0])) if self.pooler is not None else None

        if not return_dict:
            if pooled_output is not None:
                return (sequence_output, pooled_output) + encoder_outputs[1:]
            else:
                return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
    
class AlbertForPreTraining(AlbertPreTrainedModel):
    _tied_weights_keys = ["predictions.decoder.bias", "predictions.decoder.weight"]

    def __init__(self, config: AlbertConfig):
        super().__init__(config)

        self.albert = AlbertModel(config)
        self.predictions = AlbertMLMHead(config)
        self.sop_classifier = AlbertSOPHead(config)

        self.use_return_dict = config.use_return_dict
        self.vocab_size = config.vocab_size

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self) -> nn.Dense:
        return self.predictions.decoder

    def set_output_embeddings(self, new_embeddings: nn.Dense) -> None:
        self.predictions.decoder = new_embeddings

    def get_input_embeddings(self) -> nn.Embedding:
        return self.albert.embeddings.word_embeddings

    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        labels: Optional[ms.Tensor] = None,
        sentence_order_label: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = False,
    ) -> Union[AlbertForPreTrainingOutput, Tuple]:
        r"""
        labels (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        sentence_order_label (`ms.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
            (see `input_ids` docstring) Indices should be in `[0, 1]`. `0` indicates original order (sequence A, then
            sequence B), `1` indicates switched order (sequence B, then sequence A).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, AlbertForPreTraining
        >>> import mindspore as ms

        >>> tokenizer = AutoTokenizer.from_pretrained("albert/albert-base-v2")
        >>> model = AlbertForPreTraining.from_pretrained("albert/albert-base-v2")

        >>> input_ids = ms.Tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)
        >>> # Batch size 1
        >>> outputs = model(input_ids)

        >>> prediction_logits = outputs.prediction_logits
        >>> sop_logits = outputs.sop_logits
        ```
        """
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        outputs = self.albert(
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

        sequence_output, pooled_output = outputs[:2]

        prediction_scores = self.predictions(sequence_output)
        sop_scores = self.sop_classifier(pooled_output)

        total_loss = None
        if labels is not None and sentence_order_label is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.vocab_size), labels.view(-1).int())
            sentence_order_loss = loss_fct(sop_scores.view(-1, 2), sentence_order_label.view(-1).int())
            total_loss = masked_lm_loss + sentence_order_loss

        if not return_dict:
            output = (prediction_scores, sop_scores) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return AlbertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            sop_logits=sop_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



class AlbertMLMHead(nn.Cell):
    def __init__(self, config: AlbertConfig):
        super().__init__()

        self.LayerNorm = nn.LayerNorm((config.embedding_size,), epsilon=config.layer_norm_eps)
        self.bias = ms.Parameter(ops.zeros(config.vocab_size), name="bias")
        self.dense = nn.Dense(config.hidden_size, config.embedding_size)
        self.decoder = nn.Dense(config.embedding_size, config.vocab_size)
        self.activation = ACT2FN[config.hidden_act]
        self.decoder.bias = self.bias
    
    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states)

        prediction_scores = hidden_states

        return prediction_scores

    def _tie_weights(self) -> None:
        self.decoder.bias = self.bias


class AlbertSOPHead(nn.Cell):
    def __init__(self, config: AlbertConfig):
        super().__init__()

        self.dropout = nn.Dropout(p=config.classifier_dropout_prob)
        self.classifier = nn.Dense(config.hidden_size, config.num_labels)

    def construct(self, pooled_output: ms.Tensor) -> ms.Tensor:
        dropout_pooled_output = self.dropout(pooled_output)
        logits = self.classifier(dropout_pooled_output)
        return logits


class AlbertForMaskedLM(AlbertPreTrainedModel):
    _tied_weights_keys = ["predictions.decoder.bias", "predictions.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)

        self.albert = AlbertModel(config, add_pooling_layer=False)
        self.predictions = AlbertMLMHead(config)
        self.use_return_dict = config.use_return_dict
        self.vocab_size = config.vocab_size
        self.pad_token_id = config.pad_token_id

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self) -> nn.Dense:
        return self.predictions.decoder

    def set_output_embeddings(self, new_embeddings) -> None:
        self.predictions.decoder = new_embeddings
        self.predictions.bias = new_embeddings.bias
    
    def get_input_embeddings(self) -> nn.Embedding:
        return self.albert.embeddings.word_embeddings

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
        return_dict: Optional[bool] = False,
    ) -> Union[MaskedLMOutput, Tuple]:
        r"""
        labels (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        
        Returns:

        Example:

        ```python
        >>> import mindspore as ms
        >>> from transformers import AutoTokenizer, AlbertForMaskedLM

        >>> tokenizer = AutoTokenizer.from_pretrained("albert/albert-base-v2")
        >>> model = AlbertForMaskedLM.from_pretrained("albert/albert-base-v2")

        >>> # add mask_token
        >>> inputs = tokenizer("The capital of [MASK] is Paris.")
        >>> with ms.no_grad():
        ...     logits = model(**inputs).logits

        >>> # retrieve index of [MASK]
        >>> mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
        >>> predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
        >>> tokenizer.decode(predicted_token_id)
        'france'
        ```

        ```python
        >>> labels = tokenizer("The capital of France is Paris.")["input_ids"]
        >>> labels = ms.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)
        >>> outputs = model(**inputs, labels=labels)
        >>> round(outputs.loss.item(), 2)
        0.81
        ```
        """

        return_dict = return_dict if return_dict is not None else self.use_return_dict

        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_outputs = outputs[0]

        prediction_scores = self.predictions(sequence_outputs)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
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
    

class AlbertForSequenceClassification(AlbertPreTrainedModel):
    def __init__(self, config: AlbertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.use_return_dict = config.use_return_dict
        self.problem_type = config.problem_type

        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(p=config.classifier_dropout_prob)
        self.classifier = nn.Dense(config.hidden_size, config.num_labels)

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
        return_dict: Optional[bool] = False,
    ) -> Union[SequenceClassifierOutput, Tuple]:
        r"""
        labels (`ms.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        outputs = self.albert(
            input_ids=input_ids,
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


class AlbertForTokenClassification(AlbertPreTrainedModel):
    def __init__(self, config: AlbertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.albert = AlbertModel(config, add_pooling_layer=False)
        classifier_dropout_prob = (
            config.classifier_dropout_prob if config.classifier_dropout_prob is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(p=classifier_dropout_prob)
        self.classifier = nn.Dense(config.hidden_size, self.config.num_labels)
        self.use_return_dict = config.use_return_dict

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
        return_dict: Optional[bool] = False,
    ) -> Union[TokenClassifierOutput, Tuple]:
        r"""
        labels (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.albert(
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
    
class AlbertForQuestionAnswering(AlbertPreTrainedModel):
    def __init__(self, config: AlbertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.albert = AlbertModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Dense(config.hidden_size, config.num_labels)

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
        start_positions: Optional[ms.Tensor] = None,
        end_positions: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = False,
    ) -> Union[AlbertForPreTrainingOutput, Tuple]:
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

        outputs = self.albert(
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
    
class AlbertForMultipleChoice(AlbertPreTrainedModel):
    def __init__(self, config: AlbertConfig):
        super().__init__(config)

        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(p=config.classifier_dropout_prob)
        self.classifier = nn.Dense(config.hidden_size, 1)
        self.use_return_dict = config.use_return_dict

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
        return_dict: Optional[bool] = False,
    ) -> Union[AlbertForPreTrainingOutput, Tuple]:
        r"""
        labels (`ms.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where *num_choices* is the size of the second dimension of the input tensors. (see
            *input_ids* above)
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

        outputs = self.albert(
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


class LayerNorm(nn.Cell):
    r"""Applies Layer Normalization over a mini-batch of inputs.

    This layer implements the operation as described in
    the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated over the last `D` dimensions, where `D`
    is the dimension of :attr:`normalized_shape`. For example, if :attr:`normalized_shape`
    is ``(3, 5)`` (a 2-dimensional shape), the mean and standard-deviation are computed over
    the last 2 dimensions of the input (i.e. ``input.mean((-2, -1))``).
    :math:`\gamma` and :math:`\beta` are learnable affine transform parameters of
    :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.
    The standard-deviation is calculated via the biased estimator, equivalent to
    `ops.var(input, unbiased=False)`.

    .. note::
        Unlike Batch Normalization and Instance Normalization, which applies
        scalar scale and bias for each entire channel/plane with the
        :attr:`affine` option, Layer Normalization applies per-element scale and
        bias with :attr:`elementwise_affine`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        normalized_shape (int or list): input shape from an expected input
            of size

            .. math::
                [* \times \text{normalized\_shape}[0] \times \text{normalized\_shape}[1]
                    \times \ldots \times \text{normalized\_shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`\text{normalized\_shape}` when :attr:`elementwise_affine` is set to ``True``.
            The values are initialized to 1.
        bias:   the learnable bias of the module of shape
                :math:`\text{normalized\_shape}` when :attr:`elementwise_affine` is set to ``True``.
                The values are initialized to 0.

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)

    Examples::

        >>> # NLP Example
        >>> batch, sentence_length, embedding_dim = 20, 5, 10
        >>> embedding = ops.randn(batch, sentence_length, embedding_dim)
        >>> layer_norm = LayerNorm(embedding_dim)
        >>> # Activate module
        >>> layer_norm(embedding)
        >>>
        >>> # Image Example
        >>> N, C, H, W = 20, 5, 10, 10
        >>> input = ops.randn(N, C, H, W)
        >>> # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
        >>> # as shown in the image below
        >>> layer_norm = LayerNorm([C, H, W])
        >>> output = layer_norm(input)
    """

    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine: bool = True, bias=True, dtype=ms.float32):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        _weight = np.ones(normalized_shape, dtype=ms.dtype_to_nptype(dtype))
        _bias = np.zeros(normalized_shape, dtype=ms.dtype_to_nptype(dtype))
        if self.elementwise_affine:
            self.weight = ms.Parameter(ms.Tensor.from_numpy(_weight), name="weight")
            if bias:
                self.bias = ms.Parameter(ms.Tensor.from_numpy(_bias), name="bias")
            else:
                self.bias = ms.Tensor.from_numpy(_bias)
        else:
            self.weight = ms.Tensor.from_numpy(_weight)
            self.bias = ms.Tensor.from_numpy(_bias)
        # TODO: In fact, we need -len(normalized_shape) instead of -1, but LayerNorm doesn't allow it.
        #  For positive axis, the ndim of input is needed. Put it in construct?
        self.layer_norm = ops.LayerNorm(-1, -1, epsilon=eps)

    def construct(self, x: ms.Tensor):
        x, _, _ = self.layer_norm(x, self.weight, self.bias)
        return x


def scaled_dot_product_attention(query, key, value, attn_mask=None, dtype=None):
    # force fp16 precision calculation
    _dtype = query.dtype
    if dtype is not None:
        query, key, value = query.astype(dtype), key.astype(dtype), value.astype(dtype)

    if attn_mask is not None:
        attn_mask = attn_mask.masked_fill(not attn_mask, -1e5) if attn_mask.dtype == ms.bool_ else attn_mask
        attn_weight = ops.softmax(
            ops.cast(ops.matmul(query, key.swapaxes(-2, -1)) / (query.shape[-1] ** 0.5) + attn_mask, ms.float32),
            axis=-1,
        ).astype(_dtype)
    else:
        attn_weight = ops.softmax(
            ops.cast(ops.matmul(query, key.swapaxes(-2, -1)) / (query.shape[-1] ** 0.5), ms.float32), axis=-1
        ).astype(_dtype)

    out = ops.matmul(attn_weight, value)
    out = out.astype(_dtype)

    return out
