# coding=utf-8
# Copyright 2023-present NAVER Corp, The Microsoft Research Asia LayoutLM Team Authors and the HuggingFace Inc. team.
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
"""MindSpore Bros model."""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from transformers import BrosConfig
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

import mindspore as ms
from mindspore import mint
from mindspore.mint.nn import CrossEntropyLoss

from mindone.transformers.mindspore_adapter.utils import _DTYPE_2_MIN

from ...activations import ACT2FN
from ...mindspore_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "jinho8345/bros-base-uncased"
_CONFIG_FOR_DOC = "BrosConfig"


BROS_START_DOCSTRING = r"""
    This model is also a MindSpore [mindspore.nn.Cell](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Cell.html) subclass.
    Use it as a regular MindSpore Cell and refer to the MindSpore documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`BrosConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

BROS_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`mindspore.Tensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`BrosProcessor`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)

        bbox ('mindspore.Tensor' of shape '(batch_size, num_boxes, 4)'):
            Bounding box coordinates for each token in the input sequence. Each bounding box is a list of four values
            (x1, y1, x2, y2), where (x1, y1) is the top left corner, and (x2, y2) is the bottom right corner of the
            bounding box.

        attention_mask (`mindspore.Tensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

        bbox_first_token_mask (`mindspore.Tensor` of shape `({0})`, *optional*):
            Mask to indicate the first token of each bounding box. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

        token_type_ids (`mindspore.Tensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)

        position_ids (`mindspore.Tensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)

        head_mask (`mindspore.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`mindspore.Tensor` of shape `({0}, hidden_size)`, *optional*):
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
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""


@dataclass
class BrosSpadeOutput(ModelOutput):
    """
    Base class for outputs of token classification models.

    Args:
        loss (`mindspore.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided) :
            Classification loss.
        initial_token_logits (`mindspore.Tensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            Classification scores for entity initial tokens (before SoftMax).
        subsequent_token_logits (`mindspore.Tensor` of shape `(batch_size, sequence_length, sequence_length+1)`):
            Classification scores for entity sequence tokens (before SoftMax).
        hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[ms.Tensor] = None
    initial_token_logits: ms.Tensor = None
    subsequent_token_logits: ms.Tensor = None
    hidden_states: Optional[Tuple[ms.Tensor]] = None
    attentions: Optional[Tuple[ms.Tensor]] = None


class BrosPositionalEmbedding1D(ms.nn.Cell):
    # Reference: https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/mem_transformer.py#L15

    def __init__(self, config):
        super(BrosPositionalEmbedding1D, self).__init__()

        self.dim_bbox_sinusoid_emb_1d = config.dim_bbox_sinusoid_emb_1d

        inv_freq = 1 / (10000 ** (mint.arange(0.0, self.dim_bbox_sinusoid_emb_1d, 2.0) / self.dim_bbox_sinusoid_emb_1d))
        self.register_buffer("inv_freq", inv_freq)

    def construct(self, pos_seq: ms.Tensor) -> ms.Tensor:
        seq_size = pos_seq.shape
        b1, b2, b3 = seq_size
        sinusoid_inp = pos_seq.view(b1, b2, b3, 1) * self.inv_freq.view(1, 1, 1, self.dim_bbox_sinusoid_emb_1d // 2)
        pos_emb = mint.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_emb


class BrosPositionalEmbedding2D(ms.nn.Cell):
    def __init__(self, config):
        super(BrosPositionalEmbedding2D, self).__init__()

        self.dim_bbox = config.dim_bbox
        self.x_pos_emb = BrosPositionalEmbedding1D(config)
        self.y_pos_emb = BrosPositionalEmbedding1D(config)

    def construct(self, bbox: ms.Tensor) -> ms.Tensor:
        stack = []
        for i in range(self.dim_bbox):
            if i % 2 == 0:
                stack.append(self.x_pos_emb(bbox[..., i]))
            else:
                stack.append(self.y_pos_emb(bbox[..., i]))
        bbox_pos_emb = mint.cat(stack, dim=-1)
        return bbox_pos_emb


class BrosBboxEmbeddings(ms.nn.Cell):
    def __init__(self, config):
        super(BrosBboxEmbeddings, self).__init__()
        self.bbox_sinusoid_emb = BrosPositionalEmbedding2D(config)
        self.bbox_projection = mint.nn.Linear(config.dim_bbox_sinusoid_emb_2d, config.dim_bbox_projection, bias=False)

    def construct(self, bbox: ms.Tensor):
        bbox_t = bbox.transpose(0, 1)
        bbox_pos = bbox_t[None, :, :, :] - bbox_t[:, None, :, :]
        bbox_pos_emb = self.bbox_sinusoid_emb(bbox_pos).to(bbox_pos.dtype)
        bbox_pos_emb = self.bbox_projection(bbox_pos_emb)

        return bbox_pos_emb


class BrosTextEmbeddings(ms.nn.Cell):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()

        self.word_embeddings = mint.nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = mint.nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = mint.nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = mint.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = mint.nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids",
            mint.arange(config.max_position_embeddings).broadcast_to((1, config.max_position_embeddings)),
        )
        self.register_buffer(
            "token_type_ids",
            mint.zeros(
                self.position_ids.shape,
                dtype=ms.int64,
            ),
            persistent=False,
        )

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
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.broadcast_to((input_shape[0], seq_length))
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = mint.zeros(
                    input_shape,
                    dtype=ms.int64,
                )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BrosSelfAttention(ms.nn.Cell):
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

        self.dropout = mint.nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = mint.nn.Embedding(
                2 * config.max_position_embeddings - 1, self.attention_head_size
            )

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: ms.Tensor):
        new_x_shape = x.shape[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def construct(
        self,
        hidden_states: ms.Tensor,
        bbox_pos_emb: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        encoder_attention_mask: Optional[ms.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[ms.Tensor]]] = None,
        output_attentions: Optional[ms.Tensor] = False,
    ) -> Tuple[ms.Tensor]:
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
        attention_scores = mint.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.shape[1]
            position_ids_l = mint.arange(
                seq_length,
                dtype=ms.int64,
            ).view(-1, 1)
            position_ids_r = mint.arange(
                seq_length,
                dtype=ms.int64,
            ).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = mint.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = mint.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = mint.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)

                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        # bbox positional encoding
        batch_size, n_head, seq_length, d_head = query_layer.shape
        bbox_pos_emb = bbox_pos_emb.view(seq_length, seq_length, batch_size, d_head)
        bbox_pos_emb = bbox_pos_emb.permute([2, 0, 1, 3])
        bbox_pos_scores = mint.einsum("bnid,bijd->bnij", query_layer, bbox_pos_emb)

        attention_scores = attention_scores + bbox_pos_scores

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BrosModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = mint.nn.Softmax(dim=-1)(attention_scores)

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


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput with Bert->Bros
class BrosSelfOutput(ms.nn.Cell):
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


class BrosAttention(ms.nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.self = BrosSelfAttention(config)
        self.output = BrosSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads,
            self.self.num_attention_heads,
            self.self.attention_head_size,
            self.pruned_heads,
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
        bbox_pos_emb: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        encoder_attention_mask: Optional[ms.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[ms.Tensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[ms.Tensor]:
        self_outputs = self.self(
            hidden_states=hidden_states,
            bbox_pos_emb=bbox_pos_emb,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->Bros
class BrosIntermediate(ms.nn.Cell):
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


class BrosOutput(ms.nn.Cell):
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


class BrosLayer(ms.nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BrosAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise Exception(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BrosAttention(config)
        self.intermediate = BrosIntermediate(config)
        self.output = BrosOutput(config)

    def construct(
        self,
        hidden_states: ms.Tensor,
        bbox_pos_emb: ms.Tensor,
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
            bbox_pos_emb=bbox_pos_emb,
            attention_mask=attention_mask,
            head_mask=head_mask,
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
            if hasattr(self, "crossattention"):
                raise Exception(
                    f"If `encoder_hidden_states` are passed, {self} has to be\
                         instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
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
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BrosEncoder(ms.nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = ms.nn.CellList([BrosLayer(config) for _ in range(config.num_hidden_layers)])

    def construct(
        self,
        hidden_states: ms.Tensor,
        bbox_pos_emb: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        encoder_attention_mask: Optional[ms.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[ms.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[ms.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                raise NotImplementedError("Gradient checkpoint is not yet supported.")
            else:
                layer_outputs = layer_module(
                    hidden_states=hidden_states,
                    bbox_pos_emb=bbox_pos_emb,
                    attention_mask=attention_mask,
                    head_mask=layer_head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
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


# Copied from transformers.models.bert.modeling_bert.BertPooler with Bert->Bros
class BrosPooler(ms.nn.Cell):
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


class BrosRelationExtractor(ms.nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.n_relations = config.n_relations
        self.backbone_hidden_size = config.hidden_size
        self.head_hidden_size = config.hidden_size
        self.classifier_dropout_prob = config.classifier_dropout_prob

        self.drop = mint.nn.Dropout(self.classifier_dropout_prob)
        self.query = mint.nn.Linear(self.backbone_hidden_size, self.n_relations * self.head_hidden_size)

        self.key = mint.nn.Linear(self.backbone_hidden_size, self.n_relations * self.head_hidden_size)

        self.dummy_node = ms.Parameter(mint.zeros((1, self.backbone_hidden_size)))

    def construct(self, query_layer: ms.Tensor, key_layer: ms.Tensor):
        query_layer = self.query(self.drop(query_layer))

        dummy_vec = self.dummy_node.unsqueeze(0).repeat(1, key_layer.shape[1], 1)
        key_layer = mint.cat([key_layer, dummy_vec], dim=0)
        key_layer = self.key(self.drop(key_layer))

        query_layer = query_layer.view(
            query_layer.shape[0], query_layer.shape[1], self.n_relations, self.head_hidden_size
        )
        key_layer = key_layer.view(key_layer.shape[0], key_layer.shape[1], self.n_relations, self.head_hidden_size)

        relation_score = mint.matmul(
            query_layer.permute(2, 1, 0, 3), key_layer.permute(2, 1, 3, 0)
        )  # equivalent to ms.einsum("ibnd,jbnd->nbij", (query_layer, key_layer))

        return relation_score


class BrosPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BrosConfig
    base_model_prefix = "bros"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, mint.nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
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


@add_start_docstrings(
    "The bare Bros Model transformer outputting raw hidden-states without any specific head on top.",
    BROS_START_DOCSTRING,
)
class BrosModel(BrosPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BrosTextEmbeddings(config)
        self.bbox_embeddings = BrosBboxEmbeddings(config)
        self.encoder = BrosEncoder(config)

        self.pooler = BrosPooler(config) if add_pooling_layer else None

        self.init_weights()

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

    @add_start_docstrings_to_model_forward(BROS_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=BaseModelOutputWithPoolingAndCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        bbox: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        encoder_attention_mask: Optional[ms.Tensor] = None,
        past_key_values: Optional[List[ms.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[ms.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        Returns:

        Examples:

        ```python
        >>> import mindspore as ms
        >>> from mindone.transformers import BrosModel
        >>> from transformers import BrosProcessor

        >>> processor = BrosProcessor.from_pretrained("jinho8345/bros-base-uncased")

        >>> model = BrosModel.from_pretrained("jinho8345/bros-base-uncased", use_safetensors=True)

        >>> encoding = processor("Hello, my dog is cute", add_special_tokens=False, return_tensors="np")
        >>> bbox = ms.Tensor([[[0, 0, 1, 1]]]).repeat(1, encoding["input_ids"].shape[-1], 1)
        >>> encoding["bbox"] = bbox

        >>> outputs = model(input_ids=ms.Tensor(encoding["input_ids"]), bbox=bbox)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
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

        if bbox is None:
            raise ValueError("You have to specify bbox")

        batch_size, seq_length = input_shape

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = mint.ones(
                input_shape,
            )

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.broadcast_to((batch_size, seq_length))
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = mint.zeros(
                    input_shape,
                    dtype=ms.int64,
                )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: ms.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.shape
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = mint.ones(
                    encoder_hidden_shape,
                )
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
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        # if bbox has 2 points (4 float tensors) per token, convert it to 4 points (8 float tensors) per token
        if bbox.shape[-1] == 4:
            bbox = bbox[:, :, [0, 1, 2, 1, 2, 3, 0, 3]]
        scaled_bbox = bbox * self.config.bbox_scale
        bbox_position_embeddings = self.bbox_embeddings(scaled_bbox)

        encoder_outputs = self.encoder(
            embedding_output,
            bbox_pos_emb=bbox_position_embeddings,
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


@add_start_docstrings(
    """
    Bros Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    BROS_START_DOCSTRING,
)
class BrosForTokenClassification(BrosPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bros = BrosModel(config)
        classifier_dropout = (
            config.classifier_dropout if hasattr(config, "classifier_dropout") else config.hidden_dropout_prob
        )
        self.dropout = mint.nn.Dropout(classifier_dropout)
        self.classifier = mint.nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_model_forward(BROS_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        bbox: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        bbox_first_token_mask: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        labels: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[ms.Tensor], TokenClassifierOutput]:
        r"""

        Returns:

        Examples:

        ```python
        >>> import mindspore as ms
        >>> from mindone.transformers import BrosForTokenClassification
        >>> from transformers import BrosProcessor

        >>> processor = BrosProcessor.from_pretrained("jinho8345/bros-base-uncased")

        >>> model = BrosForTokenClassification.from_pretrained("jinho8345/bros-base-uncased", use_safetensors=True)

        >>> encoding = processor("Hello, my dog is cute", add_special_tokens=False, return_tensors="np")
        >>> bbox = ms.Tensor([[[0, 0, 1, 1]]]).repeat(1, encoding["input_ids"].shape[-1], 1)
        >>> encoding["bbox"] = bbox

        >>> outputs = model(input_ids=ms.Tensor(encoding["input_ids"]), bbox=bbox)
        ```"""

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bros(
            input_ids,
            bbox=bbox,
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
            loss_fct = CrossEntropyLoss()
            if bbox_first_token_mask is not None:
                bbox_first_token_mask = bbox_first_token_mask.view(-1)
                loss = loss_fct(
                    logits.view(-1, self.num_labels)[bbox_first_token_mask], labels.view(-1)[bbox_first_token_mask]
                )
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Bros Model with a token classification head on top (initial_token_layers and subsequent_token_layer on top of the
    hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. The initial_token_classifier is used to
    predict the first token of each entity, and the subsequent_token_classifier is used to predict the subsequent
    tokens within an entity. Compared to BrosForTokenClassification, this model is more robust to serialization errors
    since it predicts next token from one token.
    """,
    BROS_START_DOCSTRING,
)
class BrosSpadeEEForTokenClassification(BrosPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.n_relations = config.n_relations
        self.backbone_hidden_size = config.hidden_size

        self.bros = BrosModel(config)
        classifier_dropout = (
            config.classifier_dropout if hasattr(config, "classifier_dropout") else config.hidden_dropout_prob
        )

        # Initial token classification for Entity Extraction (NER)
        self.initial_token_classifier = ms.nn.SequentialCell(
            mint.nn.Dropout(classifier_dropout),
            mint.nn.Linear(config.hidden_size, config.hidden_size),
            mint.nn.Dropout(classifier_dropout),
            mint.nn.Linear(config.hidden_size, config.num_labels),
        )

        # Subsequent token classification for Entity Extraction (NER)
        self.subsequent_token_classifier = BrosRelationExtractor(config)

        self.init_weights()

    @add_start_docstrings_to_model_forward(BROS_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=BrosSpadeOutput, config_class=_CONFIG_FOR_DOC)
    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        bbox: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        bbox_first_token_mask: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        initial_token_labels: Optional[ms.Tensor] = None,
        subsequent_token_labels: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[ms.Tensor], BrosSpadeOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> import mindspore as ms
        >>> from mindone.transformers import BrosSpadeEEForTokenClassification
        >>> from transformers import BrosProcessor

        >>> processor = BrosProcessor.from_pretrained("jinho8345/bros-base-uncased")

        >>> model = BrosSpadeEEForTokenClassification.from_pretrained("jinho8345/bros-base-uncased", use_safetensors=True)

        >>> encoding = processor("Hello, my dog is cute", add_special_tokens=False, return_tensors="np")
        >>> bbox = ms.Tensor([[[0, 0, 1, 1]]]).repeat(1, encoding["input_ids"].shape[-1], 1)
        >>> encoding["bbox"] = bbox

        >>> outputs = model(input_ids=ms.Tensor(encoding["input_ids"]), bbox=bbox)
        ```"""

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bros(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_states = outputs[0]
        last_hidden_states = last_hidden_states.transpose(0, 1).contiguous()
        initial_token_logits = self.initial_token_classifier(last_hidden_states).transpose(0, 1).contiguous()
        subsequent_token_logits = self.subsequent_token_classifier(last_hidden_states, last_hidden_states).squeeze(0)

        if attention_mask is None:
            attention_mask = mint.ones(
                input_ids.shape,
            )
        # make subsequent token (sequence token classification) mask
        inv_attention_mask = 1 - attention_mask
        batch_size, max_seq_length = inv_attention_mask.shape
        invalid_token_mask = mint.cat([inv_attention_mask, mint.zeros([batch_size, 1])], dim=1).bool()

        subsequent_token_logits = subsequent_token_logits.masked_fill(
            invalid_token_mask[:, None, :], _DTYPE_2_MIN[subsequent_token_logits.dtype]
        )
        self_token_mask = mint.eye(max_seq_length, max_seq_length + 1).bool()
        subsequent_token_logits = subsequent_token_logits.masked_fill(
            self_token_mask[None, :, :], _DTYPE_2_MIN[subsequent_token_logits.dtype]
        )
        subsequent_token_mask = attention_mask.view(-1).bool()

        loss = None
        if initial_token_labels is not None and subsequent_token_labels is not None:
            loss_fct = CrossEntropyLoss()

            # get initial token loss
            initial_token_labels = initial_token_labels.view(-1)
            if bbox_first_token_mask is not None:
                bbox_first_token_mask = bbox_first_token_mask.view(-1)
                initial_token_loss = loss_fct(
                    initial_token_logits.view(-1, self.num_labels)[bbox_first_token_mask],
                    initial_token_labels[bbox_first_token_mask],
                )
            else:
                initial_token_loss = loss_fct(initial_token_logits.view(-1, self.num_labels), initial_token_labels)

            subsequent_token_labels = subsequent_token_labels.view(-1)
            subsequent_token_loss = loss_fct(
                subsequent_token_logits.view(-1, max_seq_length + 1)[subsequent_token_mask],
                subsequent_token_labels[subsequent_token_mask],
            )

            loss = initial_token_loss + subsequent_token_loss

        if not return_dict:
            output = (initial_token_logits, subsequent_token_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return BrosSpadeOutput(
            loss=loss,
            initial_token_logits=initial_token_logits,
            subsequent_token_logits=subsequent_token_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Bros Model with a token classification head on top (a entity_linker layer on top of the hidden-states output) e.g.
    for Entity-Linking. The entity_linker is used to predict intra-entity links (one entity to another entity).
    """,
    BROS_START_DOCSTRING,
)
class BrosSpadeELForTokenClassification(BrosPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.n_relations = config.n_relations
        self.backbone_hidden_size = config.hidden_size

        self.bros = BrosModel(config)
        (config.classifier_dropout if hasattr(config, "classifier_dropout") else config.hidden_dropout_prob)

        self.entity_linker = BrosRelationExtractor(config)

        self.init_weights()

    @add_start_docstrings_to_model_forward(BROS_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        bbox: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        bbox_first_token_mask: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        labels: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[ms.Tensor], TokenClassifierOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> import mindspore as ms
        >>> from mindone.transformers import BrosSpadeELForTokenClassification
        >>> from transformers import BrosProcessor

        >>> processor = BrosProcessor.from_pretrained("jinho8345/bros-base-uncased")

        >>> model = BrosSpadeELForTokenClassification.from_pretrained("jinho8345/bros-base-uncased", use_safetensors=True)

        >>> encoding = processor("Hello, my dog is cute", add_special_tokens=False, return_tensors="np")
        >>> bbox = ms.Tensor([[[0, 0, 1, 1]]]).repeat(1, encoding["input_ids"].shape[-1], 1)
        >>> encoding["bbox"] = bbox

        >>> outputs = model(input_ids=ms.Tensor(encoding["input_ids"]), bbox=bbox)
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bros(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_states = outputs[0]
        last_hidden_states = last_hidden_states.transpose(0, 1).contiguous()

        logits = self.entity_linker(last_hidden_states, last_hidden_states).squeeze(0)
        if attention_mask is None:
            attention_mask = mint.ones(
                input_ids.shape,
            )
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()

            batch_size, max_seq_length = attention_mask.shape

            self_token_mask = mint.eye(max_seq_length, max_seq_length + 1).bool()

            mask = bbox_first_token_mask.view(-1)
            bbox_first_token_mask = mint.cat(
                [
                    ~bbox_first_token_mask,
                    mint.zeros([batch_size, 1], dtype=ms.bool_),
                ],
                dim=1,
            )

            logits = logits.masked_fill(bbox_first_token_mask[:, None, :], _DTYPE_2_MIN[logits.dtype])
            logits = logits.masked_fill(self_token_mask[None, :, :], _DTYPE_2_MIN[logits.dtype])

            loss = loss_fct(logits.view(-1, max_seq_length + 1)[mask], labels.view(-1)[mask])

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "BrosPreTrainedModel",
    "BrosModel",
    "BrosForTokenClassification",
    "BrosSpadeEEForTokenClassification",
    "BrosSpadeELForTokenClassification",
]
