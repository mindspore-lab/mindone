# coding=utf-8
# Copyright 2023 The Intel Labs Team Authors, The Microsoft Research Team Authors and HuggingFace Inc. team. All rights reserved.
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
"""MindSpore BridgeTower Model"""

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from transformers.models.bridgetower.configuration_bridgetower import (
    BridgeTowerConfig,
    BridgeTowerTextConfig,
    BridgeTowerVisionConfig,
)

import mindspore as ms
from mindspore import mint, nn, ops
from mindspore.mint.nn import CrossEntropyLoss

from ...activations import ACT2FN, QuickGELUActivation
from ...mindspore_adapter import MultiheadAttention
from ...mindspore_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    ModelOutput,
    SequenceClassifierOutput,
)
from ...modeling_utils import PreTrainedModel, apply_chunking_to_forward
from ...utils import logging, mindspore_int

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "BridgeTowerConfig"
_CHECKPOINT_FOR_DOC = "BridgeTower/bridgetower-base"
_TOKENIZER_FOR_DOC = "RobertaTokenizer"


BRIDGETOWER_START_DOCSTRING = r"""
    This model is a MindSpore
    `ms.nn.Cell <https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Cell.html>`_ subclass. Use
    it as a regular MindSpore Cell and refer to the MindSpore documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`BridgeTowerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

BRIDGETOWER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`ms.Tensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary. Indices can be obtained using [`AutoTokenizer`]. See
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details. [What are input
            IDs?](../glossary#input-ids)

        attention_mask (`ms.Tensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)

        token_type_ids (`ms.Tensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:
            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.
            [What are token type IDs?](../glossary#token-type-ids)

        pixel_values (`ms.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`BridgeTowerImageProcessor`]. See
            [`BridgeTowerImageProcessor.__call__`] for details.

        pixel_mask (`ms.Tensor` of shape `(batch_size, height, width)`, *optional*):
            Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:

            - 1 for pixels that are real (i.e. **not masked**),
            - 0 for pixels that are padding (i.e. **masked**).
            `What are attention masks? <../glossary.html#attention-mask>`__

        head_mask (`ms.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`ms.Tensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.

        image_embeds (`ms.Tensor` of shape `(batch_size, num_patches, hidden_size)`, *optional*):
            Optionally, instead of passing `pixel_values`, you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `pixel_values` into patch embeddings.

        image_token_type_idx (`int`, *optional*):
            - The token type ids for images.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        interpolate_pos_encoding (`bool`, defaults to `False`):
            Whether to interpolate the pre-trained position encodings.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@dataclass
class BridgeTowerModelOutput(ModelOutput):
    """
    Output type of [`BridgeTowerModel`].

    Args:
        text_features (`ms.Tensor` of shape `(batch_size, text_sequence_length, hidden_size)`):
            Sequence of hidden-states at the text output of the last layer of the model.
        image_features (`ms.Tensor` of shape `(batch_size, image_sequence_length, hidden_size)`):
            Sequence of hidden-states at the image output of the last layer of the model.
        pooler_output (`ms.Tensor` of shape `(batch_size, hidden_size x 2)`):
            Concatenation of last layer hidden-state of the first token of the text and image sequence (classification
            token), respectively, after further processing through layers used for auxiliary pretraining tasks.
        hidden_states (`tuple(ms.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `ms.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of
            the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(ms.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `ms.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    text_features: Optional[ms.Tensor] = None
    image_features: Optional[ms.Tensor] = None
    pooler_output: Optional[ms.Tensor] = None
    hidden_states: Optional[Tuple[ms.Tensor]] = None
    attentions: Optional[Tuple[ms.Tensor]] = None


@dataclass
class BridgeTowerContrastiveOutput(ModelOutput):
    """
    Output type of ['BridgeTowerForContrastiveLearning']

    Args:
        loss (`ms.Tensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`:
            Image-text contrastive loss.
        logits (`ms.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        text_embeds (`ms.Tensor)`, *optional*, returned when model is initialized with `with_projection=True`):
            The text embeddings obtained by applying the projection layer to the pooler_output.
        image_embeds (`ms.Tensor)`, *optional*, returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        cross_embeds  (`ms.Tensor)`, *optional*, returned when model is initialized with `with_projection=True`):
            The text-image cross-modal embeddings obtained by applying the projection layer to the pooler_output.
        hidden_states (`tuple(ms.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `ms.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of
            the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(ms.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `ms.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
    """

    loss: Optional[ms.Tensor] = None
    logits: Optional[ms.Tensor] = None
    text_embeds: Optional[Tuple[ms.Tensor]] = None
    image_embeds: Optional[Tuple[ms.Tensor]] = None
    cross_embeds: Optional[Tuple[ms.Tensor]] = None
    hidden_states: Optional[Tuple[ms.Tensor]] = None
    attentions: Optional[Tuple[ms.Tensor]] = None


class BridgeTowerResidualAttention(nn.Cell):
    def __init__(self, config):
        super().__init__()

        self.attn = MultiheadAttention(config.hidden_size, config.hidden_size // 64)
        self.ln_1 = mint.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = nn.CellDict(
            OrderedDict(
                [
                    ("c_fc", mint.nn.Linear(config.hidden_size, config.hidden_size * 4)),
                    ("gelu", QuickGELUActivation()),
                    ("c_proj", mint.nn.Linear(config.hidden_size * 4, config.hidden_size)),
                ]
            )
        )
        self.ln_2 = mint.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn_mask = None

    def attention(self, hidden_state: ms.Tensor, attention_mask: ms.Tensor):
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=ms.bool_)
        self.attn_mask = self.attn_mask.to(dtype=hidden_state.dtype) if self.attn_mask is not None else None
        return self.attn(
            hidden_state,
            hidden_state,
            hidden_state,
            need_weights=False,
            attn_mask=self.attn_mask,
            key_padding_mask=attention_mask,
        )[0]

    def construct(self, hidden_state: ms.Tensor, attention_mask: Optional[ms.Tensor] = None):
        residual_state = hidden_state + self.attention(self.ln_1(hidden_state), attention_mask)
        hidden_state = self.ln_2(residual_state)
        for _, layer in self.mlp.items():
            hidden_state = layer(hidden_state)
        hidden_state = residual_state + hidden_state
        return hidden_state


class BridgeTowerTransformer(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        if config.remove_last_layer:
            self.resblocks = nn.CellList(
                [BridgeTowerResidualAttention(config) for _ in range(self.num_hidden_layers - 1)]
            )
        else:
            self.resblocks = nn.CellList([BridgeTowerResidualAttention(config) for _ in range(self.num_hidden_layers)])
        self.stop_gradient = config.stop_gradient

    def construct(self, hidden_state: ms.Tensor, attention_mask: Optional[ms.Tensor] = None):
        hidden_states = []
        for block in self.resblocks:
            hidden_state = block(hidden_state, attention_mask)
            if self.stop_gradient:
                hidden_states.append(ops.stop_gradient(hidden_state))
            else:
                hidden_states.append(hidden_state)
        return hidden_states


# Copied from transformers.models.clip.modeling_clip.CLIPVisionEmbeddings with CLIP->BridgeTower
class BridgeTowerVisionEmbeddings(nn.Cell):
    def __init__(self, config: BridgeTowerVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = ms.Parameter(mint.randn(self.embed_dim), name="class_embedding")

        self.patch_embedding = mint.nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = mint.nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", mint.arange(self.num_positions).broadcast_to((1, -1)), persistent=False)

    def interpolate_pos_encoding(self, embeddings: ms.Tensor, height: int, width: int) -> ms.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """

        num_patches = embeddings.shape[1] - 1
        position_embedding = self.position_embedding.weight.unsqueeze(0)
        num_positions = position_embedding.shape[1] - 1

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if num_patches == num_positions and height == width:
            return self.position_embedding(self.position_ids)

        class_pos_embed = position_embedding[:, :1]
        patch_pos_embed = position_embedding[:, 1:]

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = mindspore_int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = mint.nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return mint.cat((class_pos_embed, patch_pos_embed), dim=1)

    def construct(self, pixel_values: ms.Tensor, interpolate_pos_encoding=False) -> ms.Tensor:
        batch_size, _, height, width = pixel_values.shape
        if not interpolate_pos_encoding and (height != self.image_size or width != self.image_size):
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size}*{self.image_size})."
            )
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).swapaxes(1, 2)

        class_embeds = self.class_embedding.broadcast_to((batch_size, 1, -1))
        embeddings = mint.cat([class_embeds, patch_embeds], dim=1)
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class BridgeTowerVisionTransformer(nn.Cell):
    def __init__(self, config):
        super().__init__()

        self.embeddings = BridgeTowerVisionEmbeddings(config)
        self.ln_pre = mint.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.transformer = BridgeTowerTransformer(config)
        self.ln_post = mint.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.share_layernorm = config.share_layernorm
        if not config.share_layernorm:
            self.ln_separate = nn.CellList(
                [
                    mint.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
                    for _ in range(config.num_hidden_layers)
                ]
            )

    def construct(
        self,
        pixel_values: ms.Tensor,
        attention_mask,
        interpolate_pos_encoding: bool = False,
    ):
        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding)
        hidden_states = self.ln_pre(hidden_states)
        # NLD -> LND
        hidden_states = hidden_states.permute(1, 0, 2)

        hidden_states = self.transformer(hidden_states, attention_mask)
        # shape = [num_hidden_layers, hidden_size, *, grid ** 2]
        hidden_states = mint.stack(hidden_states, dim=0)
        # shape = [num_hidden_layers, *, hidden_size, grid ** 2]
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        if self.share_layernorm:
            hidden_states = self.ln_post(hidden_states)
        else:
            hidden_states_stack = []
            for hidden_states, ln in zip(hidden_states, self.ln_separate):
                hidden_states = ln(hidden_states)
                hidden_states_stack.append(hidden_states)
            # shape = [num_hidden_layers, *, hidden_size, grid ** 2]
            hidden_states = mint.stack(hidden_states_stack, dim=0)
        return hidden_states

    def forward_pre(
        self,
        pixel_values: ms.Tensor,
        interpolate_pos_encoding: bool = False,
    ):
        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        hidden_states = self.ln_pre(hidden_states)
        # NLD -> LND
        hidden_states = hidden_states.permute(1, 0, 2)
        return hidden_states

    def forward_post(self, hidden_state: ms.Tensor):
        visual_output_post = hidden_state.permute(1, 0, 2)
        visual_output_post = self.ln_post(visual_output_post)
        return visual_output_post


class BridgeTowerLinkTower(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.link_tower_type = config.link_tower_type
        self.hidden_size = config.hidden_size
        if config.link_tower_type in ["add", "scaled_add", "interpolate"]:
            if config.link_tower_type == "scaled_add":
                self.scaled_factor = ms.Parameter(ms.tensor(1.0), name="scaled_factor")
            elif config.link_tower_type == "interpolate":
                self.beta = ms.Parameter(ms.tensor(0.5), name="beta")
            self.LayerNorm = mint.nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        else:
            raise NotImplementedError(f"link_tower_type {config.link_tower_type} is not implemented")

    def construct(self, hidden_states, cross_modal_hidden_states, attention_mask):
        if self.link_tower_type == "add":
            return self.LayerNorm(hidden_states + cross_modal_hidden_states)
        elif self.link_tower_type == "scaled_add":
            return self.LayerNorm(hidden_states * self.scaled_factor + cross_modal_hidden_states)
        elif self.link_tower_type == "interpolate":
            return self.LayerNorm(hidden_states * (1 - self.beta) + cross_modal_hidden_states * self.beta)
        else:
            raise NotImplementedError(f"link_tower_type {self.link_tower_type} is not implemented")


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput with Bert->BridgeTower
class BridgeTowerSelfOutput(nn.Cell):
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


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->BridgeTower
class BridgeTowerIntermediate(nn.Cell):
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


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->BridgeTower
class BridgeTowerOutput(nn.Cell):
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


# Copied from transformers.models.bert.modeling_bert.BertPooler with Bert->BridgeTower
class BridgeTowerPooler(nn.Cell):
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


# Copied from transformers.models.roberta.modeling_roberta.RobertaSelfAttention with Roberta->BridgeTower
class BridgeTowerSelfAttention(nn.Cell):
    def __init__(self, config, position_embedding_type=None):
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
        self.position_embedding_type = position_embedding_type or getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = mint.nn.Embedding(
                2 * config.max_position_embeddings - 1, self.attention_head_size
            )

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
        past_key_value: Optional[Tuple[Tuple[ms.Tensor]]] = None,
        output_attentions: Optional[bool] = False,
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

        use_cache = past_key_value is not None
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

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = ms.tensor(key_length - 1, dtype=ms.int64).view(-1, 1)
            else:
                position_ids_l = mint.arange(query_length, dtype=ms.int64).view(-1, 1)
            position_ids_r = mint.arange(key_length, dtype=ms.int64).view(1, -1)
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

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BridgeTowerModel forward() function)
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
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


BRIDGE_TOWER_SELF_ATTENTION_CLASSES = {
    "eager": BridgeTowerSelfAttention,
}


# Copied from transformers.models.bert.modeling_bert.BertAttention with Bert->BridgeTower,BERT->BRIDGE_TOWER
class BridgeTowerAttention(nn.Cell):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = BRIDGE_TOWER_SELF_ATTENTION_CLASSES[config._attn_implementation](
            config, position_embedding_type=position_embedding_type
        )
        self.output = BridgeTowerSelfOutput(config)
        self.pruned_heads = set()

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


class BridgeTowerBertCrossLayer(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BridgeTowerAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        self.crossattention = BridgeTowerAttention(config)
        self.intermediate = BridgeTowerIntermediate(config)
        self.output = BridgeTowerOutput(config)

    def construct(
        self,
        hidden_states,
        encoder_hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=None,
            output_attentions=output_attentions,
            past_key_value=None,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        # add self attentions if we output attention weights
        outputs = self_attention_outputs[1:]

        cross_attention_outputs = self.crossattention(
            attention_output,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
        )
        attention_output = cross_attention_outputs[0]
        # add cross attentions if we output attention weights
        outputs = outputs + cross_attention_outputs[1:-1]

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BridgeTowerTextLayer(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BridgeTowerAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BridgeTowerAttention(config, position_embedding_type="absolute")
        self.intermediate = BridgeTowerIntermediate(config)
        self.output = BridgeTowerOutput(config)

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

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# Copied from transformers.models.roberta.modeling_roberta.RobertaEncoder with Roberta->BridgeTowerText
class BridgeTowerTextEncoder(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.CellList([BridgeTowerTextLayer(config) for _ in range(config.num_hidden_layers)])
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
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[ms.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

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


# Copied from transformers.models.roberta.modeling_roberta.RobertaEmbeddings with Roberta->BridgeTowerText
class BridgeTowerTextEmbeddings(nn.Cell):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
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
        self.position_ids = mint.arange(config.max_position_embeddings).broadcast_to((1, -1))
        self.token_type_ids = mint.zeros(self.position_ids.shape, dtype=ms.int64)

        # End copy
        self.padding_idx = config.pad_token_id
        self.position_embeddings = mint.nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def construct(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.broadcast_to((input_shape[0], seq_length))
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = mint.zeros(input_shape, dtype=ms.int64)

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

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: ms.Tensor

        Returns: ms.Tensor
        """
        input_shape = inputs_embeds.shape[:-1]
        sequence_length = input_shape[1]

        position_ids = mint.arange(self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=ms.int64)
        return position_ids.unsqueeze(0).broadcast_to(input_shape)


# Copied from transformers.models.roberta.modeling_roberta.create_position_ids_from_input_ids
def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: ms.Tensor x:

    Returns: ms.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (mint.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx


class BridgeTowerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BridgeTowerConfig
    base_model_prefix = "bridgetower"
    supports_gradient_checkpointing = False
    _no_split_modules = ["BridgeTowerSelfAttention", "BridgeTowerResidualAttention"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        pass


class BridgeTowerVisionModel(BridgeTowerPreTrainedModel):
    config_class = BridgeTowerVisionConfig

    def __init__(self, config):
        super().__init__(config)
        self.visual = BridgeTowerVisionTransformer(config)

    @property
    def dtype(self):
        return self.visual.embeddings.patch_embedding.weight.dtype

    def construct(self, image, image_mask=None, interpolate_pos_encoding=False):
        return self.visual(image.type(self.dtype), image_mask, interpolate_pos_encoding)


class BridgeTowerTextModel(BridgeTowerPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in *Attention is
    all you need*_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
    Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.

    .. _*Attention is all you need*: https://arxiv.org/abs/1706.03762

    """

    config_class = BridgeTowerTextConfig

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BridgeTowerTextEmbeddings(config)
        self.encoder = BridgeTowerTextEncoder(config)

        self.pooler = BridgeTowerPooler(config) if add_pooling_layer else None

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

    # Copied from transformers.models.clap.modeling_clap.ClapTextModel.forward
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
        past_key_values: Optional[List[ms.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[ms.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`ms.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(ms.Tensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):  # noqa: E501
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
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
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = mint.ones(((batch_size, seq_length + past_key_values_length)))

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.broadcast_to((batch_size, seq_length))
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = mint.zeros(input_shape, dtype=ms.int64)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: ms.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
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


class BridgeTowerModel(BridgeTowerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        vision_config = config.vision_config
        text_config = config.text_config

        if config.share_cross_modal_transformer_layers:
            self.cross_modal_text_transform = mint.nn.Linear(text_config.hidden_size, config.hidden_size)
            self.cross_modal_image_transform = mint.nn.Linear(vision_config.hidden_size, config.hidden_size)
        else:
            self.cross_modal_text_transform = nn.CellList(
                [mint.nn.Linear(text_config.hidden_size, config.hidden_size) for _ in range(config.num_hidden_layers)]
            )
            self.cross_modal_image_transform = nn.CellList(
                [mint.nn.Linear(vision_config.hidden_size, config.hidden_size) for _ in range(config.num_hidden_layers)]
            )

        self.token_type_embeddings = mint.nn.Embedding(2, config.hidden_size)

        self.vision_model = BridgeTowerVisionModel(vision_config)

        self.text_model = BridgeTowerTextModel(text_config)

        if not vision_config.share_layernorm and config.init_layernorm_from_vision_encoder:
            for ln in self.vision_model.visual.cross_modal_ln_separate:
                ln.weight.data = self.vision_model.visual.ln_post.weight.data
                ln.bias.data = self.vision_model.visual.ln_post.bias.data

        self.cross_modal_image_layers = nn.CellList(
            [BridgeTowerBertCrossLayer(text_config) for _ in range(config.num_hidden_layers)]
        )
        self.cross_modal_text_layers = nn.CellList(
            [BridgeTowerBertCrossLayer(text_config) for _ in range(config.num_hidden_layers)]
        )

        # Class token => Linear => Tanh
        self.cross_modal_image_pooler = BridgeTowerPooler(config)
        self.cross_modal_text_pooler = BridgeTowerPooler(config)

        # Initialize BridgeTower Components
        self.cross_modal_text_layernorm = mint.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.cross_modal_image_layernorm = mint.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        if config.share_link_tower_layers:
            self.cross_modal_text_link_tower = BridgeTowerLinkTower(config)
            self.cross_modal_image_link_tower = BridgeTowerLinkTower(config)
        else:
            self.cross_modal_text_link_tower = nn.CellList(
                [BridgeTowerLinkTower(config) for _ in range(config.num_hidden_layers - 1)]
            )
            self.cross_modal_image_link_tower = nn.CellList(
                [BridgeTowerLinkTower(config) for _ in range(config.num_hidden_layers - 1)]
            )

        self.post_init()

    def get_input_embeddings(self):
        return self.text_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.text_model.set_input_embeddings(value)

    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
        pixel_values: Optional[ms.Tensor] = None,
        pixel_mask: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        image_embeds: Optional[ms.Tensor] = None,
        image_token_type_idx: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[ms.Tensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> Union[Tuple[ms.Tensor], BridgeTowerModelOutput]:
        r"""
        output_hidden_states (`bool`, *optional*):
            If set to `True`, hidden states are returned as a list containing the hidden states of text, image, and
            cross-modal components respectively. i.e. `(hidden_states_text, hidden_states_image,
            hidden_states_cross_modal)` where each element is a list of the hidden states of the corresponding
            modality. `hidden_states_txt/img` are a list of tensors corresponding to unimodal hidden states and
            `hidden_states_cross_modal` is a list of tuples containing `cross_modal_text_hidden_states` and
            `cross_modal_image_hidden_states` of each brdige layer.
        labels (`ms.Tensor` of shape `(batch_size,)`, *optional*):
            Labels are currently not supported.
        Returns:

        Examples:

        ```python
        >>> from transformers import BridgeTowerProcessor
        >>> from mindone.transformers import BridgeTowerModel
        >>> from PIL import Image
        >>> import requests
        >>> import mindspore as ms

        >>> # prepare image and text
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "hello world"
        >>> processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-base", revision="refs/pr/2")
        >>> model = BridgeTowerModel.from_pretrained("BridgeTower/bridgetower-base", revision="refs/pr/2")

        >>> inputs = processor(image, text, return_tensors="np")
        >>> inputs = {k: ms.tensor(v) for k, v in inputs.items()}
        >>> outputs = model(**inputs)
        >>> outputs.keys()
        odict_keys(['text_features', 'image_features', 'pooler_output'])
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        all_hidden_states_text = () if output_hidden_states else None
        all_hidden_states_image = () if output_hidden_states else None
        all_hidden_states_cross = () if output_hidden_states else None
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if inputs_embeds is not None and input_ids is None:
            raise NotImplementedError(
                "BridgeTowerModel does not use `inputs_embeds`.  Make sure to pass in `input_ids` instead."
            )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        image_token_type_idx = image_token_type_idx if image_token_type_idx else 1
        input_shape = input_ids.shape
        text_embeds = self.text_model.embeddings(input_ids=input_ids)

        if output_hidden_states:
            all_hidden_states_text += (text_embeds,)

        if attention_mask is None:
            attention_mask = mint.ones(input_shape, dtype=ms.int64)
        extend_text_masks = self.text_model.get_extended_attention_mask(attention_mask, input_shape)

        # The split_index determines how many layers of the uni-modal encoder are applied before the cross-modal encoder
        split_index = len(self.text_model.encoder.layer) - self.config.num_hidden_layers + 1

        # Run the first 'split_index' layers of the textual encoder
        for layer in self.text_model.encoder.layer[:split_index]:
            text_embeds = layer(text_embeds, extend_text_masks)[0]

            if output_hidden_states:
                all_hidden_states_text += (text_embeds,)

        if image_embeds is None:
            image_embeds = self.vision_model.visual.forward_pre(
                pixel_values.type(self.vision_model.dtype), interpolate_pos_encoding=interpolate_pos_encoding
            )
        else:
            # Permute as BridgeTowerResidualAttention has batch_first=True
            image_embeds = image_embeds.permute(1, 0, 2)

        if output_hidden_states:
            all_hidden_states_image += (image_embeds,)

        # Run the first 'split_index' layers of the visual encoder
        for block in self.vision_model.visual.transformer.resblocks[:split_index]:
            image_embeds = block(image_embeds)
            if output_hidden_states:
                all_hidden_states_image += (image_embeds,)

        image_embeds_with_ln = self.vision_model.visual.forward_post(image_embeds.type(self.vision_model.dtype))

        # first layer is a special case because we don't have the output from the cross-encoder yet
        cross_modal_text = self.cross_modal_text_transform(text_embeds)

        text_token_type_embeddings = self.token_type_embeddings(mint.zeros(1, dtype=ms.int64)).expand_as(
            cross_modal_text
        )

        cross_modal_text = self.cross_modal_text_layernorm(cross_modal_text + text_token_type_embeddings)

        image_embeds_with_ln = self.cross_modal_image_transform(image_embeds_with_ln)
        image_token_type_embeddings = self.token_type_embeddings(
            mint.full((1,), image_token_type_idx, dtype=ms.int64)
        ).expand_as(image_embeds_with_ln)

        image_embeds_with_ln = image_embeds_with_ln + image_token_type_embeddings
        cross_modal_image = self.cross_modal_image_layernorm(image_embeds_with_ln)

        pixel_mask = mint.ones((cross_modal_image.shape[0], cross_modal_image.shape[1]), dtype=ms.int64)
        extend_image_masks = self.text_model.get_extended_attention_mask(pixel_mask, pixel_mask.shape)

        layer_outputs_text = self.cross_modal_text_layers[0](
            cross_modal_text,
            cross_modal_image,
            attention_mask=extend_text_masks,
            encoder_attention_mask=extend_image_masks,
            output_attentions=output_attentions,
        )
        cross_text_features = layer_outputs_text[0]

        layer_outputs_image = self.cross_modal_image_layers[0](
            cross_modal_image,
            cross_modal_text,
            attention_mask=extend_image_masks,
            encoder_attention_mask=extend_text_masks,
            output_attentions=output_attentions,
        )
        cross_image_features = layer_outputs_image[0]

        if output_hidden_states:
            all_hidden_states_cross += ((cross_text_features, cross_image_features),)

        if output_attentions:
            all_self_attentions += ((layer_outputs_text[1], layer_outputs_image[1]),)

        link_layer_index = 0

        #  Each of the top 6 layers of the visual and textual encoders ([split_index:]) is connected to each layer of
        #  the cross-modal encoder via bridge layers, which brings bottom-up alignment and fusion to the cross-modal encoder.
        for i in range(split_index, len(self.text_model.encoder.layer)):
            text_embeds = self.text_model.encoder.layer[i](text_embeds, extend_text_masks)[0]
            image_embeds = self.vision_model.visual.transformer.resblocks[i](image_embeds).type(self.vision_model.dtype)
            image_embeds_with_ln = (
                self.cross_modal_image_transform(self.vision_model.visual.forward_post(image_embeds))
                + image_token_type_embeddings
            )

            text_link_tower = self.cross_modal_text_link_tower[link_layer_index]
            image_link_tower = self.cross_modal_image_link_tower[link_layer_index]

            # Bridge layers for textual and visual encoders
            cross_text_features_ = text_link_tower(
                self.cross_modal_text_transform(text_embeds) + text_token_type_embeddings,
                cross_text_features,
                extend_text_masks,
            )
            cross_image_features_ = image_link_tower(image_embeds_with_ln, cross_image_features, extend_image_masks)

            # Cross-modal encoder via bridge layers of textual and visual encoders
            layer_outputs_text = self.cross_modal_text_layers[link_layer_index + 1](
                cross_text_features_,
                cross_image_features_,
                attention_mask=extend_text_masks,
                encoder_attention_mask=extend_image_masks,
                output_attentions=output_attentions,
            )
            cross_text_features = layer_outputs_text[0]

            layer_outputs_image = self.cross_modal_image_layers[link_layer_index + 1](
                cross_image_features_,
                cross_text_features_,
                attention_mask=extend_image_masks,
                encoder_attention_mask=extend_text_masks,
                output_attentions=output_attentions,
            )
            cross_image_features = layer_outputs_image[0]

            link_layer_index += 1

            if output_hidden_states:
                all_hidden_states_text += (text_embeds,)
                all_hidden_states_image += (image_embeds,)
                all_hidden_states_cross += ((cross_text_features, cross_image_features),)

            if output_attentions:
                all_self_attentions += ((layer_outputs_text[1], layer_outputs_image[1]),)

        #  Concatenate the cls token of the text and image features to get the final represtation
        text_features, image_features = cross_text_features, cross_image_features
        cls_features = self.get_cls_features(text_features, image_features)

        if output_hidden_states:
            all_hidden_states = (all_hidden_states_text, all_hidden_states_image, all_hidden_states_cross)

        if not return_dict:
            return tuple(
                v
                for v in [text_features, image_features, cls_features, all_hidden_states, all_self_attentions]
                if v is not None
            )

        return BridgeTowerModelOutput(
            text_features=text_features,
            image_features=image_features,
            pooler_output=cls_features,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    def get_cls_features(self, text_features, image_features):
        cls_features_text = self.cross_modal_text_pooler(text_features)
        cls_features_image = self.cross_modal_image_pooler(image_features)
        return mint.cat([cls_features_text, cls_features_image], dim=-1)


# Copied from transformers.models.vilt.modeling_vilt.ViltPredictionHeadTransform with Vilt->BridgeTower
class BridgeTowerPredictionHeadTransform(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.dense = mint.nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = mint.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def construct(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BridgeTowerMLMHead(nn.Cell):
    def __init__(self, config, weight=None):
        super().__init__()
        self.config = config
        self.transform = BridgeTowerPredictionHeadTransform(config)
        self.decoder = mint.nn.Linear(config.hidden_size, config.text_config.vocab_size, bias=False)
        self.bias = ms.Parameter(mint.zeros(config.text_config.vocab_size), name="bias")
        if weight is not None:
            self.decoder.weight = weight

    def construct(self, x):
        mlm_score = self.transform(x)
        mlm_score = self.decoder(mlm_score) + self.bias
        return mlm_score


class BridgeTowerITMHead(nn.Cell):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = mint.nn.Linear(hidden_size, 2)

    def construct(self, x):
        itm_score = self.fc(x)
        return itm_score


class BridgeTowerForMaskedLM(BridgeTowerPreTrainedModel):
    _tied_weights_keys = ["mlm_score.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)

        self.bridgetower = BridgeTowerModel(config)
        self.mlm_score = BridgeTowerMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.mlm_score.decoder

    def set_output_embeddings(self, new_embeddings):
        self.mlm_score.decoder = new_embeddings

    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
        pixel_values: Optional[ms.Tensor] = None,
        pixel_mask: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        image_embeds: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[ms.Tensor] = None,
    ) -> Union[MaskedLMOutput, Tuple[ms.Tensor]]:
        r"""
        labels (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        Returns:

        Examples:

        ```python
        >>> from transformers import BridgeTowerProcessor
        >>> from mindone.transformers import BridgeTowerForMaskedLM
        >>> from PIL import Image
        >>> import requests
        >>> import mindspore as ms

        >>> url = "http://images.cocodataset.org/val2017/000000360943.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        >>> text = "a <mask> looking out of the window"

        >>> processor = BridgeTowerProcessor.from_pretrained(
        ...     "BridgeTower/bridgetower-base-itm-mlm", revision="refs/pr/4"
        ... )
        >>> model = BridgeTowerForMaskedLM.from_pretrained("BridgeTower/bridgetower-base-itm-mlm", revision="refs/pr/4")

        >>> # prepare inputs
        >>> encoding = processor(image, text, return_tensors="np")
        >>> encoding = {k: ms.tensor(v) for k, v in encoding.items()}

        >>> # forward pass
        >>> outputs = model(**encoding)

        >>> results = processor.decode(outputs.logits.argmax(dim=-1).squeeze(0).tolist())

        >>> print(results)
        .a cat looking out of the window.
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bridgetower(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            image_embeds=image_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        mlm_logits = self.mlm_score(outputs.text_features if return_dict else outputs[0])
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token

            masked_lm_loss = loss_fct(mlm_logits.view(-1, self.config.text_config.vocab_size), labels.view(-1))

        if not return_dict:
            output = tuple(mlm_logits)
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=mlm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BridgeTowerForImageAndTextRetrieval(BridgeTowerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bridgetower = BridgeTowerModel(config)

        self.itm_score = BridgeTowerITMHead(config.hidden_size * 2)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
        pixel_values: Optional[ms.Tensor] = None,
        pixel_mask: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        image_embeds: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[ms.Tensor] = None,
    ) -> Union[SequenceClassifierOutput, Tuple[ms.Tensor]]:
        r"""
        labels (`ms.Tensor` of shape `(batch_size, 1)`, *optional*):
            Labels for computing the image-text matching loss. 0 means the pairs don't match and 1 means they match.
            The pairs with 0 will be skipped for calculation.
        Returns:

        Examples:

        ```python
        >>> from transformers import BridgeTowerProcessor
        >>> from mindone.transformers import BridgeTowerForImageAndTextRetrieval
        >>> import requests
        >>> from PIL import Image
        >>> import mindspore as ms

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> texts = ["An image of two cats chilling on a couch", "A football player scoring a goal"]

        >>> processor = BridgeTowerProcessor.from_pretrained(
        ...     "BridgeTower/bridgetower-base-itm-mlm", revision="refs/pr/4"
        ... )
        >>> model = BridgeTowerForImageAndTextRetrieval.from_pretrained(
        ...     "BridgeTower/bridgetower-base-itm-mlm", revision="refs/pr/4"
        ... )

        >>> # forward pass
        >>> scores = dict()
        >>> for text in texts:
        ...     # prepare inputs
        ...     encoding = processor(image, text, return_tensors="np")
        ...     encoding = {k: ms.tensor(v) for k, v in encoding.items()}
        ...     outputs = model(**encoding)
        ...     scores[text] = outputs.logits[0, 1].item()
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bridgetower(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            image_embeds=image_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooler_output = outputs.pooler_output if return_dict else outputs[2]

        logits = self.itm_score(pooler_output)

        itm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()

            itm_loss = loss_fct(logits, labels)

        if not return_dict:
            output = tuple(logits)
            return ((itm_loss,) + output) if itm_loss is not None else output

        return SequenceClassifierOutput(
            loss=itm_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BridgeTowerContrastiveHead(nn.Cell):
    def __init__(self, hidden_size, embed_size):
        super().__init__()
        self.fc = mint.nn.Linear(hidden_size, embed_size)

    def construct(self, x):
        x = self.fc(x)
        return x


class BridgeTowerForContrastiveLearning(BridgeTowerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bridgetower = BridgeTowerModel(config)

        self.itc_text_head = BridgeTowerContrastiveHead(config.hidden_size, config.contrastive_hidden_size)
        self.itc_image_head = BridgeTowerContrastiveHead(config.hidden_size, config.contrastive_hidden_size)
        self.itc_cross_modal_head = BridgeTowerContrastiveHead(config.hidden_size * 2, config.contrastive_hidden_size)

        self.logit_scale = ms.Parameter(ms.tensor(self.config.logit_scale_init_value), name="logit_scale")
        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
        pixel_values: Optional[ms.Tensor] = None,
        pixel_mask: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        image_embeds: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = None,
        return_loss: Optional[bool] = None,
    ) -> Union[BridgeTowerContrastiveOutput, Tuple[ms.Tensor]]:
        r"""
        return_loss (`bool`, *optional*):
            Whether or not to return the contrastive loss.
        Returns:

        Examples:

        ```python
        >>> from transformers import BridgeTowerProcessor
        >>> from mindone.transformers import BridgeTowerForContrastiveLearning
        >>> import requests
        >>> from PIL import Image
        >>> import mindspore as ms

        >>> image_urls = [
        ...     "https://farm4.staticflickr.com/3395/3428278415_81c3e27f15_z.jpg",
        ...     "http://images.cocodataset.org/val2017/000000039769.jpg",
        ... ]
        >>> texts = ["two dogs in a car", "two cats sleeping on a couch"]
        >>> images = [Image.open(requests.get(url, stream=True).raw) for url in image_urls]

        >>> processor = BridgeTowerProcessor.from_pretrained(
        ...     "BridgeTower/bridgetower-large-itm-mlm-itc", revision="refs/pr/1"
        ... )
        >>> model = BridgeTowerForContrastiveLearning.from_pretrained(
        ...     "BridgeTower/bridgetower-large-itm-mlm-itc", revision="refs/pr/1"
        ... )

        >>> inputs = processor(images, texts, padding=True, return_tensors="np")
        >>> inputs = {k: ms.tensor(v) for k, v in inputs.items()}
        >>> loss = model(**inputs, return_loss=True).loss

        >>> inputs = processor(images, texts[::-1], padding=True, return_tensors="np")
        >>> inputs = {k: ms.tensor(v) for k, v in inputs.items()}
        >>> loss_swapped = model(**inputs, return_loss=True).loss

        >>> print("Loss", round(loss.item(), 4))
        Loss 0.0019

        >>> print("Loss with swapped images", round(loss_swapped.item(), 4))
        Loss with swapped images 2.126
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bridgetower(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            image_embeds=image_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        pooler_output = outputs.pooler_output if return_dict else outputs[2]
        hidden_states_txt, hidden_states_img, hidden_states_cross_modal = (
            outputs.hidden_states if return_dict else outputs[3]
        )

        text_embeds = hidden_states_txt[-1]
        image_embeds = hidden_states_img[-1]

        image_embeds_with_ln = self.bridgetower.vision_model.visual.forward_post(image_embeds)
        image_token_type_embeddings = self.bridgetower.token_type_embeddings(
            mint.full((1,), 1, dtype=ms.int64)
        ).expand_as(image_embeds_with_ln)

        image_embeds = self.bridgetower.cross_modal_image_transform(image_embeds_with_ln) + image_token_type_embeddings

        # normalized features
        text_embeds = mint.nn.functional.normalize(self.itc_text_head(text_embeds[:, 0, :]), dim=-1, p=2)
        image_embeds = mint.nn.functional.normalize(self.itc_image_head(image_embeds[:, 0, :]), dim=-1, p=2)
        cross_embeds = mint.nn.functional.normalize(self.itc_cross_modal_head(pooler_output), dim=-1, p=2)

        logits = mint.stack([text_embeds, image_embeds, cross_embeds], dim=-2)

        logit_scale = self.logit_scale.exp()
        logits_text_to_image = mint.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_text_to_cross = mint.matmul(text_embeds, cross_embeds.t()) * logit_scale
        logits_image_to_cross = mint.matmul(image_embeds, cross_embeds.t()) * logit_scale

        itc_loss = None

        if return_loss:
            labels = mint.arange(len(logits))
            text_to_image_loss = mint.nn.functional.cross_entropy(logits_text_to_image, labels)
            text_to_cross_loss = mint.nn.functional.cross_entropy(logits_text_to_cross, labels)
            image_to_cross_loss = mint.nn.functional.cross_entropy(logits_image_to_cross, labels)
            itc_loss = (text_to_image_loss + text_to_cross_loss + image_to_cross_loss) / 3.0

        if not return_dict:
            output = (logits, text_embeds, image_embeds, cross_embeds) + outputs[3:]
            return ((itc_loss,) + output) if itc_loss is not None else output

        return BridgeTowerContrastiveOutput(
            loss=itc_loss,
            logits=logits,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            cross_embeds=cross_embeds,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "BridgeTowerForContrastiveLearning",
    "BridgeTowerForImageAndTextRetrieval",
    "BridgeTowerForMaskedLM",
    "BridgeTowerModel",
    "BridgeTowerPreTrainedModel",
]
