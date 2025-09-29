# coding=utf-8
# Copyright 2023 The Intel AIA Team Authors, and HuggingFace Inc. team. All rights reserved.
#
# This code is adapted from https://github.com/huggingface/transformers
# with modifications to run transformers on mindspore.
#
# Licensed under the Apache License=, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing=, software
# distributed under the License is distributed on an "AS IS" BASIS=,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND=, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MindSpore TVP Model"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from transformers import TvpConfig
from transformers.file_utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

import mindspore as ms
from mindspore import mint, nn
from mindspore.common.initializer import HeNormal, initializer

from mindone.models.utils import normal_, ones_, zeros_

from ...activations import ACT2FN
from ...mindspore_utils import prune_linear_layer
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from ...utils.backbone_utils import load_backbone

logger = logging.get_logger(__name__)


@dataclass
class TvpVideoGroundingOutput(ModelOutput):
    """
    Args:
        loss (`ms.Tensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Temporal-Distance IoU loss for video grounding.
        logits (`ms.Tensor` of shape `(batch_size, 2)`):
            Contains start_time/duration and end_time/duration. It is the time slot of the videos corresponding to the
            input texts.
        hidden_states (`tuple(ms.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `ms.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of
            the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(ms.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `ms.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
    """

    loss: Optional[ms.Tensor] = None
    logits: ms.Tensor = None
    hidden_states: Optional[Tuple[ms.Tensor, ...]] = None
    attentions: Optional[Tuple[ms.Tensor, ...]] = None


class TvpLoss(nn.Cell):
    """
    This class computes the losses for `TvpForVideoGrounding`. The process happens in two steps: 1) we compute
    hungarian assignment between ground truth boxes and the outputs of the model 2) we supervise each pair of matched
    ground-truth / prediction (supervise class and box).

    Args:
        losses (`List[str]`):
            List of all the losses to be applied.
    """

    def __init__(self, losses):
        super().__init__()
        self.loss_map = {
            "iou": self.loss_iou,
            "distance": self.loss_distance,
            "duration": self.loss_duration,
        }
        for loss in losses:
            if loss not in self.loss_map:
                raise ValueError(f"Loss {loss} not supported")

        self.losses = losses

    def loss_iou(self, start_time, end_time, candidates_start_time, candidates_end_time, duration):
        """
        Measure the intersection over union.
        """
        inter = mint.min(candidates_end_time, end_time) - mint.max(candidates_start_time, start_time)
        union = mint.max(candidates_end_time, end_time) - mint.min(candidates_start_time, start_time)
        iou = 1 - inter.clamp(min=0) / union

        return iou

    def loss_distance(self, start_time, end_time, candidates_start_time, candidates_end_time, duration):
        """
        Measure the distance of mid points.
        """
        mid_candidates = mint.div(mint.add(candidates_start_time, candidates_end_time), 2.0)
        mid_groundtruth = mint.div(mint.add(start_time, end_time), 2.0)
        distance_diff = mint.div(
            mint.max(mid_candidates, mid_groundtruth) - mint.min(mid_candidates, mid_groundtruth), duration
        ).clamp(min=0.2)

        return distance_diff

    def loss_duration(self, start_time, end_time, candidates_start_time, candidates_end_time, duration):
        """
        Measure the difference of duration.
        """
        duration_candidates = mint.sub(candidates_end_time, candidates_start_time)
        duration_groundtruth = mint.sub(end_time, start_time)
        duration_diff = mint.square(mint.div(mint.sub(duration_candidates, duration_groundtruth), duration))
        duration_diff = duration_diff.clamp(min=0.4)

        return duration_diff

    def construct(self, logits, labels):
        """
        This performs the loss computation.

        Args:
            logits (`ms.Tensor`):
                The output logits of head module.
            labels (`List[ms.Tensor]`):
                List of tensors ([start, end, duration]), which contains start time, end time of the video corresponding to the text, and also the duration.
        """
        duration, start_time, end_time = labels
        candidates = mint.mul(logits, duration)
        candidates_start_time, candidates_end_time = candidates[:, 0].float(), candidates[:, 1].float()

        losses_dict = {}
        for loss in self.losses:
            losses_dict.update(
                {loss: self.loss_map[loss](start_time, end_time, candidates_start_time, candidates_end_time, duration)}
            )

        return losses_dict


class TvpVisionModel(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.backbone = load_backbone(config)

        if config.backbone_config is not None:
            in_channels = config.backbone_config.hidden_sizes[-1]
        elif hasattr(self.backbone, "config") and hasattr(self.backbone.config, "hidden_sizes"):
            in_channels = self.backbone.config.hidden_sizes[-1]
        elif hasattr(self.backbone, "config") and hasattr(self.backbone.config, "hidden_size"):
            in_channels = self.backbone.config.hidden_size
        else:
            raise ValueError("Backbone config not found")

        self.grid_encoder_conv = mint.nn.Conv2d(
            in_channels,
            config.hidden_size,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        )

    def construct(self, pixel_values):
        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        # (batch_size * num_frames, num_channels, height, width)
        pixel_values = pixel_values.view(batch_size * num_frames, num_channels, height, width)
        grid_feat_outputs = self.backbone(pixel_values.float())["feature_maps"][0]  # not support bf16
        grid = self.grid_encoder_conv(grid_feat_outputs)
        grid = mint.nn.functional.max_pool2d(grid, kernel_size=2, stride=2)
        grid = mint.nn.functional.relu(grid, inplace=True)
        new_channel, new_height, new_width = grid.shape[-3:]
        # (batch_size, num_frames, num_channels, height, width)
        grid = grid.view(batch_size, num_frames, new_channel, new_height, new_width)
        # (batch_size, num_frames, height, width, num_channels)
        grid = grid.permute(0, 1, 3, 4, 2)
        return grid.to(pixel_values.dtype)


class TvpVisualInputEmbedding(nn.Cell):
    """
    Takes input of both image and video (multi-frame)
    """

    def __init__(self, config):
        super().__init__()
        # sequence embedding
        self.position_embeddings = mint.nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.row_position_embeddings = mint.nn.Embedding(config.max_grid_row_position_embeddings, config.hidden_size)
        self.col_position_embeddings = mint.nn.Embedding(config.max_grid_col_position_embeddings, config.hidden_size)
        self.token_type_embeddings = mint.nn.Embedding(1, config.hidden_size)
        self.layer_norm = mint.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = mint.nn.Dropout(config.hidden_dropout_prob)
        self.max_grid_row_position_embeddings = config.max_grid_row_position_embeddings
        self.max_grid_col_position_embeddings = config.max_grid_col_position_embeddings

    def interpolate_pos_encoding(self, embedding: ms.Tensor, height: int, width: int) -> ms.Tensor:
        """
        This method allows to interpolate the pre-trained pad weights , to be able to use the model on collection of high
        resolution images (high resolution videos).

        """
        h0 = w0 = 1
        # if height dimension is to be interpolated
        if height > self.max_grid_row_position_embeddings:
            h0 = height / self.max_grid_row_position_embeddings
        # if width dimension is to be interpolated
        if width > self.max_grid_col_position_embeddings:
            w0 = width / self.max_grid_col_position_embeddings
        embedding = embedding.permute(0, 3, 1, 2)  # (batch_size, hidden_dim, height, width)
        embedding = mint.nn.functional.interpolate(
            embedding,
            scale_factor=(h0, w0),
            mode="bicubic",
            align_corners=False,
        )
        embedding = embedding.permute(0, 2, 3, 1)  # (batch_size, height, width, hidden_dim)
        return embedding

    def add_2d_positional_embeddings(self, grid, interpolate_pos_encoding: bool = False):
        """
        Args:
            grid: (batch_size, height, width, hidden_dim)
            interpolate_pos_encoding: (`bool`, *optional*, defaults to `False`):
                Whether to interpolate the pre-trained position encodings.
        Returns:
            grid + col_position_embeddings.view(*col_shape): (batch_size, *, height, width, hidden_dim)
        """
        batch_size, height, width, hidden_dim = grid.shape

        # add row-wise position embeddings
        # (height, )
        row_height = min(self.max_grid_row_position_embeddings, height)
        row_position_ids = mint.arange(row_height, dtype=ms.int64)
        # (height, hidden_dim)
        row_position_embeddings = self.row_position_embeddings(row_position_ids)
        row_shape = (1,) * (len(grid.shape) - 3) + (row_height, 1, hidden_dim)
        # (batch_size, height, 1, hidden_dim)
        row_position_embeddings = row_position_embeddings.view(*row_shape)

        # add column-wise position embeddings
        row_width = min(self.max_grid_col_position_embeddings, width)
        col_position_ids = mint.arange(row_width, dtype=ms.int64)
        # (width, hidden_dim)
        col_position_embeddings = self.col_position_embeddings(col_position_ids)
        col_shape = (batch_size, 1, row_width, hidden_dim)
        # (batch_size, 1, width, hidden_dim)
        col_position_embeddings = col_position_embeddings.view(*col_shape)
        # (batch_size, height, width, hidden_dim)
        positional_embeddings = row_position_embeddings + col_position_embeddings

        # This interpolation gets triggered ONLY when the input image dim is larger in any dimenstion than the original position embeddings
        if interpolate_pos_encoding and (
            height > self.max_grid_row_position_embeddings or width > self.max_grid_col_position_embeddings
        ):
            grid = grid + self.interpolate_pos_encoding(positional_embeddings, height, width)
        else:
            grid = grid + positional_embeddings
        return grid

    def construct(self, grid, interpolate_pos_encoding: bool = False):
        """
        Args:
            grid: Array of shape (batch_size, num_frames, height, width, num_channels).
                It contains processed frames extracted from videos, and is generated by Tvp image preprocessor. Note,
                num_frames can be 1
            interpolate_pos_encoding: (bool, *optional*, defaults to `False`):
                Whether to interpolate the pre-trained position encodings.

        Returns:
            embeddings: The embedding of grid with size (batch_size, height*width, num_channels)

        """
        batch_size, num_frames, height, width, num_channels = grid.shape
        # temporal mean pooling, (batch_size, height, width, hidden_size)
        grid = grid.mean(1)
        grid = self.add_2d_positional_embeddings(grid, interpolate_pos_encoding=interpolate_pos_encoding)
        # image token sequence, (batch_size, height*width, num_channels)
        visual_tokens = grid.view(batch_size, -1, num_channels)
        visual_tokens_shape = visual_tokens.shape[:-1]

        # image token type embeddings.
        token_type_ids = mint.zeros(visual_tokens_shape, dtype=ms.int64)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = visual_tokens + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class TvpTextInputEmbeddings(nn.Cell):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = mint.nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = mint.nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = mint.nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.layer_norm = mint.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = mint.nn.Dropout(config.hidden_dropout_prob)

    def construct(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = mint.arange(seq_length, dtype=ms.int64)
            position_ids = position_ids.unsqueeze(0).broadcast_to(input_shape)
        if token_type_ids is None:
            token_type_ids = mint.zeros(input_shape, dtype=ms.int64)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class TvpAttention(nn.Cell):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of the number of attention heads {config.num_attention_heads}"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = mint.nn.Linear(config.hidden_size, self.all_head_size)
        self.key = mint.nn.Linear(config.hidden_size, self.all_head_size)
        self.value = mint.nn.Linear(config.hidden_size, self.all_head_size)
        self.attn_dropout = mint.nn.Dropout(config.attention_probs_dropout_prob)

        self.dense = mint.nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = mint.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = mint.nn.Dropout(config.hidden_dropout_prob)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = mint.ones((self.num_attention_heads, self.attention_head_size))
        heads = set(heads) - self.pruned_heads  # Convert to set and remove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = mint.arange(len(mask))[mask].long()

        # Prune linear layers
        self.query = prune_linear_layer(self.query, index)
        self.key = prune_linear_layer(self.key, index)
        self.value = prune_linear_layer(self.value, index)
        self.dense = prune_linear_layer(self.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.num_attention_heads = self.num_attention_heads - len(heads)
        self.all_head_size = self.attention_head_size * self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def _reshape(self, tensor: ms.Tensor, sequence_length: int, batch_size: int):
        return (
            tensor.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size)
            .swapaxes(1, 2)
            .contiguous()
        )

    def construct(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions: Optional[bool] = None,
    ):
        batch_size, sequence_length = hidden_states.shape[:2]
        mixed_query_layer = self.query(hidden_states)

        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self._reshape(mixed_query_layer, sequence_length, batch_size)
        key_layer = self._reshape(mixed_key_layer, sequence_length, batch_size)
        value_layer = self._reshape(mixed_value_layer, sequence_length, batch_size)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = mint.matmul(query_layer, key_layer.swapaxes(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = mint.nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attn_dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        attn_output = mint.matmul(attention_probs, value_layer)
        attn_output = attn_output.swapaxes(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, sequence_length, self.all_head_size)

        attn_output = self.dense(attn_output)
        attn_output = self.dropout(attn_output)
        attn_output = self.layer_norm(attn_output + hidden_states)
        # add attentions if we output them
        outputs = (attn_output, attention_probs) if output_attentions else (attn_output,)
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->Tvp
class TvpIntermediate(nn.Cell):
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


class TvpOutputLayer(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.dense = mint.nn.Linear(config.intermediate_size, config.hidden_size)
        self.layer_norm = mint.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = mint.nn.Dropout(config.hidden_dropout_prob)

    def construct(self, hidden_states: ms.Tensor, input_tensor: ms.Tensor) -> ms.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class TvpEncodeLayer(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.attention = TvpAttention(config)
        self.intermediate = TvpIntermediate(config)
        self.output = TvpOutputLayer(config)

    def construct(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions: Optional[bool] = None,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


class TvpEncoder(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.CellList([TvpEncodeLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def construct(
        self,
        hidden_states,
        attention_mask=None,
        head_mask: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        all_hidden_states = ()
        all_attentions = ()

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    (head_mask[i] if head_mask is not None else None),
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], output_attentions)

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            outputs = (hidden_states,)
            if output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if output_attentions:
                outputs = outputs + (all_attentions,)
            return outputs  # last-layer hidden state, (all hidden states), (all attentions)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states if output_hidden_states else None,
            attentions=all_attentions if output_attentions else None,
        )


# Copied from transformers.models.bert.modeling_bert.BertPooler with Bert->Tvp
class TvpPooler(nn.Cell):
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


class TvpPreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = TvpConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (mint.nn.Linear, mint.nn.Embedding)):
            normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, mint.nn.LayerNorm):
            zeros_(module.bias)
            ones_(module.weight)

        if isinstance(module, mint.nn.Linear) and module.bias is not None:
            zeros_(module.bias)

        if isinstance(module, mint.nn.Conv2d):
            module.weight.set_data(
                initializer(HeNormal(mode="fan_out", nonlinearity="relu"), module.weight.shape, module.weight.dtype)
            )
            if module.bias is not None:
                zeros_(module.bias)


TVP_START_DOCSTRING = r"""
    This model is a MindSpore [nn.Cell](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Cell.html) subclass. Use it
    as a regular MindSpore Module and refer to the MindSpore documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`TvpConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

TVP_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`ms.Tensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Indices can be obtained using [`AutoTokenizer`]. See
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details. [What are input
            IDs?](../glossary#input-ids)

        pixel_values (`ms.Tensor` of shape `(batch_size, num_frames, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`TvpImageProcessor`]. See [`TvpImageProcessor.__call__`]
            for details.

        attention_mask (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)

        head_mask (`ms.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.

        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        interpolate_pos_encoding (`bool`, *optional*, defaults to `False`):
            Whether to interpolate the pre-trained image pad prompter encodings and positional encodings.
"""


class TvpFrameDownPadPrompter(nn.Cell):
    """
    Pad frames extracted from videos only at the bottom.
    """

    def __init__(self, config):
        if config.visual_prompter_apply not in ("add", "replace", "remove"):
            raise ValueError("`visual_prompter_apply` must be in (add, replace, remove)")

        super().__init__()
        self.visual_prompt_size = config.visual_prompt_size
        self.frame_num = config.frame_num
        self.max_img_size = config.max_img_size
        self.visual_prompter_apply = config.visual_prompter_apply

        self.pad_down = ms.Parameter(
            mint.randn([1, config.frame_num, 3, config.visual_prompt_size, config.max_img_size])
        )

    def construct(self, pixel_values):
        if self.visual_prompter_apply != "add":
            visual_prompt_mask = mint.ones([self.max_img_size, self.max_img_size], dtype=pixel_values.dtype)
            visual_prompt_mask[self.max_img_size - self.visual_prompt_size : self.max_img_size, :] = 0.0
            pixel_values *= visual_prompt_mask
        if self.visual_prompter_apply != "remove":
            prompt = mint.zeros(
                [pixel_values.shape[0], pixel_values.shape[1], 3, self.max_img_size, self.max_img_size],
            )
            start_point = self.max_img_size - self.visual_prompt_size
            prompt[:, :, :, start_point : self.max_img_size, :] = self.pad_down
            pixel_values += prompt.to(pixel_values.dtype)
        return pixel_values


class TvpFramePadPrompter(nn.Cell):
    """
    Pad frames extracted from videos in the surroundings.
    """

    def __init__(self, config):
        if config.visual_prompter_apply not in ("add", "replace", "remove"):
            raise ValueError("`visual_prompter_apply` must be in (add, replace, remove)")

        super().__init__()
        self.num_frames = config.num_frames
        self.max_img_size = config.max_img_size
        self.visual_prompter_apply = config.visual_prompter_apply
        self.base_size = config.max_img_size - config.visual_prompt_size * 2
        self.pad_up = ms.Parameter(
            mint.randn([1, config.num_frames, 3, config.visual_prompt_size, config.max_img_size])
        )
        self.pad_down = ms.Parameter(
            mint.randn([1, config.num_frames, 3, config.visual_prompt_size, config.max_img_size])
        )
        self.pad_left = ms.Parameter(
            mint.randn(
                [
                    1,
                    config.num_frames,
                    3,
                    config.max_img_size - config.visual_prompt_size * 2,
                    config.visual_prompt_size,
                ]
            )
        )
        self.pad_right = ms.Parameter(
            mint.randn(
                [
                    1,
                    config.num_frames,
                    3,
                    config.max_img_size - config.visual_prompt_size * 2,
                    config.visual_prompt_size,
                ]
            )
        )

    def interpolate_pad_encoding(self, prompt: ms.Tensor, height: int, width: int) -> ms.Tensor:
        """
        This method allows to interpolate the pre-trained pad weights, to be able to use the model on collection of high
        resolution images (high resolution videos).

        """

        # creates scale factor from height and width of original image wrt to the config.max_img_size
        h0, w0 = height / self.max_img_size, width / self.max_img_size

        batch, num_frames, channels, prompt_height, prompt_width = prompt.shape

        # reshaping the batch and num_frames dimension into a single one (i.e (b,frames,c,h,w)-->(b*frames,c,h,w)), to apply bicubic interpolation
        prompt = prompt.reshape(batch * num_frames, channels, prompt_height, prompt_width)
        prompt = mint.nn.functional.interpolate(
            prompt,
            scale_factor=(h0, w0),
            mode="bicubic",
            align_corners=False,
        )
        # reversing back to (batch,frames,channels,height,width), where height and width is the new interpolated height and width
        prompt = prompt.reshape(batch, num_frames, channels, height, width)
        return prompt

    def construct(self, pixel_values, interpolate_pad_encoding: bool = False):
        height, width = (
            (pixel_values.shape[-2], pixel_values.shape[-1])
            if interpolate_pad_encoding
            else (self.max_img_size, self.max_img_size)
        )
        if self.visual_prompter_apply not in ("add", "remove", "replace"):
            raise ValueError(f"Invalid visual_prompter_apply value {self.visual_prompter_apply}")
        if self.visual_prompter_apply in ("replace", "remove"):
            visual_prompt_mask = mint.ones([height, width], dtype=pixel_values.dtype)
            pixel_values *= visual_prompt_mask
        if self.visual_prompter_apply in ("replace", "add"):
            base = mint.zeros((1, self.num_frames, 3, self.base_size, self.base_size), dtype=self.pad_left.dtype)

            prompt = mint.cat([self.pad_left, base, self.pad_right], dim=4)
            prompt = mint.cat([self.pad_up, prompt, self.pad_down], dim=3)
            prompt = mint.cat(pixel_values.shape[0] * [prompt])
            if interpolate_pad_encoding:
                prompt = self.interpolate_pad_encoding(prompt, height, width)
            pixel_values = pixel_values + prompt.to(pixel_values.dtype)
        return pixel_values


TVP_PROMPTER_CLASSES_MAPPING = {
    "framedownpad": TvpFrameDownPadPrompter,
    "framepad": TvpFramePadPrompter,
}


@add_start_docstrings(
    "The bare Tvp Model transformer outputting BaseModelOutputWithPooling object without any specific head on" " top.",
    TVP_START_DOCSTRING,
)
class TvpModel(TvpPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.vision_model = TvpVisionModel(config)
        self.embeddings = TvpTextInputEmbeddings(config)
        self.visual_embeddings = TvpVisualInputEmbedding(config)
        self.encoder = TvpEncoder(config)
        self.pooler = TvpPooler(config)
        self.text_prompt = ms.Parameter(mint.randn([1, 10, config.hidden_size]))
        self.dropout = mint.nn.Dropout(config.hidden_dropout_prob)
        if config.visual_prompter_type not in TVP_PROMPTER_CLASSES_MAPPING:
            raise ValueError("`visual_prompter_type` must be in (framedownpad, framepad)")
        self.visual_prompter = TVP_PROMPTER_CLASSES_MAPPING[config.visual_prompter_type](config)

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(TVP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=TvpConfig)
    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        pixel_values: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ):
        r"""
        Returns:

        Examples:
        ```python
        >>> import mindspore as ms
        >>> from mindspore import mint
        >>> from transformers import AutoTokenizer
        >>> from mindone.transformers import TvpModel

        >>> model = TvpModel.from_pretrained("Jiqing/tiny-random-tvp")

        >>> tokenizer = AutoTokenizer.from_pretrained("Jiqing/tiny-random-tvp")

        >>> pixel_values = mint.rand(1, 1, 3, 448, 448)
        >>> text_inputs = tokenizer("This is an example input", return_tensors="np")
        >>> output = model(ms.tensor(text_inputs.input_ids), pixel_values, ms.tensor(text_inputs.attention_mask))
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        # Add visual prompt, it compensates for the spatiotemporal information loss in 2D visual features.
        pixel_values = self.vision_model(
            self.visual_prompter(pixel_values, interpolate_pad_encoding=interpolate_pos_encoding)
        )
        # (batch_size, sequence_length, hidden_size)
        text_embedding_output = self.embeddings(input_ids=input_ids)
        # (batch_size, visual_sequence_length, hidden_size)
        visual_embedding_output = self.visual_embeddings(
            pixel_values, interpolate_pos_encoding=interpolate_pos_encoding
        )

        if attention_mask is not None:
            # (batch_size, visual_sequence_length)
            visual_attention_mask = attention_mask.new_ones(visual_embedding_output.shape[:2])
            pt_mask = mint.ones((attention_mask.shape[0], 10)).to(dtype=attention_mask.dtype)
            attention_mask = mint.cat([pt_mask, attention_mask, visual_attention_mask], dim=-1)
            # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
            # ourselves in which case we just need to make it broadcastable to all heads.
            attention_mask = self.get_extended_attention_mask(attention_mask, input_ids.shape)
        text_prompt = self.text_prompt.broadcast_to((text_embedding_output.shape[0], -1, -1))
        # (batch_size, sequence_length + visual_sequence_length, hidden_size)
        embedding_output = mint.cat([text_prompt, text_embedding_output, visual_embedding_output], dim=1)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            head_mask=self.get_head_mask(head_mask, self.config.num_hidden_layers),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = encoder_outputs.last_hidden_state if return_dict else encoder_outputs[0]
        pooled_output = self.pooler(last_hidden_state)
        last_hidden_state = self.dropout(last_hidden_state)
        pooled_output = self.dropout(pooled_output)
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class TvpVideoGroundingHead(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.layer_0 = mint.nn.Linear(config.hidden_size, config.hidden_size * 2)
        self.layer_1 = mint.nn.Linear(config.hidden_size * 2, 2)
        self.activation_0 = mint.nn.ReLU()
        self.activation_1 = mint.nn.Sigmoid()

    def construct(self, pooler_output):
        logits = self.activation_0(self.layer_0(pooler_output))
        logits = self.activation_1(self.layer_1(logits))
        return logits


@add_start_docstrings(
    """
    Tvp Model with a video grounding head on top computing IoU, distance, and duration loss.
    """,
    TVP_START_DOCSTRING,
)
class TvpForVideoGrounding(TvpPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = TvpModel(config)
        self.video_grounding_head = TvpVideoGroundingHead(config)

        self.post_init()

    @add_start_docstrings_to_model_forward(TVP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TvpVideoGroundingOutput, config_class=TvpConfig)
    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        pixel_values: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        labels: Tuple[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ):
        r"""
        labels (`ms.Tensor` of shape `(batch_size, 3)`, *optional*):
            The labels contains duration, start time, and end time of the video corresponding to the text.
        Returns:

        Examples:
        ```python
        >>> import mindspore as ms
        >>> import mindspore.mint as mint
        >>> from transformers import AutoTokenizer
        >>> from mindone.transformers import TvpForVideoGrounding

        >>> model = TvpForVideoGrounding.from_pretrained("Jiqing/tiny-random-tvp")

        >>> tokenizer = AutoTokenizer.from_pretrained("Jiqing/tiny-random-tvp")

        >>> pixel_values = mint.rand(1, 1, 3, 448, 448)
        >>> text_inputs = tokenizer("This is an example input", return_tensors="np")
        >>> output = model(ms.tensor(text_inputs.input_ids), pixel_values, ms.tensor(text_inputs.attention_mask))
        ```

        ```python
        >>> import av
        >>> import cv2
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from huggingface_hub import hf_hub_download
        >>> from mindone.transformers import AutoProcessor, TvpForVideoGrounding

        >>> def pyav_decode(container, sampling_rate, num_frames, clip_idx, num_clips, target_fps):
        ...     # Convert the video from its original fps to the target_fps and decode the video with PyAV decoder.
        ...     video = container.streams.video[0]
        ...     fps = float(video.average_rate)
        ...     clip_size = sampling_rate * num_frames / target_fps * fps
        ...     delta = max(num_frames - clip_size, 0)
        ...     start_idx = delta * clip_idx / num_clips
        ...     end_idx = start_idx + clip_size - 1
        ...     timebase = video.duration / num_frames
        ...     video_start_pts = int(start_idx * timebase)
        ...     video_end_pts = int(end_idx * timebase)
        ...     seek_offset = max(video_start_pts - 1024, 0)
        ...     container.seek(seek_offset, any_frame=False, backward=True, stream=video)
        ...     frames = {}
        ...     for frame in container.decode(video=0):
        ...         if frame.pts < video_start_pts:
        ...             continue
        ...         frames[frame.pts] = frame
        ...         if frame.pts > video_end_pts:
        ...             break
        ...     frames = [frames[pts] for pts in sorted(frames)]
        ...     return frames, fps


        >>> def decode(container, sampling_rate, num_frames, clip_idx, num_clips, target_fps):
        ...     # Decode the video and perform temporal sampling.
        ...     assert clip_idx >= -2, "Not a valid clip_idx {}".format(clip_idx)
        ...     frames, fps = pyav_decode(container, sampling_rate, num_frames, clip_idx, num_clips, target_fps)
        ...     clip_size = sampling_rate * num_frames / target_fps * fps
        ...     index = np.linspace(0, clip_size - 1, num_frames)
        ...     index = np.clip(index, 0, len(frames) - 1).astype(np.int64)
        ...     frames = np.array([frames[idx].to_rgb().to_ndarray() for idx in index])
        ...     frames = frames.transpose(0, 3, 1, 2)
        ...     return frames


        >>> file = hf_hub_download(repo_id="Intel/tvp_demo", filename="AK2KG.mp4", repo_type="dataset")
        >>> model = TvpForVideoGrounding.from_pretrained("Intel/tvp-base")

        >>> decoder_kwargs = dict(
        ...     container=av.open(file, metadata_errors="ignore"),
        ...     sampling_rate=1,
        ...     num_frames=model.config.num_frames,
        ...     clip_idx=0,
        ...     num_clips=1,
        ...     target_fps=3,
        ... )
        >>> raw_sampled_frms = decode(**decoder_kwargs)

        >>> text = "a person is sitting on a bed."
        >>> processor = AutoProcessor.from_pretrained("Intel/tvp-base")
        >>> model_inputs = processor(
        ...     text=[text], videos=list(raw_sampled_frms), return_tensors="np", max_text_length=100,
        ... )
        >>> for k, v in model_inputs.items():
        ...     model_inputs[k] = ms.tensor(v)
        ...     if model_inputs[k].dtype == ms.int64:
        ...         model_inputs[k] = model_inputs[k].to(ms.int32)
        ...     else:
        ...         model_inputs[k] = model_inputs[k].to(model.dtype)
        >>> output = model(**model_inputs)

        >>> def get_video_duration(filename):
        ...     cap = cv2.VideoCapture(filename)
        ...     if cap.isOpened():
        ...         rate = cap.get(5)
        ...         frame_num = cap.get(7)
        ...         duration = frame_num/rate
        ...         return duration
        ...     return -1

        >>> duration = get_video_duration(file)
        >>> start, end = processor.post_process_video_grounding(output.logits.asnumpy(), duration)

        >>> print(f"The time slot of the video corresponding to the text \"{text}\" is from {start}s to {end}s")
        The time slot of the video corresponding to the text "a person is sitting on a bed." is from 0.0s to 6.8s
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        outputs = self.model(
            input_ids,
            pixel_values,
            attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        pooler_output = outputs[1]
        logits = self.video_grounding_head(pooler_output)

        loss = None
        if labels is not None:
            criterion = TvpLoss(["iou", "distance", "duration"])
            loss_dict = criterion(logits, labels)
            loss = (
                loss_dict["iou"]
                + self.config.distance_loss_weight * loss_dict["distance"]
                + self.config.duration_loss_weight * loss_dict["duration"]
            )
        if not return_dict:
            outputs = (logits,) + outputs[2:]
            if loss is not None:
                outputs = (loss,) + outputs
            return outputs

        return TvpVideoGroundingOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = ["TvpModel", "TvpPreTrainedModel", "TvpForVideoGrounding"]
