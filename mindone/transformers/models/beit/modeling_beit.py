# coding=utf-8
# Copyright 2021 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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
"""MindSpore BEiT model."""

import collections.abc
import math
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from transformers.models.beit.configuration_beit import BeitConfig

import mindspore as ms
from mindspore import Tensor, mint, nn
from mindspore.mint.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...mindspore_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...modeling_outputs import (
    BackboneOutput,
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
    MaskedLMOutput,
    SemanticSegmenterOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from ...utils.backbone_utils import BackboneMixin

logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "BeitConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "microsoft/beit-base-patch16-224-pt22k"
_EXPECTED_OUTPUT_SHAPE = [1, 197, 768]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "microsoft/beit-base-patch16-224"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"


# modified from mindone.transformers.utils.generic.mindspore_int for adapting graph mode
def mindspore_int(x):
    """
    Casts an input to a mindspore int64 tensor if we are in a tracing context, otherwise to a Python int.
    """

    return x.to(ms.int64) if isinstance(x, ms.Tensor) else int(x)


@dataclass
class BeitModelOutputWithPooling(BaseModelOutputWithPooling):
    """
    Class for outputs of [`BeitModel`].

    Args:
        last_hidden_state (`ms.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`ms.Tensor` of shape `(batch_size, hidden_size)`):
            Average of the last layer hidden states of the patch tokens (excluding the *[CLS]* token) if
            *config.use_mean_pooling* is set to True. If set to False, then the final hidden state of the *[CLS]* token
            will be returned.
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


def drop_path(input: ms.Tensor, drop_prob: float = 0.0, training: bool = False) -> ms.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + mint.rand(shape, dtype=input.dtype)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


class BeitDropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


# Based on timm implementation, which can be found here:
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
class BeitEmbeddings(nn.Cell):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.

    """

    def __init__(self, config: BeitConfig) -> None:
        super().__init__()

        self.cls_token = ms.Parameter(mint.zeros((1, 1, config.hidden_size)), name="cls_token")
        if config.use_mask_token:
            self.mask_token = ms.Parameter(mint.zeros((1, 1, config.hidden_size)), name="mask_token")
        else:
            self.mask_token = None
        self.patch_embeddings = BeitPatchEmbeddings(config)
        self.patch_size = config.patch_size
        self.image_size = (
            config.image_size
            if isinstance(config.image_size, collections.abc.Iterable)
            else (config.image_size, config.image_size)
        )
        num_patches = self.patch_embeddings.num_patches
        if config.use_absolute_position_embeddings:
            self.position_embeddings = ms.Parameter(
                mint.zeros((1, num_patches + 1, config.hidden_size)), name="position_embeddings"
            )
        else:
            self.position_embeddings = None
        self.dropout = mint.nn.Dropout(config.hidden_dropout_prob)

    # Copied from transformers.models.vit.modeling_vit.ViTEmbeddings.interpolate_pos_encoding
    def interpolate_pos_encoding(self, embeddings: ms.Tensor, height: int, width: int) -> ms.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """

        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if num_patches == num_positions and height == width:
            return self.position_embeddings

        class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]

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

    def construct(
        self,
        pixel_values: ms.Tensor,
        bool_masked_pos: Optional[ms.Tensor] = None,
        interpolate_pos_encoding: Optional[bool] = None,
    ) -> ms.Tensor:
        if self.position_embeddings is not None and interpolate_pos_encoding is not None:
            warnings.warn(
                "`interpolate_pos_encoding` argument has no effect for BEiTEmbeddings, embeddings are always "
                "interpolated to the input image size. The argument will be removed in transformers v4.51.0."
            )

        _, _, height, width = pixel_values.shape
        embeddings, (patch_height, patch_width) = self.patch_embeddings(pixel_values)
        batch_size, seq_len, _ = embeddings.shape

        if bool_masked_pos is not None:
            mask_tokens = self.mask_token.broadcast_to((batch_size, seq_len, -1))
            # replace the masked visual tokens by mask_tokens
            w = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1 - w) + mask_tokens * w

        cls_tokens = self.cls_token.broadcast_to((batch_size, -1, -1))
        embeddings = mint.cat((cls_tokens, embeddings), dim=1)

        if self.position_embeddings is not None:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)

        embeddings = self.dropout(embeddings)

        return embeddings, (patch_height, patch_width)


class BeitPatchEmbeddings(nn.Cell):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        patch_shape = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.patch_shape = patch_shape

        self.projection = mint.nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def construct(self, pixel_values: ms.Tensor) -> ms.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )

        embeddings = self.projection(pixel_values)
        patch_height, patch_width = embeddings.shape[2], embeddings.shape[3]
        embeddings = embeddings.flatten(2).swapaxes(1, 2)

        return embeddings, (patch_height, patch_width)


class BeitSelfAttention(nn.Cell):
    def __init__(self, config: BeitConfig, window_size: Optional[tuple] = None) -> None:
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = mint.nn.Linear(config.hidden_size, self.all_head_size)
        self.key = mint.nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.value = mint.nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = mint.nn.Dropout(config.attention_probs_dropout_prob)

        self.has_relative_position_bias = bool(window_size)
        if self.has_relative_position_bias:
            self.relative_position_bias = BeitRelativePositionBias(config, window_size=window_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def construct(
        self,
        hidden_states: ms.Tensor,
        head_mask: Optional[ms.Tensor] = None,
        output_attentions: bool = False,
        relative_position_bias: Optional[ms.Tensor] = None,
        interpolate_pos_encoding: bool = False,
        resolution: Optional[Tuple[int]] = None,
    ) -> Union[Tuple[ms.Tensor], Tuple[ms.Tensor, ms.Tensor]]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = mint.matmul(query_layer, key_layer.swapaxes(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Add relative position bias if present.
        if self.has_relative_position_bias:
            height, width = resolution
            window_size = (height // self.config.patch_size, width // self.config.patch_size)
            attention_scores = attention_scores + self.relative_position_bias(
                window_size, interpolate_pos_encoding, dim_size=hidden_states.shape[1]
            )

        # Add shared relative position bias if provided.
        if relative_position_bias is not None:
            attention_scores = attention_scores + relative_position_bias

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

        return outputs


class BeitSelfOutput(nn.Cell):
    """
    The residual connection is defined in BeitLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: BeitConfig) -> None:
        super().__init__()
        self.dense = mint.nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = mint.nn.Dropout(config.hidden_dropout_prob)

    def construct(self, hidden_states: ms.Tensor, input_tensor: ms.Tensor, gamma=None) -> ms.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


BEIT_SELF_ATTENTION_CLASSES = {
    "eager": BeitSelfAttention,
    "sdpa": NotImplementedError,
}


class BeitAttention(nn.Cell):
    def __init__(self, config: BeitConfig, window_size: Optional[tuple] = None) -> None:
        super().__init__()
        self.attention = BEIT_SELF_ATTENTION_CLASSES[config._attn_implementation](config, window_size=window_size)
        self.output = BeitSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def construct(
        self,
        hidden_states: ms.Tensor,
        head_mask: Optional[ms.Tensor] = None,
        output_attentions: bool = False,
        relative_position_bias: Optional[ms.Tensor] = None,
        interpolate_pos_encoding: bool = False,
        resolution: Optional[Tuple[int]] = None,
    ) -> Union[Tuple[ms.Tensor], Tuple[ms.Tensor, ms.Tensor]]:
        self_outputs = self.attention(
            hidden_states, head_mask, output_attentions, relative_position_bias, interpolate_pos_encoding, resolution
        )

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BeitIntermediate(nn.Cell):
    def __init__(self, config: BeitConfig) -> None:
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


class BeitOutput(nn.Cell):
    def __init__(self, config: BeitConfig) -> None:
        super().__init__()
        self.dense = mint.nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = mint.nn.Dropout(config.hidden_dropout_prob)

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class BeitLayer(nn.Cell):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: BeitConfig, window_size: Optional[tuple] = None, drop_path_rate: float = 0.0) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BeitAttention(config, window_size=window_size)
        self.intermediate = BeitIntermediate(config)
        self.output = BeitOutput(config)
        self.layernorm_before = mint.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.drop_path = BeitDropPath(drop_path_rate) if drop_path_rate > 0.0 else mint.nn.Identity()
        self.layernorm_after = mint.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        init_values = config.layer_scale_init_value
        if init_values > 0:
            self.lambda_1 = ms.Parameter(
                init_values * mint.ones((config.hidden_size)), requires_grad=True, name="lambda_1"
            )
            self.lambda_2 = ms.Parameter(
                init_values * mint.ones((config.hidden_size)), requires_grad=True, name="lambda_2"
            )
        else:
            self.lambda_1, self.lambda_2 = None, None

    def construct(
        self,
        hidden_states: ms.Tensor,
        head_mask: Optional[ms.Tensor] = None,
        output_attentions: bool = False,
        relative_position_bias: Optional[ms.Tensor] = None,
        interpolate_pos_encoding: bool = False,
        resolution: Optional[Tuple[int]] = None,
    ) -> Union[Tuple[ms.Tensor], Tuple[ms.Tensor, ms.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in BEiT, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
            relative_position_bias=relative_position_bias,
            interpolate_pos_encoding=interpolate_pos_encoding,
            resolution=resolution,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # apply lambda_1 if present
        if self.lambda_1 is not None:
            attention_output = self.lambda_1 * attention_output

        # first residual connection
        hidden_states = self.drop_path(attention_output) + hidden_states

        # in BEiT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)

        layer_output = self.intermediate(layer_output)
        layer_output = self.output(layer_output)

        if self.lambda_2 is not None:
            layer_output = self.lambda_2 * layer_output

        # second residual connection
        layer_output = self.drop_path(layer_output) + hidden_states

        outputs = (layer_output,) + outputs

        return outputs


class BeitRelativePositionBias(nn.Cell):
    def __init__(self, config: BeitConfig, window_size: tuple) -> None:
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = ms.Parameter(
            mint.zeros((self.num_relative_distance, config.num_attention_heads)), name="relative_position_bias_table"
        )  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

    def generate_relative_position_index(self, window_size: Tuple[int, int]) -> ms.Tensor:
        """
        This method creates the relative position index, modified to support arbitrary window sizes,
        as introduced in [MiDaS v3.1](https://arxiv.org/abs/2307.14460).
        """
        num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        # cls to token & token 2 cls & cls to cls
        # get pair-wise relative position index for each token inside the window
        window_area = window_size[0] * window_size[1]
        grid = mint.meshgrid(mint.arange(window_size[0]), mint.arange(window_size[1]), indexing="ij")
        coords = mint.stack(grid)  # 2, Wh, Ww
        coords_flatten = mint.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = mint.zeros(size=(window_area + 1,) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = num_relative_distance - 3
        relative_position_index[0:, 0] = num_relative_distance - 2
        relative_position_index[0, 0] = num_relative_distance - 1
        return relative_position_index

    def construct(self, window_size, interpolate_pos_encoding: bool = False, dim_size=None) -> ms.Tensor:
        """
        Modification of timm.models.beit.py: Attention._get_rel_pos_bias to support arbitrary window sizes.
        """
        old_height = 2 * self.window_size[0] - 1
        old_width = 2 * self.window_size[1] - 1

        new_height = 2 * window_size[0] - 1
        new_width = 2 * window_size[1] - 1

        old_relative_position_bias_table = self.relative_position_bias_table

        old_num_relative_distance = self.num_relative_distance
        new_num_relative_distance = new_height * new_width + 3

        old_sub_table = old_relative_position_bias_table[: old_num_relative_distance - 3]

        old_sub_table = old_sub_table.reshape(1, old_width, old_height, -1).permute(0, 3, 1, 2)
        if old_sub_table.dtype == ms.bfloat16:
            new_sub_table = mint.nn.functional.interpolate(
                old_sub_table.half(), size=(mindspore_int(new_height), mindspore_int(new_width)), mode="bilinear"
            ).to(ms.bfloat16)
        else:
            new_sub_table = mint.nn.functional.interpolate(
                old_sub_table, size=(mindspore_int(new_height), mindspore_int(new_width)), mode="bilinear"
            )
        new_sub_table = new_sub_table.permute(0, 2, 3, 1).reshape(new_num_relative_distance - 3, -1)

        new_relative_position_bias_table = mint.cat(
            [new_sub_table, old_relative_position_bias_table[old_num_relative_distance - 3 :]]
        )

        relative_position_index = self.generate_relative_position_index(window_size)
        relative_position_bias = new_relative_position_bias_table[relative_position_index.view(-1)]

        # patch_size*num_patches_height, patch_size*num_patches_width, num_attention_heads
        relative_position_bias = relative_position_bias.view(
            window_size[0] * window_size[1] + 1, window_size[0] * window_size[1] + 1, -1
        )
        # num_attention_heads, patch_size*num_patches_width, patch_size*num_patches_height
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()

        if interpolate_pos_encoding:
            relative_position_bias = mint.nn.functional.interpolate(
                relative_position_bias.unsqueeze(1),
                size=(dim_size, dim_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        return relative_position_bias.unsqueeze(0)


class BeitEncoder(nn.Cell):
    def __init__(self, config: BeitConfig, window_size: Optional[tuple] = None) -> None:
        super().__init__()
        self.config = config
        self.has_relative_position_bias = config.use_shared_relative_position_bias
        if self.has_relative_position_bias:
            self.relative_position_bias = BeitRelativePositionBias(config, window_size=window_size)

        # stochastic depth decay rule
        dpr = [x.item() for x in mint.linspace(0, config.drop_path_rate, config.num_hidden_layers)]
        self.layer = nn.CellList(
            [
                BeitLayer(
                    config,
                    window_size=window_size if config.use_relative_position_bias else None,
                    drop_path_rate=dpr[i],
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.gradient_checkpointing = False

    def construct(
        self,
        hidden_states: ms.Tensor,
        head_mask: Optional[ms.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        interpolate_pos_encoding: bool = False,
        resolution: Optional[Tuple[int, int]] = None,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.has_relative_position_bias:
                height, width = resolution
                window_size = (height // self.config.patch_size, width // self.config.patch_size)
                relative_position_bias = self.relative_position_bias(
                    window_size, interpolate_pos_encoding=interpolate_pos_encoding, dim_size=hidden_states.shape[1]
                )
            else:
                relative_position_bias = None

            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states,
                layer_head_mask,
                output_attentions,
                relative_position_bias,
                interpolate_pos_encoding,
                resolution,
            )

            hidden_states = layer_outputs[0]

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


class BeitPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BeitConfig
    base_model_prefix = "beit"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["BeitLayer"]
    _keys_to_ignore_on_load_unexpected = [r".*relative_position_index.*"]
    _supports_sdpa = True

    def _init_weights(self, module):
        """Initialize the weights"""
        pass


BEIT_START_DOCSTRING = r"""
    This model is a MindSpore
    [ms.nn.Cell](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Cell.html) subclass. Use it
    as a regular MindSpore Cell and refer to the MindSpore documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`BeitConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

BEIT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`ms.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`BeitImageProcessor.__call__`] for details.

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
        interpolate_pos_encoding (`bool`, *optional*, defaults to `False`):
            Whether to interpolate the pre-trained position encodings.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class BeitModel(BeitPreTrainedModel):
    def __init__(self, config: BeitConfig, add_pooling_layer: bool = True) -> None:
        super().__init__(config)
        self.config = config

        self.embeddings = BeitEmbeddings(config)
        self.encoder = BeitEncoder(config, window_size=self.embeddings.patch_embeddings.patch_shape)

        self.layernorm = (
            mint.nn.Identity()
            if config.use_mean_pooling
            else mint.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        )
        self.pooler = BeitPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def construct(
        self,
        pixel_values: ms.Tensor,
        bool_masked_pos: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, BeitModelOutputWithPooling]:
        r"""
        bool_masked_pos (`ms.Tensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output, _ = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)
        resolution = pixel_values.shape[2:]

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            resolution=resolution,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        return BeitModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class BeitPooler(nn.Cell):
    def __init__(self, config: BeitConfig) -> None:
        super().__init__()
        self.layernorm = (
            mint.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) if config.use_mean_pooling else None
        )

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        if self.layernorm is not None:
            # Mean pool the final hidden states of the patch tokens
            patch_tokens = hidden_states[:, 1:, :]
            pooled_output = self.layernorm(patch_tokens.mean(1))
        else:
            # Pool by simply taking the final hidden state of the [CLS] token
            pooled_output = hidden_states[:, 0]

        return pooled_output


class BeitForMaskedImageModeling(BeitPreTrainedModel):
    def __init__(self, config: BeitConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.beit = BeitModel(config, add_pooling_layer=False)

        # Classifier head
        self.layernorm = mint.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = mint.nn.Linear(config.hidden_size, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        pixel_values: Optional[ms.Tensor] = None,
        bool_masked_pos: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        labels: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, MaskedLMOutput]:
        r"""
        bool_masked_pos (`ms.Tensor` of shape `(batch_size, num_patches)`):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).

        labels (`ms.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from mindone.transformers import AutoImageProcessor, BeitForMaskedImageModeling
        >>> import mindspore as ms
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
        >>> model = BeitForMaskedImageModeling.from_pretrained("microsoft/beit-base-patch16-224-pt22k")

        >>> num_patches = (model.config.image_size // model.config.patch_size) ** 2
        >>> pixel_values = ms.tensor(image_processor(images=image, return_tensors="np").pixel_values)
        >>> # create random boolean mask of shape (batch_size, num_patches)
        >>> bool_masked_pos = mint.randint(low=0, high=2, size=(1, num_patches)).bool()

        >>> outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
        >>> loss, logits = outputs.loss, outputs.logits
        >>> list(logits.shape)
        [1, 196, 8192]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.beit(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.layernorm(sequence_output)
        prediction_scores = self.lm_head(sequence_output[:, 1:])

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores[bool_masked_pos], labels)

        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BeitForImageClassification(BeitPreTrainedModel):
    def __init__(self, config: BeitConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.beit = BeitModel(config, add_pooling_layer=True)

        # Classifier head
        self.classifier = (
            mint.nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else mint.nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        pixel_values: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        labels: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`ms.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.beit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        pooled_output = outputs.pooler_output if return_dict else outputs[1]

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

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BeitConvModule(nn.Cell):
    """
    A convolutional block that bundles conv/norm/activation layers. This block simplifies the usage of convolution
    layers, which are commonly used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).

    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int], str] = 0,
        bias: bool = False,
        dilation: Union[int, Tuple[int, int]] = 1,
    ) -> None:
        super().__init__()
        self.conv = mint.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            dilation=dilation,
        )
        self.bn = mint.nn.BatchNorm2d(out_channels)
        self.activation = mint.nn.ReLU()

    def construct(self, input: ms.Tensor) -> ms.Tensor:
        output = self.conv(input)
        output = self.bn(output)
        output = self.activation(output)

        return output


class BeitPyramidPoolingBlock(nn.Cell):
    def __init__(self, pool_scale: int, in_channels: int, channels: int) -> None:
        super().__init__()
        self.layers = [
            mint.nn.AdaptiveAvgPool2d(pool_scale),
            BeitConvModule(in_channels, channels, kernel_size=1),
        ]
        for i, layer in enumerate(self.layers):
            self.insert_child_to_cell(str(i), layer)

    def construct(self, input: ms.Tensor) -> ms.Tensor:
        hidden_state = input
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class BeitPyramidPoolingModule(nn.Cell):
    """
    Pyramid Pooling Module (PPM) used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        align_corners (bool): align_corners argument of F.interpolate.

    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """

    def __init__(self, pool_scales: Tuple[int, ...], in_channels: int, channels: int, align_corners: bool) -> None:
        super().__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.blocks = []
        for i, pool_scale in enumerate(pool_scales):
            block = BeitPyramidPoolingBlock(pool_scale=pool_scale, in_channels=in_channels, channels=channels)
            self.blocks.append(block)
            self.insert_child_to_cell(str(i), block)

    def construct(self, x: ms.Tensor) -> List[ms.Tensor]:
        ppm_outs = []
        for ppm in self.blocks:
            ppm_out = ppm(x)
            upsampled_ppm_out = mint.nn.functional.interpolate(
                ppm_out, size=x.shape[2:], mode="bilinear", align_corners=self.align_corners
            )
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


class BeitUperHead(nn.Cell):
    """
    Unified Perceptual Parsing for Scene Understanding. This head is the implementation of
    [UPerNet](https://arxiv.org/abs/1807.10221).

    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """

    def __init__(self, config: BeitConfig) -> None:
        super().__init__()

        self.pool_scales = config.pool_scales  # e.g. (1, 2, 3, 6)
        self.in_channels = [config.hidden_size] * 4  # e.g. [768, 768, 768, 768]
        self.channels = config.hidden_size
        self.align_corners = False
        self.classifier = mint.nn.Conv2d(self.channels, config.num_labels, kernel_size=1)

        # PSP Module
        self.psp_modules = BeitPyramidPoolingModule(
            self.pool_scales,
            self.in_channels[-1],
            self.channels,
            align_corners=self.align_corners,
        )
        self.bottleneck = BeitConvModule(
            self.in_channels[-1] + len(self.pool_scales) * self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
        )
        # FPN Module
        self.lateral_convs = []
        self.fpn_convs = []
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = BeitConvModule(in_channels, self.channels, kernel_size=1)
            fpn_conv = BeitConvModule(self.channels, self.channels, kernel_size=3, padding=1)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.lateral_convs = nn.CellList(self.lateral_convs)
        self.fpn_convs = nn.CellList(self.fpn_convs)

        self.fpn_bottleneck = BeitConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
        )

    def psp_forward(self, inputs):
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = mint.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def construct(self, encoder_hidden_states: ms.Tensor) -> ms.Tensor:
        # build laterals
        laterals = [lateral_conv(encoder_hidden_states[i]) for i, lateral_conv in enumerate(self.lateral_convs)]

        laterals.append(self.psp_forward(encoder_hidden_states))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + mint.nn.functional.interpolate(
                laterals[i], size=prev_shape, mode="bilinear", align_corners=self.align_corners
            )

        # build outputs
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = mint.nn.functional.interpolate(
                fpn_outs[i], size=fpn_outs[0].shape[2:], mode="bilinear", align_corners=self.align_corners
            )
        fpn_outs = mint.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)
        output = self.classifier(output)

        return output


class BeitFCNHead(nn.Cell):
    """
    Fully Convolution Networks for Semantic Segmentation. This head is implemented of
    [FCNNet](https://arxiv.org/abs/1411.4038>).

    Args:
        config (BeitConfig): Configuration.
        in_channels
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        dilation (int): The dilation rate for convs in the head. Default: 1.


    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """

    def __init__(
        self, config: BeitConfig, in_index: int = 2, kernel_size: int = 3, dilation: Union[int, Tuple[int, int]] = 1
    ) -> None:
        super().__init__()
        self.in_channels = config.hidden_size
        self.channels = config.auxiliary_channels
        self.num_convs = config.auxiliary_num_convs
        self.concat_input = config.auxiliary_concat_input
        self.in_index = in_index

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            BeitConvModule(
                self.in_channels, self.channels, kernel_size=kernel_size, padding=conv_padding, dilation=dilation
            )
        )
        for i in range(self.num_convs - 1):
            convs.append(
                BeitConvModule(
                    self.channels, self.channels, kernel_size=kernel_size, padding=conv_padding, dilation=dilation
                )
            )
        if self.num_convs == 0:
            self.convs = mint.nn.Identity()
        else:
            self.convs = nn.SequentialCell(*convs)
        if self.concat_input:
            self.conv_cat = BeitConvModule(
                self.in_channels + self.channels, self.channels, kernel_size=kernel_size, padding=kernel_size // 2
            )

        self.classifier = mint.nn.Conv2d(self.channels, config.num_labels, kernel_size=1)

    def construct(self, encoder_hidden_states: ms.Tensor) -> ms.Tensor:
        # just take the relevant feature maps
        hidden_states = encoder_hidden_states[self.in_index]
        output = self.convs(hidden_states)
        if self.concat_input:
            output = self.conv_cat(mint.cat([hidden_states, output], dim=1))
        output = self.classifier(output)
        return output


class BeitForSemanticSegmentation(BeitPreTrainedModel):
    def __init__(self, config: BeitConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.beit = BeitModel(config, add_pooling_layer=False)

        # FPNs
        if len(self.config.out_indices) != 4:
            raise ValueError(
                "BeitForSemanticSegmentation requires config.out_indices to be a list of 4 integers, "
                "specifying which features to use from the backbone. One can use [3, 5, 7, 11] in case of "
                "a base-sized architecture."
            )
        self.fpn1 = nn.SequentialCell(
            mint.nn.ConvTranspose2d(config.hidden_size, config.hidden_size, kernel_size=2, stride=2),
            mint.nn.BatchNorm2d(config.hidden_size),
            mint.nn.GELU(),
            mint.nn.ConvTranspose2d(config.hidden_size, config.hidden_size, kernel_size=2, stride=2),
        )
        self.fpn2 = nn.SequentialCell(
            mint.nn.ConvTranspose2d(config.hidden_size, config.hidden_size, kernel_size=2, stride=2),
        )
        self.fpn3 = mint.nn.Identity()
        self.fpn4 = mint.nn.MaxPool2d(kernel_size=2, stride=2)

        # Semantic segmentation head(s)
        self.decode_head = BeitUperHead(config)
        self.auxiliary_head = BeitFCNHead(config) if config.use_auxiliary_head else None

        # Initialize weights and apply final processing
        self.post_init()

    def compute_loss(self, logits, auxiliary_logits, labels):
        # upsample logits to the images' original size
        upsampled_logits = mint.nn.functional.interpolate(
            logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )
        if auxiliary_logits is not None:
            upsampled_auxiliary_logits = mint.nn.functional.interpolate(
                auxiliary_logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
        # compute weighted loss
        loss_fct = CrossEntropyLoss(ignore_index=self.config.semantic_loss_ignore_index)
        main_loss = loss_fct(upsampled_logits, labels)
        loss = main_loss
        if auxiliary_logits is not None:
            auxiliary_loss = loss_fct(upsampled_auxiliary_logits, labels)
            loss += self.config.auxiliary_loss_weight * auxiliary_loss

        return loss

    def construct(
        self,
        pixel_values: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        labels: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, SemanticSegmenterOutput]:
        r"""
        labels (`ms.Tensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from mindone.transformers import AutoImageProcessor, BeitForSemanticSegmentation
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("microsoft/beit-base-finetuned-ade-640-640")
        >>> model = BeitForSemanticSegmentation.from_pretrained("microsoft/beit-base-finetuned-ade-640-640")

        >>> inputs = image_processor(images=image, return_tensors="np")
        >>> inputs = {k: ms.tensor(v) for k, v in inputs.items()}
        >>> outputs = model(**inputs)
        >>> # logits are of shape (batch_size, num_labels, height, width)
        >>> logits = outputs.logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if labels is not None and self.config.num_labels == 1:
            raise ValueError("The number of labels should be greater than one")

        outputs = self.beit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,  # we need the intermediate hidden states
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]

        # only keep certain features, and reshape
        # note that we do +1 as the encoder_hidden_states also includes the initial embeddings
        features = [feature for idx, feature in enumerate(encoder_hidden_states) if idx + 1 in self.config.out_indices]
        batch_size = pixel_values.shape[0]
        patch_resolution = self.config.image_size // self.config.patch_size
        features = [
            x[:, 1:, :].permute(0, 2, 1).reshape(batch_size, -1, patch_resolution, patch_resolution) for x in features
        ]

        # apply FPNs
        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(features)):
            features[i] = ops[i](features[i])

        logits = self.decode_head(features)

        auxiliary_logits = None
        if self.auxiliary_head is not None:
            auxiliary_logits = self.auxiliary_head(features)

        loss = None
        if labels is not None:
            loss = self.compute_loss(logits, auxiliary_logits, labels)

        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )


class BeitBackbone(BeitPreTrainedModel, BackboneMixin):
    def __init__(self, config):
        super().__init__(config)
        super()._init_backbone(config)

        self.num_features = [config.hidden_size for _ in range(config.num_hidden_layers + 1)]
        self.embeddings = BeitEmbeddings(config)
        self.encoder = BeitEncoder(config, window_size=self.embeddings.patch_embeddings.patch_shape)

        if config.add_fpn:
            if len(self.config.out_indices) != 4:
                raise ValueError(
                    "BeitBackbone requires config.out_indices to be a list of 4 integers, "
                    "specifying which features to use from the backbone. One can use [3, 5, 7, 11] in case of "
                    "a base-sized architecture."
                )
            hidden_size = config.hidden_size
            self.fpn1 = nn.SequentialCell(
                mint.nn.ConvTranspose2d(hidden_size, hidden_size, kernel_size=2, stride=2),
                mint.nn.BatchNorm2d(hidden_size, eps=config.batch_norm_eps),
                mint.nn.GELU(),
                mint.nn.ConvTranspose2d(hidden_size, hidden_size, kernel_size=2, stride=2),
            )

            self.fpn2 = nn.SequentialCell(mint.nn.ConvTranspose2d(hidden_size, hidden_size, kernel_size=2, stride=2))
            self.fpn3 = mint.nn.Identity()
            self.fpn4 = mint.nn.MaxPool2d(kernel_size=2, stride=2)

        # initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def construct(
        self,
        pixel_values: Tensor,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> BackboneOutput:
        """
        Returns:

        Examples:

        ```python
        >>> from mindone.transformers import AutoImageProcessor, AutoBackbone
        >>> import mindspore as ms
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> processor = AutoImageProcessor.from_pretrained("microsoft/beit-base-patch16-224")
        >>> model = AutoBackbone.from_pretrained(
        ...     "microsoft/beit-base-patch16-224", out_features=["stage1", "stage2", "stage3", "stage4"]
        ... )

        >>> inputs = processor(image, return_tensors="np")
        >>> inputs = {k: ms.tensor(v) for k, v in inputs.items()}

        >>> outputs = model(**inputs)
        >>> feature_maps = outputs.feature_maps
        >>> list(feature_maps[-1].shape)
        [1, 768, 14, 14]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        batch_size = pixel_values.shape[0]
        embedding_output, (patch_height, patch_width) = self.embeddings(pixel_values)
        resolution = pixel_values.shape[2:]

        outputs = self.encoder(
            embedding_output,
            output_hidden_states=True,
            output_attentions=output_attentions,
            resolution=resolution,
            return_dict=return_dict,
        )

        hidden_states = outputs.hidden_states if return_dict else outputs[1]

        feature_maps = ()
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                if self.config.reshape_hidden_states:
                    hidden_state = hidden_state[:, 1:, :]
                    hidden_state = hidden_state.permute(0, 2, 1)
                    hidden_state = hidden_state.reshape(batch_size, -1, patch_height, patch_width)

                feature_maps += (hidden_state,)

        if self.config.add_fpn:
            feature_maps = [
                self.fpn1(feature_maps[0]),
                self.fpn2(feature_maps[1]),
                self.fpn3(feature_maps[2]),
                self.fpn4(feature_maps[3]),
            ]
            feature_maps = tuple(feature_maps)

        if not return_dict:
            if output_hidden_states:
                output = (feature_maps,) + outputs[1:]
            else:
                output = (feature_maps,) + outputs[2:]
            return output

        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )


__all__ = [
    "BeitForImageClassification",
    "BeitForMaskedImageModeling",
    "BeitForSemanticSegmentation",
    "BeitModel",
    "BeitPreTrainedModel",
    "BeitBackbone",
]
