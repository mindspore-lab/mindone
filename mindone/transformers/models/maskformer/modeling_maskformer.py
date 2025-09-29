# coding=utf-8
# Copyright 2022 Meta Platforms, Inc.s and The HuggingFace Inc. team. All rights reserved.
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
"""MindSpore MaskFormer model."""

import math
from dataclasses import dataclass
from numbers import Number
from typing import Dict, List, Optional, Tuple

import numpy as np
from transformers.models.detr import DetrConfig
from transformers.models.maskformer.configuration_maskformer import MaskFormerConfig
from transformers.models.maskformer.configuration_maskformer_swin import MaskFormerSwinConfig
from transformers.utils import ModelOutput

import mindspore as ms
from mindspore import Tensor, mint, nn

from ...activations import ACT2FN
from ...mindspore_adapter import ReLU
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...utils import is_scipy_available, logging, requires_backends
from ...utils.backbone_utils import load_backbone

if is_scipy_available():
    from scipy.optimize import linear_sum_assignment

logger = logging.get_logger(__name__)


_CONFIG_FOR_DOC = "MaskFormerConfig"
_CHECKPOINT_FOR_DOC = "facebook/maskformer-swin-base-ade"


@dataclass
# Copied from transformers.models.detr.modeling_detr.DetrDecoderOutput
class DetrDecoderOutput(BaseModelOutputWithCrossAttentions):
    """
    Base class for outputs of the DETR decoder. This class adds one attribute to BaseModelOutputWithCrossAttentions,
    namely an optional stack of intermediate decoder activations, i.e. the output of each decoder layer, each of them
    gone through a layernorm. This is useful when training the model with auxiliary decoding losses.

    Args:
        last_hidden_state (`ms.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(ms.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):  # noqa: E501
            Tuple of `ms.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(ms.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):  # noqa: E501
            Tuple of `ms.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        cross_attentions (`tuple(ms.Tensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):  # noqa: E501
            Tuple of `ms.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax,
            used to compute the weighted average in the cross-attention heads.
        intermediate_hidden_states (`ms.Tensor` of shape `(config.decoder_layers, batch_size, num_queries, hidden_size)`, *optional*, returned when `config.auxiliary_loss=True`):  # noqa: E501
            Intermediate decoder activations, i.e. the output of each decoder layer, each of them gone through a
            layernorm.
    """

    intermediate_hidden_states: Optional[ms.Tensor] = None


@dataclass
class MaskFormerPixelLevelModuleOutput(ModelOutput):
    """
    MaskFormer's pixel level module output. It returns both the last and (optionally) the hidden states from the
    `encoder` and `decoder`. By default, the `encoder` is a MaskFormerSwin Transformer and the `decoder` is a Feature
    Pyramid Network (FPN).

    The `encoder_last_hidden_state` are referred on the paper as **images features**, while `decoder_last_hidden_state`
    as **pixel embeddings**

    Args:
        encoder_last_hidden_state (`ms.Tensor` of shape`(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the encoder.
        encoder_hidden_states (`tuple(ms.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `ms.Tensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the model at
            the output of each stage.
        decoder_last_hidden_state (`ms.Tensor` of shape`(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the decoder.
        decoder_hidden_states (`tuple(ms.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `ms.Tensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the model at
            the output of each stage.
    """

    encoder_last_hidden_state: Optional[ms.Tensor] = None
    decoder_last_hidden_state: Optional[ms.Tensor] = None
    encoder_hidden_states: Optional[Tuple[ms.Tensor]] = None
    decoder_hidden_states: Optional[Tuple[ms.Tensor]] = None


@dataclass
class MaskFormerPixelDecoderOutput(ModelOutput):
    """
    MaskFormer's pixel decoder module output, practically a Feature Pyramid Network. It returns the last hidden state
    and (optionally) the hidden states.

    Args:
        last_hidden_state (`ms.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the model.
        hidden_states (`tuple(ms.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `ms.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(ms.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `ms.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights from Detr's decoder after the attention softmax, used to compute the
            weighted average in the self-attention heads.
    """

    last_hidden_state: Optional[ms.Tensor] = None
    hidden_states: Optional[Tuple[ms.Tensor]] = None
    attentions: Optional[Tuple[ms.Tensor]] = None


@dataclass
class MaskFormerModelOutput(ModelOutput):
    """
    Class for outputs of [`MaskFormerModel`]. This class returns all the needed hidden states to compute the logits.

    Args:
        encoder_last_hidden_state (`ms.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the encoder model (backbone).
        pixel_decoder_last_hidden_state (`ms.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the pixel decoder model (FPN).
        transformer_decoder_last_hidden_state (`ms.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Last hidden states (final feature map) of the last stage of the transformer decoder model.
        encoder_hidden_states (`tuple(ms.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):  # noqa: E501
            Tuple of `ms.Tensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the encoder
            model at the output of each stage.
        pixel_decoder_hidden_states (`tuple(ms.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):  # noqa: E501
            Tuple of `ms.Tensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the pixel
            decoder model at the output of each stage.
        transformer_decoder_hidden_states (`tuple(ms.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):  # noqa: E501
            Tuple of `ms.Tensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states (also called feature maps) of the
            transformer decoder at the output of each stage.
        hidden_states `tuple(ms.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):  # noqa: E501
            Tuple of `ms.Tensor` containing `encoder_hidden_states`, `pixel_decoder_hidden_states` and
            `decoder_hidden_states`
        attentions (`tuple(ms.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):  # noqa: E501
            Tuple of `ms.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights from Detr's decoder after the attention softmax, used to compute the
            weighted average in the self-attention heads.
    """

    encoder_last_hidden_state: Optional[ms.Tensor] = None
    pixel_decoder_last_hidden_state: Optional[ms.Tensor] = None
    transformer_decoder_last_hidden_state: Optional[ms.Tensor] = None
    encoder_hidden_states: Optional[Tuple[ms.Tensor]] = None
    pixel_decoder_hidden_states: Optional[Tuple[ms.Tensor]] = None
    transformer_decoder_hidden_states: Optional[Tuple[ms.Tensor]] = None
    hidden_states: Optional[Tuple[ms.Tensor]] = None
    attentions: Optional[Tuple[ms.Tensor]] = None


@dataclass
class MaskFormerForInstanceSegmentationOutput(ModelOutput):
    """
    Class for outputs of [`MaskFormerForInstanceSegmentation`].

    This output can be directly passed to [`~MaskFormerImageProcessor.post_process_semantic_segmentation`] or or
    [`~MaskFormerImageProcessor.post_process_instance_segmentation`] or
    [`~MaskFormerImageProcessor.post_process_panoptic_segmentation`] depending on the task. Please, see
    [`~MaskFormerImageProcessor] for details regarding usage.

    Args:
        loss (`ms.Tensor`, *optional*):
            The computed loss, returned when labels are present.
        class_queries_logits (`ms.Tensor`):
            A tensor of shape `(batch_size, num_queries, num_labels + 1)` representing the proposed classes for each
            query. Note the `+ 1` is needed because we incorporate the null class.
        masks_queries_logits (`ms.Tensor`):
            A tensor of shape `(batch_size, num_queries, height, width)` representing the proposed masks for each
            query.
        encoder_last_hidden_state (`ms.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the encoder model (backbone).
        pixel_decoder_last_hidden_state (`ms.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the pixel decoder model (FPN).
        transformer_decoder_last_hidden_state (`ms.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Last hidden states (final feature map) of the last stage of the transformer decoder model.
        encoder_hidden_states (`tuple(ms.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):  # noqa: E501
            Tuple of `ms.Tensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the encoder
            model at the output of each stage.
        pixel_decoder_hidden_states (`tuple(ms.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):  # noqa: E501
            Tuple of `ms.Tensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the pixel
            decoder model at the output of each stage.
        transformer_decoder_hidden_states (`tuple(ms.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):  # noqa: E501
            Tuple of `ms.Tensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the transformer decoder at the output
            of each stage.
        hidden_states `tuple(ms.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):  # noqa: E501
            Tuple of `ms.Tensor` containing `encoder_hidden_states`, `pixel_decoder_hidden_states` and
            `decoder_hidden_states`.
        attentions (`tuple(ms.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):  # noqa: E501
            Tuple of `ms.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights from Detr's decoder after the attention softmax, used to compute the
            weighted average in the self-attention heads.
    """

    loss: Optional[ms.Tensor] = None
    class_queries_logits: Optional[ms.Tensor] = None
    masks_queries_logits: Optional[ms.Tensor] = None
    auxiliary_logits: Optional[ms.Tensor] = None
    encoder_last_hidden_state: Optional[ms.Tensor] = None
    pixel_decoder_last_hidden_state: Optional[ms.Tensor] = None
    transformer_decoder_last_hidden_state: Optional[ms.Tensor] = None
    encoder_hidden_states: Optional[Tuple[ms.Tensor]] = None
    pixel_decoder_hidden_states: Optional[Tuple[ms.Tensor]] = None
    transformer_decoder_hidden_states: Optional[Tuple[ms.Tensor]] = None
    hidden_states: Optional[Tuple[ms.Tensor]] = None
    attentions: Optional[Tuple[ms.Tensor]] = None


def upsample_like(pixel_values: Tensor, like: Tensor, mode: str = "bilinear") -> Tensor:
    """
    An utility function that upsamples `pixel_values` to match the dimension of `like`.

    Args:
        pixel_values (`ms.Tensor`):
            The tensor we wish to upsample.
        like (`ms.Tensor`):
            The tensor we wish to use as size target.
        mode (str, *optional*, defaults to `"bilinear"`):
            The interpolation mode.

    Returns:
        `ms.Tensor`: The upsampled tensor
    """
    _, _, height, width = like.shape
    upsampled = mint.nn.functional.interpolate(pixel_values, size=(height, width), mode=mode, align_corners=False)
    return upsampled


# refactored from original implementation
def dice_loss(inputs: Tensor, labels: Tensor, num_masks: int) -> Tensor:
    r"""
    Compute the DICE loss, similar to generalized IOU for masks as follows:

    $$ \mathcal{L}_{\text{dice}(x, y) = 1 - \frac{2 * x \cap y }{x \cup y + 1}} $$

    In practice, since `labels` is a binary mask, (only 0s and 1s), dice can be computed as follow

    $$ \mathcal{L}_{\text{dice}(x, y) = 1 - \frac{2 * x * y }{x + y + 1}} $$

    Args:
        inputs (`ms.Tensor`):
            A tensor representing a mask.
        labels (`ms.Tensor`):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).
        num_masks (`int`):
            The number of masks present in the current batch, used for normalization.

    Returns:
        `ms.Tensor`: The computed loss.
    """
    probs = inputs.sigmoid().flatten(1)
    numerator = 2 * (probs * labels).sum(-1)
    denominator = probs.sum(-1) + labels.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    loss = loss.sum() / num_masks
    return loss


# refactored from original implementation
def sigmoid_focal_loss(inputs: Tensor, labels: Tensor, num_masks: int, alpha: float = 0.25, gamma: float = 2) -> Tensor:
    r"""
    Focal loss proposed in [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) originally used in
    RetinaNet. The loss is computed as follows:

    $$ \mathcal{L}_{\text{focal loss} = -(1 - p_t)^{\gamma}\log{(p_t)} $$

    where \\(CE(p_t) = -\log{(p_t)}}\\), CE is the standard Cross Entropy Loss

    Please refer to equation (1,2,3) of the paper for a better understanding.

    Args:
        inputs (`ms.Tensor`):
            A float tensor of arbitrary shape.
        labels (`ms.Tensor`):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).
        num_masks (`int`):
            The number of masks present in the current batch, used for normalization.
        alpha (float, *optional*, defaults to 0.25):
            Weighting factor in range (0,1) to balance positive vs negative examples.
        gamma (float, *optional*, defaults to 2.0):
            Exponent of the modulating factor \\(1 - p_t\\) to balance easy vs hard examples.

    Returns:
        `ms.Tensor`: The computed loss.
    """
    criterion = mint.nn.BCEWithLogitsLoss(reduction="none")
    probs = inputs.sigmoid()
    cross_entropy_loss = criterion(inputs, labels)
    p_t = probs * labels + (1 - probs) * (1 - labels)
    loss = cross_entropy_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * labels + (1 - alpha) * (1 - labels)
        loss = alpha_t * loss

    loss = loss.mean(1).sum() / num_masks
    return loss


# refactored from original implementation
def pair_wise_dice_loss(inputs: Tensor, labels: Tensor) -> Tensor:
    """
    A pair wise version of the dice loss, see `dice_loss` for usage.

    Args:
        inputs (`ms.Tensor`):
            A tensor representing a mask
        labels (`ms.Tensor`):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).

    Returns:
        `ms.Tensor`: The computed loss between each pairs.
    """
    inputs = inputs.sigmoid().flatten(1)
    numerator = 2 * mint.matmul(inputs, labels.T)
    # using broadcasting to get a [num_queries, NUM_CLASSES] matrix
    denominator = inputs.sum(-1)[:, None] + labels.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


# refactored from original implementation
def pair_wise_sigmoid_focal_loss(inputs: Tensor, labels: Tensor, alpha: float = 0.25, gamma: float = 2.0) -> Tensor:
    r"""
    A pair wise version of the focal loss, see `sigmoid_focal_loss` for usage.

    Args:
        inputs (`ms.Tensor`):
            A tensor representing a mask.
        labels (`ms.Tensor`):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).
        alpha (float, *optional*, defaults to 0.25):
            Weighting factor in range (0,1) to balance positive vs negative examples.
        gamma (float, *optional*, defaults to 2.0):
            Exponent of the modulating factor \\(1 - p_t\\) to balance easy vs hard examples.

    Returns:
        `ms.Tensor`: The computed loss between each pairs.
    """
    if alpha < 0:
        raise ValueError("alpha must be positive")

    height_and_width = inputs.shape[1]

    criterion = mint.nn.BCEWithLogitsLoss(reduction="none")
    prob = inputs.sigmoid()
    cross_entropy_loss_pos = criterion(inputs, mint.ones_like(inputs))
    focal_pos = ((1 - prob) ** gamma) * cross_entropy_loss_pos
    focal_pos *= alpha

    cross_entropy_loss_neg = criterion(inputs, mint.zeros_like(inputs))

    focal_neg = (prob**gamma) * cross_entropy_loss_neg
    focal_neg *= 1 - alpha

    loss = mint.matmul(focal_pos, labels.T) + mint.matmul(focal_neg, (1 - labels).T)

    return loss / height_and_width


# Copied from transformers.models.detr.modeling_detr.DetrAttention
class DetrAttention(nn.Cell):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.

    Here, we add position embeddings to the queries and keys (as explained in the DETR paper).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.k_proj = mint.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = mint.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = mint.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = mint.nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: ms.Tensor, seq_len: int, batch_size: int):
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).swapaxes(1, 2).contiguous()

    def with_pos_embed(self, tensor: ms.Tensor, object_queries: Optional[Tensor]):
        return tensor if object_queries is None else tensor + object_queries

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        object_queries: Optional[ms.Tensor] = None,
        key_value_states: Optional[ms.Tensor] = None,
        spatial_position_embeddings: Optional[ms.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[ms.Tensor, Optional[ms.Tensor], Optional[Tuple[ms.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        batch_size, target_len, embed_dim = hidden_states.shape

        # add position embeddings to the hidden states before projecting to queries and keys
        if object_queries is not None:
            hidden_states_original = hidden_states
            hidden_states = self.with_pos_embed(hidden_states, object_queries)

        # add key-value position embeddings to the key value states
        if spatial_position_embeddings is not None:
            key_value_states_original = key_value_states
            key_value_states = self.with_pos_embed(key_value_states, spatial_position_embeddings)

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, batch_size)
            value_states = self._shape(self.v_proj(key_value_states_original), -1, batch_size)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, batch_size)
            value_states = self._shape(self.v_proj(hidden_states_original), -1, batch_size)

        proj_shape = (batch_size * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, target_len, batch_size).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        source_len = key_states.shape[1]

        attn_weights = mint.bmm(query_states, key_states.swapaxes(1, 2))

        if attn_weights.shape != (batch_size * self.num_heads, target_len, source_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size * self.num_heads, target_len, source_len)}, but is"
                f" {attn_weights.shape}"
            )

        if attention_mask is not None:
            if attention_mask.shape != (batch_size, 1, target_len, source_len):
                raise ValueError(
                    f"Attention mask should be of size {(batch_size, 1, target_len, source_len)}, but is"
                    f" {attention_mask.shape}"
                )
            attn_weights = attn_weights.view(batch_size, self.num_heads, target_len, source_len) + attention_mask
            attn_weights = attn_weights.view(batch_size * self.num_heads, target_len, source_len)

        attn_weights = mint.nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(batch_size, self.num_heads, target_len, source_len)
            attn_weights = attn_weights_reshaped.view(batch_size * self.num_heads, target_len, source_len)
        else:
            attn_weights_reshaped = None

        attn_probs = mint.nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = mint.bmm(attn_probs, value_states)

        if attn_output.shape != (batch_size * self.num_heads, target_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, target_len, self.head_dim)}, but is"
                f" {attn_output.shape}"
            )

        attn_output = attn_output.view(batch_size, self.num_heads, target_len, self.head_dim)
        attn_output = attn_output.swapaxes(1, 2)
        attn_output = attn_output.reshape(batch_size, target_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


# Copied from transformers.models.detr.modeling_detr.DetrDecoderLayer
class DetrDecoderLayer(nn.Cell):
    def __init__(self, config: DetrConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = DetrAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = mint.nn.LayerNorm(self.embed_dim)
        self.encoder_attn = DetrAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.encoder_attn_layer_norm = mint.nn.LayerNorm(self.embed_dim)
        self.fc1 = mint.nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = mint.nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = mint.nn.LayerNorm(self.embed_dim)

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        object_queries: Optional[ms.Tensor] = None,
        query_position_embeddings: Optional[ms.Tensor] = None,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        encoder_attention_mask: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ):
        """
        Args:
            hidden_states (`ms.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`ms.Tensor`): attention mask of size
                `(batch, 1, target_len, source_len)` where padding elements are indicated by very large negative
                values.
            object_queries (`ms.Tensor`, *optional*):
                object_queries that are added to the hidden states
            in the cross-attention layer.
            query_position_embeddings (`ms.Tensor`, *optional*):
                position embeddings that are added to the queries and keys
            in the self-attention layer.
            encoder_hidden_states (`ms.Tensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`ms.Tensor`): encoder attention mask of size
                `(batch, 1, target_len, source_len)` where padding elements are indicated by very large negative
                values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            object_queries=query_position_embeddings,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )

        hidden_states = mint.nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            hidden_states, cross_attn_weights = self.encoder_attn(
                hidden_states=hidden_states,
                object_queries=query_position_embeddings,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                spatial_position_embeddings=object_queries,
                output_attentions=output_attentions,
            )

            hidden_states = mint.nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = mint.nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = mint.nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs


class DetrDecoder(nn.Cell):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`DetrDecoderLayer`].

    The decoder updates the query embeddings through multiple self-attention and cross-attention layers.

    Some small tweaks for DETR:

    - object_queries and query_position_embeddings are added to the forward pass.
    - if self.config.auxiliary_loss is set to True, also returns a stack of activations from all decoding layers.

    Args:
        config: DetrConfig
    """

    def __init__(self, config: DetrConfig):
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop

        self.layers = nn.CellList([DetrDecoderLayer(config) for _ in range(config.decoder_layers)])
        # in DETR, the decoder uses layernorm after the last decoder layer output
        self.layernorm = mint.nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False

    def construct(
        self,
        inputs_embeds=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        object_queries=None,
        query_position_embeddings=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            inputs_embeds (`ms.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                The query embeddings that are passed into the decoder.

            attention_mask (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on certain queries. Mask values selected in `[0, 1]`:

                - 1 for queries that are **not masked**,
                - 0 for queries that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states (`ms.Tensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`ms.Tensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding pixel_values of the encoder. Mask values selected
                in `[0, 1]`:

                - 1 for pixels that are real (i.e. **not masked**),
                - 0 for pixels that are padding (i.e. **masked**).

            object_queries (`ms.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Position embeddings that are added to the queries and keys in each cross-attention layer.
            query_position_embeddings (`ms.Tensor` of shape `(batch_size, num_queries, hidden_size)`):
                , *optional*): Position embeddings that are added to the queries and keys in each self-attention layer.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is not None:
            hidden_states = inputs_embeds
            input_shape = inputs_embeds.shape[:-1]

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _prepare_4d_attention_mask(
                encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            )

        # optional intermediate hidden states
        intermediate = () if self.config.auxiliary_loss else None

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = mint.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=None,
                object_queries=object_queries,
                query_position_embeddings=query_position_embeddings,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if self.config.auxiliary_loss:
                hidden_states = self.layernorm(hidden_states)
                intermediate += (hidden_states,)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # finally, apply layernorm
        hidden_states = self.layernorm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # stack intermediate decoder activations
        if self.config.auxiliary_loss:
            intermediate = mint.stack(intermediate)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attns, all_cross_attentions, intermediate]
                if v is not None
            )
        return DetrDecoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
            intermediate_hidden_states=intermediate,
        )


# refactored from original implementation
class MaskFormerHungarianMatcher(nn.Cell):
    """This class computes an assignment between the labels and the predictions of the network.

    For efficiency reasons, the labels don't include the no_object. Because of this, in general, there are more
    predictions than labels. In this case, we do a 1-to-1 matching of the best predictions, while the others are
    un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1.0, cost_mask: float = 1.0, cost_dice: float = 1.0):
        """Creates the matcher

        Params:
            cost_class (float, *optional*, defaults to 1.0):
                This is the relative weight of the classification error in the matching cost.
            cost_mask (float, *optional*,  defaults to 1.0):
                This is the relative weight of the focal loss of the binary mask in the matching cost.
            cost_dice (float, *optional*, defaults to 1.0):
                This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        if cost_class == 0 and cost_mask == 0 and cost_dice == 0:
            raise ValueError("All costs cant be 0")
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

    def construct(self, masks_queries_logits, class_queries_logits, mask_labels, class_labels) -> List[Tuple[Tensor]]:
        """Performs the matching

        Params:
            masks_queries_logits (`ms.Tensor`):
                A tensor` of dim `batch_size, num_queries, num_labels` with the
                  classification logits.
            class_queries_logits (`ms.Tensor`):
                A tensor` of dim `batch_size, num_queries, height, width` with the
                  predicted masks.

            class_labels (`ms.Tensor`):
                A tensor` of dim `num_target_boxes` (where num_target_boxes is the number
                  of ground-truth objects in the target) containing the class labels.
            mask_labels (`ms.Tensor`):
                A tensor` of dim `num_target_boxes, height, width` containing the target
                  masks.

        Returns:
            `List[Tuple[Tensor]]`: A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected labels (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes).
        """
        indices: List[Tuple[np.array]] = []

        preds_masks = masks_queries_logits
        preds_probs = class_queries_logits
        # iterate through batch size
        for pred_probs, pred_mask, target_mask, labels in zip(preds_probs, preds_masks, mask_labels, class_labels):
            # downsample the target mask, save memory
            target_mask = mint.nn.functional.interpolate(
                target_mask[:, None], size=pred_mask.shape[-2:], mode="nearest"
            )
            pred_probs = pred_probs.softmax(-1)
            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -pred_probs[:, labels]
            # flatten spatial dimension "q h w -> q (h w)"
            pred_mask_flat = pred_mask.flatten(1)  # [num_queries, height*width]
            # same for target_mask "c h w -> c (h w)"
            target_mask_flat = target_mask[:, 0].flatten(1)  # [num_total_labels, height*width]
            # compute the focal loss between each mask pairs -> shape (num_queries, num_labels)
            cost_mask = pair_wise_sigmoid_focal_loss(pred_mask_flat, target_mask_flat)
            # Compute the dice loss betwen each mask pairs -> shape (num_queries, num_labels)
            cost_dice = pair_wise_dice_loss(pred_mask_flat, target_mask_flat)
            # final cost matrix
            cost_matrix = self.cost_mask * cost_mask + self.cost_class * cost_class + self.cost_dice * cost_dice
            # do the assigmented using the hungarian algorithm in scipy
            assigned_indices: Tuple[np.array] = linear_sum_assignment(cost_matrix)
            indices.append(assigned_indices)

        # It could be stacked in one tensor
        matched_indices = [(ms.tensor(i, dtype=ms.int64), ms.tensor(j, dtype=ms.int64)) for i, j in indices]
        return matched_indices

    def __repr__(self):
        head = "Matcher " + self.__class__.__name__
        body = [
            f"cost_class: {self.cost_class}",
            f"cost_mask: {self.cost_mask}",
            f"cost_dice: {self.cost_dice}",
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


# copied and adapted from original implementation
class MaskFormerLoss(nn.Cell):
    def __init__(
        self,
        num_labels: int,
        matcher: MaskFormerHungarianMatcher,
        weight_dict: Dict[str, float],
        eos_coef: float,
    ):
        """
        The MaskFormer Loss. The loss is computed very similar to DETR. The process happens in two steps: 1) we compute
        hungarian assignment between ground truth masks and the outputs of the model 2) we supervise each pair of
        matched ground-truth / prediction (supervise class and mask)

        Args:
            num_labels (`int`):
                The number of classes.
            matcher (`MaskFormerHungarianMatcher`):
                A mindspore cell that computes the assigments between the predictions and labels.
            weight_dict (`Dict[str, float]`):
                A dictionary of weights to be applied to the different losses.
            eos_coef (`float`):
                Weight to apply to the null class.
        """

        super().__init__()
        requires_backends(self, ["scipy"])
        self.num_labels = num_labels
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        empty_weight = mint.ones(self.num_labels + 1)
        empty_weight[-1] = self.eos_coef
        self.empty_weight = ms.Parameter(empty_weight, requires_grad=False, name="empty_weight")

    def _max_by_axis(self, the_list: List[List[int]]) -> List[int]:
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    def _pad_images_to_max_in_batch(self, tensors: List[Tensor]) -> Tuple[Tensor, Tensor]:
        # get the maximum size in the batch
        max_size = self._max_by_axis([list(tensor.shape) for tensor in tensors])
        batch_size = len(tensors)
        # compute finel size
        batch_shape = [batch_size] + max_size
        b, _, h, w = batch_shape
        # get metadata
        dtype = tensors[0].dtype
        padded_tensors = mint.zeros(batch_shape, dtype=dtype)
        padding_masks = mint.ones((b, h, w), dtype=ms.bool_)
        # pad the tensors to the size of the biggest one
        for tensor, padded_tensor, padding_mask in zip(tensors, padded_tensors, padding_masks):
            padded_tensor[: tensor.shape[0], : tensor.shape[1], : tensor.shape[2]].copy_(tensor)
            padding_mask[: tensor.shape[1], : tensor.shape[2]] = False

        return padded_tensors, padding_masks

    def loss_labels(
        self, class_queries_logits: Tensor, class_labels: List[Tensor], indices: Tuple[np.array]
    ) -> Dict[str, Tensor]:
        """Compute the losses related to the labels using cross entropy.

        Args:
            class_queries_logits (`ms.Tensor`):
                A tensor of shape `batch_size, num_queries, num_labels`
            class_labels (`List[ms.Tensor]`):
                List of class labels of shape `(labels)`.
            indices (`Tuple[np.array])`:
                The indices computed by the Hungarian matcher.

        Returns:
            `Dict[str, Tensor]`: A dict of `ms.Tensor` containing the following key:
            - **loss_cross_entropy** -- The loss computed using cross entropy on the predicted and ground truth labels.
        """

        pred_logits = class_queries_logits
        batch_size, num_queries, _ = pred_logits.shape
        criterion = mint.nn.CrossEntropyLoss(weight=self.empty_weight)
        idx = self._get_predictions_permutation_indices(indices)
        # shape = (batch_size, num_queries)
        target_classes_o = mint.cat([target[j] for target, (_, j) in zip(class_labels, indices)])
        # shape = (batch_size, num_queries)
        target_classes = mint.full((batch_size, num_queries), fill_value=self.num_labels, dtype=ms.int64)
        target_classes[idx] = target_classes_o
        # target_classes is a (batch_size, num_labels, num_queries), we need to permute pred_logits "b q c -> b c q"
        pred_logits_transposed = pred_logits.swapaxes(1, 2)
        loss_ce = criterion(pred_logits_transposed, target_classes)
        losses = {"loss_cross_entropy": loss_ce}
        return losses

    def loss_masks(
        self, masks_queries_logits: Tensor, mask_labels: List[Tensor], indices: Tuple[np.array], num_masks: int
    ) -> Dict[str, Tensor]:
        """Compute the losses related to the masks using focal and dice loss.

        Args:
            masks_queries_logits (`ms.Tensor`):
                A tensor of shape `batch_size, num_queries, height, width`
            mask_labels (`ms.Tensor`):
                List of mask labels of shape `(labels, height, width)`.
            indices (`Tuple[np.array])`:
                The indices computed by the Hungarian matcher.
            num_masks (`int)`:
                The number of masks, used for normalization.

        Returns:
            `Dict[str, Tensor]`: A dict of `ms.Tensor` containing two keys:
            - **loss_mask** -- The loss computed using sigmoid focal loss on the predicted and ground truth masks.
            - **loss_dice** -- The loss computed using dice loss on the predicted on the predicted and ground truth
              masks.
        """
        src_idx = self._get_predictions_permutation_indices(indices)
        tgt_idx = self._get_targets_permutation_indices(indices)
        # shape (batch_size * num_queries, height, width)
        pred_masks = masks_queries_logits[src_idx]
        # shape (batch_size, num_queries, height, width)
        # pad all and stack the targets to the num_labels dimension
        target_masks, _ = self._pad_images_to_max_in_batch(mask_labels)
        target_masks = target_masks[tgt_idx]
        # upsample predictions to the target size, we have to add one dim to use interpolate
        pred_masks = mint.nn.functional.interpolate(
            pred_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False
        )
        pred_masks = pred_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        losses = {
            "loss_mask": sigmoid_focal_loss(pred_masks, target_masks, num_masks),
            "loss_dice": dice_loss(pred_masks, target_masks, num_masks),
        }
        return losses

    def _get_predictions_permutation_indices(self, indices):
        # permute predictions following indices
        batch_indices = mint.cat([mint.full_like(src, i) for i, (src, _) in enumerate(indices)])
        predictions_indices = mint.cat([src for (src, _) in indices])
        return batch_indices, predictions_indices

    def _get_targets_permutation_indices(self, indices):
        # permute labels following indices
        batch_indices = mint.cat([mint.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        target_indices = mint.cat([tgt for (_, tgt) in indices])
        return batch_indices, target_indices

    def construct(
        self,
        masks_queries_logits: Tensor,
        class_queries_logits: Tensor,
        mask_labels: List[Tensor],
        class_labels: List[Tensor],
        auxiliary_predictions: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """
        This performs the loss computation.

        Args:
            masks_queries_logits (`ms.Tensor`):
                A tensor of shape `batch_size, num_queries, height, width`
            class_queries_logits (`ms.Tensor`):
                A tensor of shape `batch_size, num_queries, num_labels`
            mask_labels (`ms.Tensor`):
                List of mask labels of shape `(labels, height, width)`.
            class_labels (`List[ms.Tensor]`):
                List of class labels of shape `(labels)`.
            auxiliary_predictions (`Dict[str, ms.Tensor]`, *optional*):
                if `use_auxiliary_loss` was set to `true` in [`MaskFormerConfig`], then it contains the logits from the
                inner layers of the Detr's Decoder.

        Returns:
            `Dict[str, Tensor]`: A dict of `ms.Tensor` containing two keys:
            - **loss_cross_entropy** -- The loss computed using cross entropy on the predicted and ground truth labels.
            - **loss_mask** -- The loss computed using sigmoid focal loss on the predicted and ground truth masks.
            - **loss_dice** -- The loss computed using dice loss on the predicted on the predicted and ground truth
              masks.
            if `use_auxiliary_loss` was set to `true` in [`MaskFormerConfig`], the dictionary contains addional losses
            for each auxiliary predictions.
        """

        # retrieve the matching between the outputs of the last layer and the labels
        indices = self.matcher(masks_queries_logits, class_queries_logits, mask_labels, class_labels)
        # compute the average number of target masks for normalization purposes
        num_masks: Number = self.get_num_masks(class_labels)
        # get all the losses
        losses: Dict[str, Tensor] = {
            **self.loss_masks(masks_queries_logits, mask_labels, indices, num_masks),
            **self.loss_labels(class_queries_logits, class_labels, indices),
        }
        # in case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if auxiliary_predictions is not None:
            for idx, aux_outputs in enumerate(auxiliary_predictions):
                masks_queries_logits = aux_outputs["masks_queries_logits"]
                class_queries_logits = aux_outputs["class_queries_logits"]
                loss_dict = self.construct(masks_queries_logits, class_queries_logits, mask_labels, class_labels)
                loss_dict = {f"{key}_{idx}": value for key, value in loss_dict.items()}
                losses.update(loss_dict)

        return losses

    def get_num_masks(self, class_labels: ms.Tensor) -> ms.Tensor:
        """
        Computes the average number of target masks across the batch, for normalization purposes.
        """
        num_masks = sum([len(classes) for classes in class_labels])
        num_masks = ms.tensor(num_masks, dtype=ms.float32)
        world_size = 1

        num_masks = mint.clamp(num_masks / world_size, min=1)
        return num_masks


class MaskFormerFPNConvLayer(nn.Cell):
    def __init__(self, in_features: int, out_features: int, kernel_size: int = 3, padding: int = 1):
        """
        A basic module that executes conv - norm - in sequence used in MaskFormer.

        Args:
            in_features (`int`):
                The number of input features (channels).
            out_features (`int`):
                The number of outputs features (channels).
        """
        super().__init__()
        self.layers = [
            mint.nn.Conv2d(in_features, out_features, kernel_size=kernel_size, padding=padding, bias=False),
            mint.nn.GroupNorm(32, out_features),
            ReLU(inplace=True),
        ]
        for i, layer in enumerate(self.layers):
            # Provide backwards compatibility from when the class inherited from nn.Sequential
            # In nn.Sequential subclasses, the name given to the layer is its index in the sequence.
            # In nn.Module subclasses they derived from the instance attribute they are assigned to e.g.
            # self.my_layer_name = Layer()
            # We can't give instance attributes integer names i.e. self.0 is not permitted and so need to register
            # explicitly
            self.insert_child_to_cell(str(i), layer)

    def construct(self, input: Tensor) -> Tensor:
        hidden_state = input
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class MaskFormerFPNLayer(nn.Cell):
    def __init__(self, in_features: int, lateral_features: int):
        """
        A Feature Pyramid Network Layer (FPN) layer. It creates a feature map by aggregating features from the previous
        and backbone layer. Due to the spatial mismatch, the tensor coming from the previous layer is upsampled.

        Args:
            in_features (`int`):
                The number of input features (channels).
            lateral_features (`int`):
                The number of lateral features (channels).
        """
        super().__init__()
        self.proj = nn.SequentialCell(
            mint.nn.Conv2d(lateral_features, in_features, kernel_size=1, padding=0, bias=False),
            mint.nn.GroupNorm(32, in_features),
        )

        self.block = MaskFormerFPNConvLayer(in_features, in_features)

    def construct(self, down: Tensor, left: Tensor) -> Tensor:
        left = self.proj(left)
        down = mint.nn.functional.interpolate(down, size=left.shape[-2:], mode="nearest")
        down += left
        down = self.block(down)
        return down


class MaskFormerFPNModel(nn.Cell):
    def __init__(self, in_features: int, lateral_widths: List[int], feature_size: int = 256):
        """
        Feature Pyramid Network, given an input tensor and a set of feature map of different feature/spatial size, it
        creates a list of feature maps with the same feature size.

        Args:
            in_features (`int`):
                The number of input features (channels).
            lateral_widths (`List[int]`):
                A list with the features (channels) size of each lateral connection.
            feature_size (int, *optional*, defaults to 256):
                The features (channels) of the resulting feature maps.
        """
        super().__init__()
        self.stem = MaskFormerFPNConvLayer(in_features, feature_size)
        self.layers = nn.SequentialCell(
            *[MaskFormerFPNLayer(feature_size, lateral_width) for lateral_width in lateral_widths[::-1]]
        )

    def construct(self, features: List[Tensor]) -> List[Tensor]:
        fpn_features = []
        last_feature = features[-1]
        other_features = features[:-1]
        output = self.stem(last_feature)
        for layer, left in zip(self.layers, other_features[::-1]):
            output = layer(output, left)
            fpn_features.append(output)
        return fpn_features


class MaskFormerPixelDecoder(nn.Cell):
    def __init__(self, *args, feature_size: int = 256, mask_feature_size: int = 256, **kwargs):
        r"""
        Pixel Decoder Module proposed in [Per-Pixel Classification is Not All You Need for Semantic
        Segmentation](https://arxiv.org/abs/2107.06278). It first runs the backbone's features into a Feature Pyramid
        Network creating a list of feature maps. Then, it projects the last one to the correct `mask_size`.

        Args:
            feature_size (`int`, *optional*, defaults to 256):
                The feature size (channel dimension) of the FPN feature maps.
            mask_feature_size (`int`, *optional*, defaults to 256):
                The features (channels) of the target masks size \\(C_{\epsilon}\\) in the paper.
        """
        super().__init__()

        self.fpn = MaskFormerFPNModel(*args, feature_size=feature_size, **kwargs)
        self.mask_projection = mint.nn.Conv2d(feature_size, mask_feature_size, kernel_size=3, padding=1)

    def construct(
        self, features: List[Tensor], output_hidden_states: bool = False, return_dict: bool = True
    ) -> MaskFormerPixelDecoderOutput:
        fpn_features = self.fpn(features)
        # we use the last feature map
        last_feature_projected = self.mask_projection(fpn_features[-1])

        if not return_dict:
            return (last_feature_projected, tuple(fpn_features)) if output_hidden_states else (last_feature_projected,)

        return MaskFormerPixelDecoderOutput(
            last_hidden_state=last_feature_projected, hidden_states=tuple(fpn_features) if output_hidden_states else ()
        )


# copied and adapted from original implementation, also practically equal to DetrSinePositionEmbedding
class MaskFormerSinePositionEmbedding(nn.Cell):
    """
    This is a more standard version of the position embedding, very similar to the one used by the Attention is all you
    need paper, generalized to work on images.
    """

    def __init__(
        self, num_pos_feats: int = 64, temperature: int = 10000, normalize: bool = False, scale: Optional[float] = None
    ):
        super().__init__()
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi if scale is None else scale

    def construct(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if mask is None:
            mask = mint.zeros((x.shape[0], x.shape[2], x.shape[3]), dtype=ms.bool_)
        not_mask = (~mask).to(x.dtype)
        y_embed = not_mask.cumsum(1)
        x_embed = not_mask.cumsum(2)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = mint.arange(self.num_pos_feats, dtype=ms.int64).type_as(x)
        dim_t = self.temperature ** (2 * mint.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = mint.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = mint.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = mint.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PredictionBlock(nn.Cell):
    def __init__(self, in_dim: int, out_dim: int, activation: nn.Cell) -> None:
        super().__init__()
        self.layers = [mint.nn.Linear(in_dim, out_dim), activation]
        # Maintain submodule indexing as if part of a Sequential block
        for i, layer in enumerate(self.layers):
            self.insert_child_to_cell(str(i), layer)

    def construct(self, input: Tensor) -> Tensor:
        hidden_state = input
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class MaskformerMLPPredictionHead(nn.Cell):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3):
        """
        A classic Multi Layer Perceptron (MLP).

        Args:
            input_dim (`int`):
                The input dimensions.
            hidden_dim (`int`):
                The hidden dimensions.
            output_dim (`int`):
                The output dimensions.
            num_layers (int, *optional*, defaults to 3):
                The number of layers.
        """
        super().__init__()
        in_dims = [input_dim] + [hidden_dim] * (num_layers - 1)
        out_dims = [hidden_dim] * (num_layers - 1) + [output_dim]

        self.layers = []
        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            activation = mint.nn.ReLU() if i < num_layers - 1 else mint.nn.Identity()
            layer = PredictionBlock(in_dim, out_dim, activation=activation)
            self.layers.append(layer)
            # Provide backwards compatibility from when the class inherited from nn.Sequential
            # In nn.Sequential subclasses, the name given to the layer is its index in the sequence.
            # In nn.Module subclasses they derived from the instance attribute they are assigned to e.g.
            # self.my_layer_name = Layer()
            # We can't give instance attributes integer names i.e. self.0 is not permitted and so need to register
            # explicitly
            self.insert_child_to_cell(str(i), layer)

    def construct(self, input: Tensor) -> Tensor:
        hidden_state = input
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class MaskFormerPixelLevelModule(nn.Cell):
    def __init__(self, config: MaskFormerConfig):
        """
        Pixel Level Module proposed in [Per-Pixel Classification is Not All You Need for Semantic
        Segmentation](https://arxiv.org/abs/2107.06278). It runs the input image through a backbone and a pixel
        decoder, generating an image feature map and pixel embeddings.

        Args:
            config ([`MaskFormerConfig`]):
                The configuration used to instantiate this model.
        """
        super().__init__()
        if getattr(config, "backbone_config") is not None and config.backbone_config.model_type == "swin":
            # for backwards compatibility
            backbone_config = config.backbone_config
            backbone_config = MaskFormerSwinConfig.from_dict(backbone_config.to_dict())
            backbone_config.out_features = ["stage1", "stage2", "stage3", "stage4"]
            config.backbone_config = backbone_config
        self.encoder = load_backbone(config)

        feature_channels = self.encoder.channels
        self.decoder = MaskFormerPixelDecoder(
            in_features=feature_channels[-1],
            feature_size=config.fpn_feature_size,
            mask_feature_size=config.mask_feature_size,
            lateral_widths=feature_channels[:-1],
        )

    def construct(
        self, pixel_values: Tensor, output_hidden_states: bool = False, return_dict: bool = True
    ) -> MaskFormerPixelLevelModuleOutput:
        features = self.encoder(pixel_values).feature_maps
        decoder_output = self.decoder(features, output_hidden_states, return_dict=return_dict)

        if not return_dict:
            last_hidden_state = decoder_output[0]
            outputs = (features[-1], last_hidden_state)
            if output_hidden_states:
                hidden_states = decoder_output[1]
                outputs = outputs + (tuple(features),) + (hidden_states,)
            return outputs

        return MaskFormerPixelLevelModuleOutput(
            # the last feature is actually the output from the last layer
            encoder_last_hidden_state=features[-1],
            decoder_last_hidden_state=decoder_output.last_hidden_state,
            encoder_hidden_states=tuple(features) if output_hidden_states else (),
            decoder_hidden_states=decoder_output.hidden_states if output_hidden_states else (),
        )


class MaskFormerTransformerModule(nn.Cell):
    """
    The MaskFormer's transformer module.
    """

    def __init__(self, in_features: int, config: MaskFormerConfig):
        super().__init__()
        hidden_size = config.decoder_config.hidden_size
        should_project = in_features != hidden_size
        self.position_embedder = MaskFormerSinePositionEmbedding(num_pos_feats=hidden_size // 2, normalize=True)
        self.queries_embedder = mint.nn.Embedding(config.decoder_config.num_queries, hidden_size)
        self.input_projection = mint.nn.Conv2d(in_features, hidden_size, kernel_size=1) if should_project else None
        self.decoder = DetrDecoder(config=config.decoder_config)

    def construct(
        self,
        image_features: Tensor,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: Optional[bool] = None,
    ) -> DetrDecoderOutput:
        if self.input_projection is not None:
            image_features = self.input_projection(image_features)
        object_queries = self.position_embedder(image_features)
        # repeat the queries "q c -> b q c"
        batch_size = image_features.shape[0]
        queries_embeddings = self.queries_embedder.weight.unsqueeze(0).tile((batch_size, 1, 1))
        inputs_embeds = mint.zeros_like(queries_embeddings)

        # if self.training:
        #     inputs_embeds.requires_grad_(True)

        batch_size, num_channels, height, width = image_features.shape
        # rearrange both image_features and object_queries "b c h w -> b (h w) c"
        image_features = image_features.view(batch_size, num_channels, height * width).permute(0, 2, 1)
        object_queries = object_queries.view(batch_size, num_channels, height * width).permute(0, 2, 1)

        decoder_output: DetrDecoderOutput = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=None,
            encoder_hidden_states=image_features,
            encoder_attention_mask=None,
            object_queries=object_queries,
            query_position_embeddings=queries_embeddings,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return decoder_output


MASKFORMER_START_DOCSTRING = r"""
    This model is a MindSpore
    [ms.nn.Cell](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Cell.html) sub-class. Use
    it as a regular MindSpore Cell and refer to the MindSpore documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`MaskFormerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

MASKFORMER_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`ms.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`MaskFormerImageProcessor.__call__`] for details.
        pixel_mask (`ms.Tensor` of shape `(batch_size, height, width)`, *optional*):
            Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:

            - 1 for pixels that are real (i.e. **not masked**),
            - 0 for pixels that are padding (i.e. **masked**).

            [What are attention masks?](../glossary#attention-mask)
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of Detr's decoder attention layers.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~MaskFormerModelOutput`] instead of a plain tuple.
"""


class MaskFormerPreTrainedModel(PreTrainedModel):
    config_class = MaskFormerConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"

    def _init_weights(self, module: nn.Cell):
        pass


class MaskFormerModel(MaskFormerPreTrainedModel):
    def __init__(self, config: MaskFormerConfig):
        super().__init__(config)
        self.pixel_level_module = MaskFormerPixelLevelModule(config)
        self.transformer_module = MaskFormerTransformerModule(
            in_features=self.pixel_level_module.encoder.channels[-1], config=config
        )

        self.post_init()

    def construct(
        self,
        pixel_values: Tensor,
        pixel_mask: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> MaskFormerModelOutput:
        r"""
        Returns:

        Examples:

        ```python
        >>> from mindone.transformers import AutoImageProcessor, MaskFormerModel
        >>> import mindspore as ms
        >>> from PIL import Image
        >>> import requests

        >>> # load MaskFormer fine-tuned on ADE20k semantic segmentation
        >>> image_processor = AutoImageProcessor.from_pretrained(
        ...     "facebook/maskformer-swin-base-ade", revision="refs/pr/1"
        ... )
        >>> model = MaskFormerModel.from_pretrained("facebook/maskformer-swin-base-ade", revision="refs/pr/1")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = image_processor(image, return_tensors="ms")

        >>> # forward pass
        >>> outputs = model(**inputs)

        >>> # the decoder of MaskFormer outputs hidden states of shape (batch_size, num_queries, hidden_size)
        >>> transformer_decoder_last_hidden_state = outputs.transformer_decoder_last_hidden_state
        >>> list(transformer_decoder_last_hidden_state.shape)
        [1, 100, 256]
        ```"""

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, _, height, width = pixel_values.shape

        if pixel_mask is None:
            pixel_mask = mint.ones((batch_size, height, width))

        pixel_level_module_output = self.pixel_level_module(pixel_values, output_hidden_states, return_dict=return_dict)
        image_features = pixel_level_module_output[0]
        pixel_embeddings = pixel_level_module_output[1]

        transformer_module_output = self.transformer_module(image_features, output_hidden_states, output_attentions)
        queries = transformer_module_output.last_hidden_state

        encoder_hidden_states = None
        pixel_decoder_hidden_states = None
        transformer_decoder_hidden_states = None
        hidden_states = None

        if output_hidden_states:
            encoder_hidden_states = pixel_level_module_output[2]
            pixel_decoder_hidden_states = pixel_level_module_output[3]
            transformer_decoder_hidden_states = transformer_module_output[1]
            hidden_states = encoder_hidden_states + pixel_decoder_hidden_states + transformer_decoder_hidden_states

        output = MaskFormerModelOutput(
            encoder_last_hidden_state=image_features,
            pixel_decoder_last_hidden_state=pixel_embeddings,
            transformer_decoder_last_hidden_state=queries,
            encoder_hidden_states=encoder_hidden_states,
            pixel_decoder_hidden_states=pixel_decoder_hidden_states,
            transformer_decoder_hidden_states=transformer_decoder_hidden_states,
            hidden_states=hidden_states,
            attentions=transformer_module_output.attentions,
        )

        if not return_dict:
            output = tuple(v for v in output.values())

        return output


class MaskFormerForInstanceSegmentation(MaskFormerPreTrainedModel):
    def __init__(self, config: MaskFormerConfig):
        super().__init__(config)
        self.model = MaskFormerModel(config)
        hidden_size = config.decoder_config.hidden_size
        # + 1 because we add the "null" class
        self.class_predictor = mint.nn.Linear(hidden_size, config.num_labels + 1)
        self.mask_embedder = MaskformerMLPPredictionHead(hidden_size, hidden_size, config.mask_feature_size)

        self.matcher = MaskFormerHungarianMatcher(
            cost_class=1.0, cost_dice=config.dice_weight, cost_mask=config.mask_weight
        )

        self.weight_dict: Dict[str, float] = {
            "loss_cross_entropy": config.cross_entropy_weight,
            "loss_mask": config.mask_weight,
            "loss_dice": config.dice_weight,
        }

        self.criterion = MaskFormerLoss(
            config.num_labels,
            matcher=self.matcher,
            weight_dict=self.weight_dict,
            eos_coef=config.no_object_weight,
        )

        self.post_init()

    def get_loss_dict(
        self,
        masks_queries_logits: Tensor,
        class_queries_logits: Tensor,
        mask_labels: Tensor,
        class_labels: Tensor,
        auxiliary_logits: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        loss_dict: Dict[str, Tensor] = self.criterion(
            masks_queries_logits, class_queries_logits, mask_labels, class_labels, auxiliary_logits
        )
        # weight each loss by `self.weight_dict[<LOSS_NAME>]` including auxiliary losses
        for key, weight in self.weight_dict.items():
            for loss_key, loss in loss_dict.items():
                if key in loss_key:
                    loss *= weight

        return loss_dict

    def get_loss(self, loss_dict: Dict[str, Tensor]) -> Tensor:
        return sum(loss_dict.values())

    def get_logits(self, outputs: MaskFormerModelOutput) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        pixel_embeddings = outputs.pixel_decoder_last_hidden_state
        # get the auxiliary predictions (one for each decoder's layer)
        auxiliary_logits: List[str, Tensor] = []

        # This code is a little bit cumbersome, an improvement can be to return a list of predictions. If we have auxiliary loss then we are going to return more than one element in the list  # noqa: E501
        if self.config.use_auxiliary_loss:
            stacked_transformer_decoder_outputs = mint.stack(outputs.transformer_decoder_hidden_states)
            classes = self.class_predictor(stacked_transformer_decoder_outputs)
            class_queries_logits = classes[-1]
            # get the masks
            mask_embeddings = self.mask_embedder(stacked_transformer_decoder_outputs)

            binaries_masks = mint.einsum("lbqc, bchw -> lbqhw", mask_embeddings, pixel_embeddings)

            masks_queries_logits = binaries_masks[-1]
            # go til [:-1] because the last one is always used
            for aux_binary_masks, aux_classes in zip(binaries_masks[:-1], classes[:-1]):
                auxiliary_logits.append({"masks_queries_logits": aux_binary_masks, "class_queries_logits": aux_classes})

        else:
            transformer_decoder_hidden_states = outputs.transformer_decoder_last_hidden_state
            classes = self.class_predictor(transformer_decoder_hidden_states)
            class_queries_logits = classes
            # get the masks
            mask_embeddings = self.mask_embedder(transformer_decoder_hidden_states)
            # sum up over the channels

            masks_queries_logits = mint.einsum("bqc, bchw -> bqhw", mask_embeddings, pixel_embeddings)

        return class_queries_logits, masks_queries_logits, auxiliary_logits

    def construct(
        self,
        pixel_values: Tensor,
        mask_labels: Optional[List[Tensor]] = None,
        class_labels: Optional[List[Tensor]] = None,
        pixel_mask: Optional[Tensor] = None,
        output_auxiliary_logits: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> MaskFormerForInstanceSegmentationOutput:
        r"""
        mask_labels (`List[ms.Tensor]`, *optional*):
            List of mask labels of shape `(num_labels, height, width)` to be fed to a model
        class_labels (`List[ms.Tensor]`, *optional*):
            list of target class labels of shape `(num_labels, height, width)` to be fed to a model. They identify the
            labels of `mask_labels`, e.g. the label of `mask_labels[i][j]` if `class_labels[i][j]`.

        Returns:

        Examples:

        Semantic segmentation example:

        ```python
        >>> from mindone.transformers import AutoImageProcessor, MaskFormerForInstanceSegmentation
        >>> import mindspore as ms
        >>> from PIL import Image
        >>> import requests

        >>> # load MaskFormer fine-tuned on ADE20k semantic segmentation
        >>> image_processor = AutoImageProcessor.from_pretrained(
        ...     "facebook/maskformer-swin-base-ade", revision="refs/pr/1"
        ... )
        >>> model = MaskFormerForInstanceSegmentation.from_pretrained(
        ...     "facebook/maskformer-swin-base-ade", revision="refs/pr/1"
        ... )

        >>> url = (
        ...     "https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg"
        ... )
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = image_processor(images=image, return_tensors="ms")

        >>> outputs = model(**inputs)
        >>> # model predicts class_queries_logits of shape `(batch_size, num_queries)`
        >>> # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
        >>> class_queries_logits = outputs.class_queries_logits
        >>> masks_queries_logits = outputs.masks_queries_logits

        >>> # you can pass them to image_processor for postprocessing
        >>> predicted_semantic_map = image_processor.post_process_semantic_segmentation(
        ...     outputs, target_sizes=[(image.height, image.width)]
        ... )[0]

        >>> # we refer to the demo notebooks for visualization (see "Resources" section in the MaskFormer docs)
        >>> list(predicted_semantic_map.shape)
        [512, 683]
        ```

        Panoptic segmentation example:

        ```python
        >>> from mindone.transformers import AutoImageProcessor, MaskFormerForInstanceSegmentation
        >>> import mindspore as ms
        >>> from PIL import Image
        >>> import requests

        >>> # load MaskFormer fine-tuned on COCO panoptic segmentation
        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/maskformer-swin-base-coco")
        >>> model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-coco")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = image_processor(images=image, return_tensors="ms")

        >>> outputs = model(**inputs)
        >>> # model predicts class_queries_logits of shape `(batch_size, num_queries)`
        >>> # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
        >>> class_queries_logits = outputs.class_queries_logits
        >>> masks_queries_logits = outputs.masks_queries_logits

        >>> # you can pass them to image_processor for postprocessing
        >>> result = image_processor.post_process_panoptic_segmentation(outputs, target_sizes=[(image.height, image.width)])[0]

        >>> # we refer to the demo notebooks for visualization (see "Resources" section in the MaskFormer docs)
        >>> predicted_panoptic_map = result["segmentation"]
        >>> list(predicted_panoptic_map.shape)
        [480, 640]
        ```
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        raw_outputs = self.model(
            pixel_values,
            pixel_mask,
            output_hidden_states=output_hidden_states or self.config.use_auxiliary_loss,
            return_dict=return_dict,
            output_attentions=output_attentions,
        )

        outputs = MaskFormerModelOutput(
            encoder_last_hidden_state=raw_outputs[0],
            pixel_decoder_last_hidden_state=raw_outputs[1],
            transformer_decoder_last_hidden_state=raw_outputs[2],
            encoder_hidden_states=raw_outputs[3] if output_hidden_states else None,
            pixel_decoder_hidden_states=raw_outputs[4] if output_hidden_states else None,
            transformer_decoder_hidden_states=raw_outputs[5] if output_hidden_states else None,
            hidden_states=raw_outputs[6] if output_hidden_states else None,
            attentions=raw_outputs[-1] if output_attentions else None,
        )

        loss, loss_dict, auxiliary_logits = None, None, None

        class_queries_logits, masks_queries_logits, auxiliary_logits = self.get_logits(outputs)

        if mask_labels is not None and class_labels is not None:
            loss_dict: Dict[str, Tensor] = self.get_loss_dict(
                masks_queries_logits, class_queries_logits, mask_labels, class_labels, auxiliary_logits
            )
            loss = self.get_loss(loss_dict)

        output_auxiliary_logits = (
            self.config.output_auxiliary_logits if output_auxiliary_logits is None else output_auxiliary_logits
        )
        if not output_auxiliary_logits:
            auxiliary_logits = None

        if not return_dict:
            output = tuple(
                v
                for v in (loss, class_queries_logits, masks_queries_logits, auxiliary_logits, *outputs.values())
                if v is not None
            )
            return output

        return MaskFormerForInstanceSegmentationOutput(
            loss=loss,
            **outputs,
            class_queries_logits=class_queries_logits,
            masks_queries_logits=masks_queries_logits,
            auxiliary_logits=auxiliary_logits,
        )


__all__ = ["MaskFormerForInstanceSegmentation", "MaskFormerModel", "MaskFormerPreTrainedModel"]
