# coding=utf-8
# Copyright 2022 Microsoft Research Asia and The HuggingFace Inc. team. All rights reserved.
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
"""MindSpore Conditional DETR model."""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from transformers.models.conditional_detr.configuration_conditional_detr import ConditionalDetrConfig
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
    requires_backends,
)

import mindspore as ms
from mindspore import Tensor, mint, nn
from mindspore.common.initializer import Constant, HeUniform, Normal, Uniform, XavierUniform

from ...activations import ACT2FN
from ...mindspore_adapter import dtype_to_max
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithCrossAttentions, Seq2SeqModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils.backbone_utils import load_backbone

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "ConditionalDetrConfig"
_CHECKPOINT_FOR_DOC = "microsoft/conditional-detr-resnet-50"


@dataclass
class ConditionalDetrDecoderOutput(BaseModelOutputWithCrossAttentions):
    """
    Base class for outputs of the Conditional DETR decoder. This class adds one attribute to
    BaseModelOutputWithCrossAttentions, namely an optional stack of intermediate decoder activations, i.e. the output
    of each decoder layer, each of them gone through a layernorm. This is useful when training the model with auxiliary
    decoding losses.

    Args:
        last_hidden_state (`ms.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(ms.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `ms.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(ms.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `ms.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        cross_attentions (`tuple(ms.Tensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `ms.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax,
            used to compute the weighted average in the cross-attention heads.
        intermediate_hidden_states (`ms.Tensor` of shape `(config.decoder_layers, batch_size, num_queries, hidden_size)`, *optional*, returned when `config.auxiliary_loss=True`):
            Intermediate decoder activations, i.e. the output of each decoder layer, each of them gone through a
            layernorm.
        reference_points (`ms.Tensor` of shape `(config.decoder_layers, batch_size, num_queries, 2 (anchor points))`):
            Reference points (reference points of each layer of the decoder).
    """

    intermediate_hidden_states: Optional[ms.Tensor] = None
    reference_points: Optional[Tuple[ms.Tensor]] = None


@dataclass
class ConditionalDetrModelOutput(Seq2SeqModelOutput):
    """
    Base class for outputs of the Conditional DETR encoder-decoder model. This class adds one attribute to
    Seq2SeqModelOutput, namely an optional stack of intermediate decoder activations, i.e. the output of each decoder
    layer, each of them gone through a layernorm. This is useful when training the model with auxiliary decoding
    losses.

    Args:
        last_hidden_state (`ms.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.
        decoder_hidden_states (`tuple(ms.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `ms.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the decoder at the output of each
            layer plus the initial embedding outputs.
        decoder_attentions (`tuple(ms.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `ms.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder, after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        cross_attentions (`tuple(ms.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `ms.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax,
            used to compute the weighted average in the cross-attention heads.
        encoder_last_hidden_state (`ms.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(ms.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `ms.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the encoder at the output of each
            layer plus the initial embedding outputs.
        encoder_attentions (`tuple(ms.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `ms.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the encoder, after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        intermediate_hidden_states (`ms.Tensor` of shape `(config.decoder_layers, batch_size, sequence_length, hidden_size)`, *optional*, returned when `config.auxiliary_loss=True`):
            Intermediate decoder activations, i.e. the output of each decoder layer, each of them gone through a
            layernorm.
        reference_points (`ms.Tensor` of shape `(config.decoder_layers, batch_size, num_queries, 2 (anchor points))`):
            Reference points (reference points of each layer of the decoder).
    """

    intermediate_hidden_states: Optional[ms.Tensor] = None
    reference_points: Optional[Tuple[ms.Tensor]] = None


@dataclass
# Copied from transformers.models.detr.modeling_detr.DetrObjectDetectionOutput with Detr->ConditionalDetr
class ConditionalDetrObjectDetectionOutput(ModelOutput):
    """
    Output type of [`ConditionalDetrForObjectDetection`].

    Args:
        loss (`ms.Tensor` of shape `(1,)`, *optional*, returned when `labels` are provided)):
            Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
            bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
            scale-invariant IoU loss.
        loss_dict (`Dict`, *optional*):
            A dictionary containing the individual losses. Useful for logging.
        logits (`ms.Tensor` of shape `(batch_size, num_queries, num_classes + 1)`):
            Classification logits (including no-object) for all queries.
        pred_boxes (`ms.Tensor` of shape `(batch_size, num_queries, 4)`):
            Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
            values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
            possible padding). You can use [`~ConditionalDetrImageProcessor.post_process_object_detection`] to retrieve the
            unnormalized bounding boxes.
        auxiliary_outputs (`list[Dict]`, *optional*):
            Optional, only returned when auxilary losses are activated (i.e. `config.auxiliary_loss` is set to `True`)
            and labels are provided. It is a list of dictionaries containing the two above keys (`logits` and
            `pred_boxes`) for each decoder layer.
        last_hidden_state (`ms.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.
        decoder_hidden_states (`tuple(ms.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `ms.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the decoder at the output of each
            layer plus the initial embedding outputs.
        decoder_attentions (`tuple(ms.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `ms.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder, after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        cross_attentions (`tuple(ms.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `ms.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax,
            used to compute the weighted average in the cross-attention heads.
        encoder_last_hidden_state (`ms.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(ms.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `ms.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the encoder at the output of each
            layer plus the initial embedding outputs.
        encoder_attentions (`tuple(ms.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `ms.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the encoder, after the attention softmax, used to compute the
            weighted average in the self-attention heads.
    """

    loss: Optional[ms.Tensor] = None
    loss_dict: Optional[Dict] = None
    logits: ms.Tensor = None
    pred_boxes: ms.Tensor = None
    auxiliary_outputs: Optional[List[Dict]] = None
    last_hidden_state: Optional[ms.Tensor] = None
    decoder_hidden_states: Optional[Tuple[ms.Tensor]] = None
    decoder_attentions: Optional[Tuple[ms.Tensor]] = None
    cross_attentions: Optional[Tuple[ms.Tensor]] = None
    encoder_last_hidden_state: Optional[ms.Tensor] = None
    encoder_hidden_states: Optional[Tuple[ms.Tensor]] = None
    encoder_attentions: Optional[Tuple[ms.Tensor]] = None


@dataclass
# Copied from transformers.models.detr.modeling_detr.DetrSegmentationOutput with Detr->ConditionalDetr
class ConditionalDetrSegmentationOutput(ModelOutput):
    """
    Output type of [`ConditionalDetrForSegmentation`].

    Args:
        loss (`ms.Tensor` of shape `(1,)`, *optional*, returned when `labels` are provided)):
            Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
            bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
            scale-invariant IoU loss.
        loss_dict (`Dict`, *optional*):
            A dictionary containing the individual losses. Useful for logging.
        logits (`ms.Tensor` of shape `(batch_size, num_queries, num_classes + 1)`):
            Classification logits (including no-object) for all queries.
        pred_boxes (`ms.Tensor` of shape `(batch_size, num_queries, 4)`):
            Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
            values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
            possible padding). You can use [`~ConditionalDetrImageProcessor.post_process_object_detection`] to retrieve the
            unnormalized bounding boxes.
        pred_masks (`ms.Tensor` of shape `(batch_size, num_queries, height/4, width/4)`):
            Segmentation masks logits for all queries. See also
            [`~ConditionalDetrImageProcessor.post_process_semantic_segmentation`] or
            [`~ConditionalDetrImageProcessor.post_process_instance_segmentation`]
            [`~ConditionalDetrImageProcessor.post_process_panoptic_segmentation`] to evaluate semantic, instance and panoptic
            segmentation masks respectively.
        auxiliary_outputs (`list[Dict]`, *optional*):
            Optional, only returned when auxiliary losses are activated (i.e. `config.auxiliary_loss` is set to `True`)
            and labels are provided. It is a list of dictionaries containing the two above keys (`logits` and
            `pred_boxes`) for each decoder layer.
        last_hidden_state (`ms.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.
        decoder_hidden_states (`tuple(ms.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `ms.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the decoder at the output of each
            layer plus the initial embedding outputs.
        decoder_attentions (`tuple(ms.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `ms.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder, after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        cross_attentions (`tuple(ms.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `ms.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax,
            used to compute the weighted average in the cross-attention heads.
        encoder_last_hidden_state (`ms.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(ms.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `ms.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the encoder at the output of each
            layer plus the initial embedding outputs.
        encoder_attentions (`tuple(ms.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `ms.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the encoder, after the attention softmax, used to compute the
            weighted average in the self-attention heads.
    """

    loss: Optional[ms.Tensor] = None
    loss_dict: Optional[Dict] = None
    logits: ms.Tensor = None
    pred_boxes: ms.Tensor = None
    pred_masks: ms.Tensor = None
    auxiliary_outputs: Optional[List[Dict]] = None
    last_hidden_state: Optional[ms.Tensor] = None
    decoder_hidden_states: Optional[Tuple[ms.Tensor]] = None
    decoder_attentions: Optional[Tuple[ms.Tensor]] = None
    cross_attentions: Optional[Tuple[ms.Tensor]] = None
    encoder_last_hidden_state: Optional[ms.Tensor] = None
    encoder_hidden_states: Optional[Tuple[ms.Tensor]] = None
    encoder_attentions: Optional[Tuple[ms.Tensor]] = None


# Copied from transformers.models.detr.modeling_detr.DetrFrozenBatchNorm2d with Detr->ConditionalDetr
class ConditionalDetrFrozenBatchNorm2d(nn.Cell):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt, without which any other models than
    torchvision.models.resnet[18,34,50,101] produce nans.
    """

    def __init__(self, n):
        super().__init__()
        # self.register_buffer("weight", mint.ones((n,)))
        # self.register_buffer("bias", mint.zeros((n,)))
        # self.register_buffer("running_mean", mint.zeros((n,)))
        # self.register_buffer("running_var", mint.ones((n,)))
        self.weight = ms.Parameter(mint.ones((n,)), name="weight")
        self.bias = ms.Parameter(mint.zeros((n,)), name="bias")
        self.running_mean = ms.Parameter(mint.zeros((n,)), name="running_mean")
        self.running_var = ms.Parameter(mint.ones((n,)), name="running_var")

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def construct(self, x):
        # move reshapes to the beginning
        # to make it user-friendly
        weight = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        running_var = self.running_var.reshape(1, -1, 1, 1)
        running_mean = self.running_mean.reshape(1, -1, 1, 1)
        epsilon = 1e-5
        scale = weight * (running_var + epsilon).rsqrt()
        bias = bias - running_mean * scale
        return x * scale + bias


# Copied from transformers.models.detr.modeling_detr.replace_batch_norm with Detr->ConditionalDetr
def replace_batch_norm(model):
    r"""
    Recursively replace all `mindspore.mint.nn.BatchNorm2d` with `ConditionalDetrFrozenBatchNorm2d`.

    Args:
        model (mindspore.nn.Cell):
            input model
    """
    for name, module in model.name_cells().items():
        if isinstance(module, mint.nn.BatchNorm2d):
            new_module = ConditionalDetrFrozenBatchNorm2d(module.num_features)

            new_module.weight.copy_(module.weight)
            new_module.bias.copy_(module.bias)
            new_module.running_mean.copy_(module.running_mean)
            new_module.running_var.copy_(module.running_var)

            model._cells[name] = new_module

        if len(list(module.cells())) > 0:
            replace_batch_norm(module)


# Copied from transformers.models.detr.modeling_detr.DetrConvEncoder with Detr->ConditionalDetr
class ConditionalDetrConvEncoder(nn.Cell):
    """
    Convolutional backbone, using either the AutoBackbone API or one from the timm library.

    nn.BatchNorm2d layers are replaced by ConditionalDetrFrozenBatchNorm2d as defined above.

    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        backbone = load_backbone(config)

        # replace batch norm by frozen batch norm
        replace_batch_norm(backbone)
        self.model = backbone
        self.intermediate_channel_sizes = self.model.channels

        backbone_model_type = None
        if config.backbone is not None:
            backbone_model_type = config.backbone
        elif config.backbone_config is not None:
            backbone_model_type = config.backbone_config.model_type
        else:
            raise ValueError("Either `backbone` or `backbone_config` should be provided in the config")

        if "resnet" in backbone_model_type:
            for name, parameter in self.model.parameters_and_names():
                if "stage.1" not in name and "stage.2" not in name and "stage.3" not in name:
                    parameter.requires_grad = False

    def construct(self, pixel_values: ms.Tensor, pixel_mask: ms.Tensor):
        # send pixel_values through the model to get list of feature maps
        features = self.model(pixel_values).feature_maps

        out = []
        for feature_map in features:
            # downsample pixel_mask to match shape of corresponding feature_map
            mask = mint.nn.functional.interpolate(pixel_mask[None].float(), size=feature_map.shape[-2:]).to(ms.bool_)[0]
            out.append((feature_map, mask))
        return out


# Copied from transformers.models.detr.modeling_detr.DetrConvModel with Detr->ConditionalDetr
class ConditionalDetrConvModel(nn.Cell):
    """
    This module adds 2D position embeddings to all intermediate feature maps of the convolutional encoder.
    """

    def __init__(self, conv_encoder, position_embedding):
        super().__init__()
        self.conv_encoder = conv_encoder
        self.position_embedding = position_embedding

    def construct(self, pixel_values, pixel_mask):
        # send pixel_values and pixel_mask through backbone to get list of (feature_map, pixel_mask) tuples
        out = self.conv_encoder(pixel_values, pixel_mask)
        pos = []
        for feature_map, mask in out:
            # position encoding
            pos.append(self.position_embedding(feature_map, mask).to(feature_map.dtype))

        return out, pos


class ConditionalDetrSinePositionEmbedding(nn.Cell):
    """
    This is a more standard version of the position embedding, very similar to the one used by the Attention is all you
    need paper, generalized to work on images.
    """

    def __init__(self, embedding_dim=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def construct(self, pixel_values, pixel_mask):
        if pixel_mask is None:
            raise ValueError("No pixel mask provided")
        y_embed = pixel_mask.cumsum(1, dtype=ms.float32)
        x_embed = pixel_mask.cumsum(2, dtype=ms.float32)
        if self.normalize:
            y_embed = y_embed / (y_embed[:, -1:, :] + 1e-6) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + 1e-6) * self.scale

        dim_t = mint.arange(self.embedding_dim, dtype=ms.int64).float()
        dim_t = self.temperature ** (2 * mint.div(dim_t, 2, rounding_mode="floor") / self.embedding_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = mint.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = mint.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = mint.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


# Copied from transformers.models.detr.modeling_detr.DetrLearnedPositionEmbedding with Detr->ConditionalDetr
class ConditionalDetrLearnedPositionEmbedding(nn.Cell):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, embedding_dim=256):
        super().__init__()
        self.row_embeddings = mint.nn.Embedding(50, embedding_dim)
        self.column_embeddings = mint.nn.Embedding(50, embedding_dim)

    def construct(self, pixel_values, pixel_mask=None):
        height, width = pixel_values.shape[-2:]
        width_values = mint.arange(width)
        height_values = mint.arange(height)
        x_emb = self.column_embeddings(width_values)
        y_emb = self.row_embeddings(height_values)
        pos = mint.cat([x_emb.unsqueeze(0).repeat(height, 1, 1), y_emb.unsqueeze(1).repeat(1, width, 1)], dim=-1)
        pos = pos.permute(2, 0, 1)
        pos = pos.unsqueeze(0)
        pos = pos.repeat(pixel_values.shape[0], 1, 1, 1)
        return pos


# Copied from transformers.models.detr.modeling_detr.build_position_encoding with Detr->ConditionalDetr
def build_position_encoding(config):
    n_steps = config.d_model // 2
    if config.position_embedding_type == "sine":
        # TODO find a better way of exposing other arguments
        position_embedding = ConditionalDetrSinePositionEmbedding(n_steps, normalize=True)
    elif config.position_embedding_type == "learned":
        position_embedding = ConditionalDetrLearnedPositionEmbedding(n_steps)
    else:
        raise ValueError(f"Not supported {config.position_embedding_type}")

    return position_embedding


# function to generate sine positional embedding for 2d coordinates
def gen_sine_position_embeddings(pos_tensor, d_model):
    scale = 2 * math.pi
    dim = d_model // 2
    dim_t = mint.arange(dim, dtype=ms.float32)
    dim_t = 10000 ** (2 * mint.div(dim_t, 2, rounding_mode="floor") / dim)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t.to(x_embed.dtype)
    pos_y = y_embed[:, :, None] / dim_t.to(y_embed.dtype)
    pos_x = mint.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = mint.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = mint.cat((pos_y, pos_x), dim=2)
    return pos


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return mint.log(x1 / x2)


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
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

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

        attn_weights = mint.bmm(query_states, key_states.transpose(1, 2))

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
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, target_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


class ConditionalDetrAttention(nn.Cell):
    """
    Cross-Attention used in Conditional DETR 'Conditional DETR for Fast Training Convergence' paper.

    The key q_proj, k_proj, v_proj are defined outside the attention. This attention allows the dim of q, k to be
    different to v.
    """

    def __init__(
        self,
        embed_dim: int,
        out_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {num_heads})."
            )
        # head dimension of values
        self.v_head_dim = out_dim // num_heads
        if self.v_head_dim * num_heads != self.out_dim:
            raise ValueError(
                f"out_dim must be divisible by num_heads (got `out_dim`: {self.out_dim} and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.out_proj = mint.nn.Linear(out_dim, out_dim, bias=bias)

    def _qk_shape(self, tensor: ms.Tensor, seq_len: int, batch_size: int):
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _v_shape(self, tensor: ms.Tensor, seq_len: int, batch_size: int):
        return tensor.view(batch_size, seq_len, self.num_heads, self.v_head_dim).transpose(1, 2).contiguous()

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        key_states: Optional[ms.Tensor] = None,
        value_states: Optional[ms.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[ms.Tensor, Optional[ms.Tensor], Optional[Tuple[ms.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, target_len, _ = hidden_states.shape

        # get query proj
        query_states = hidden_states * self.scaling
        # get key, value proj
        key_states = self._qk_shape(key_states, -1, batch_size)
        value_states = self._v_shape(value_states, -1, batch_size)

        proj_shape = (batch_size * self.num_heads, -1, self.head_dim)
        v_proj_shape = (batch_size * self.num_heads, -1, self.v_head_dim)
        query_states = self._qk_shape(query_states, target_len, batch_size).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*v_proj_shape)

        source_len = key_states.shape[1]

        attn_weights = mint.bmm(query_states, key_states.transpose(1, 2))

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

        if attn_output.shape != (batch_size * self.num_heads, target_len, self.v_head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, target_len, self.v_head_dim)}, but is"
                f" {attn_output.shape}"
            )

        attn_output = attn_output.view(batch_size, self.num_heads, target_len, self.v_head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, target_len, self.out_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


# Copied from transformers.models.detr.modeling_detr.DetrEncoderLayer with DetrEncoderLayer->ConditionalDetrEncoderLayer,DetrConfig->ConditionalDetrConfig
class ConditionalDetrEncoderLayer(nn.Cell):
    def __init__(self, config: ConditionalDetrConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = DetrAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = mint.nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = mint.nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = mint.nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = mint.nn.LayerNorm(self.embed_dim)

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: ms.Tensor,
        object_queries: ms.Tensor = None,
        output_attentions: bool = False,
    ):
        """
        Args:
            hidden_states (`ms.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`ms.Tensor`): attention mask of size
                `(batch, 1, target_len, source_len)` where padding elements are indicated by very large negative
                values.
            object_queries (`ms.Tensor`, *optional*):
                Object queries (also called content embeddings), to be added to the hidden states.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            object_queries=object_queries,
            output_attentions=output_attentions,
        )

        hidden_states = mint.nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = mint.nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)

        hidden_states = self.fc2(hidden_states)
        hidden_states = mint.nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if self.training:
            if mint.isinf(hidden_states).any() or mint.isnan(hidden_states).any():
                clamp_value = dtype_to_max(hidden_states.dtype) - 1000
                hidden_states = mint.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class ConditionalDetrDecoderLayer(nn.Cell):
    def __init__(self, config: ConditionalDetrConfig):
        super().__init__()
        self.embed_dim = config.d_model

        d_model = config.d_model
        # Decoder Self-Attention projections
        self.sa_qcontent_proj = mint.nn.Linear(d_model, d_model)
        self.sa_qpos_proj = mint.nn.Linear(d_model, d_model)
        self.sa_kcontent_proj = mint.nn.Linear(d_model, d_model)
        self.sa_kpos_proj = mint.nn.Linear(d_model, d_model)
        self.sa_v_proj = mint.nn.Linear(d_model, d_model)

        self.self_attn = ConditionalDetrAttention(
            embed_dim=self.embed_dim,
            out_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = mint.nn.LayerNorm(self.embed_dim)

        # Decoder Cross-Attention projections
        self.ca_qcontent_proj = mint.nn.Linear(d_model, d_model)
        self.ca_qpos_proj = mint.nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = mint.nn.Linear(d_model, d_model)
        self.ca_kpos_proj = mint.nn.Linear(d_model, d_model)
        self.ca_v_proj = mint.nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = mint.nn.Linear(d_model, d_model)

        self.encoder_attn = ConditionalDetrAttention(
            self.embed_dim * 2, self.embed_dim, config.decoder_attention_heads, dropout=config.attention_dropout
        )
        self.encoder_attn_layer_norm = mint.nn.LayerNorm(self.embed_dim)
        self.fc1 = mint.nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = mint.nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = mint.nn.LayerNorm(self.embed_dim)
        self.nhead = config.decoder_attention_heads

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        object_queries: Optional[ms.Tensor] = None,
        query_position_embeddings: Optional[ms.Tensor] = None,
        query_sine_embed: Optional[ms.Tensor] = None,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        encoder_attention_mask: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = False,
        is_first: Optional[bool] = False,
    ):
        """
        Args:
            hidden_states (`ms.Tensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (`ms.Tensor`): attention mask of size
                `(batch, 1, target_len, source_len)` where padding elements are indicated by very large negative
                values.
            object_queries (`ms.Tensor`, *optional*):
                object_queries that are added to the queries and keys
            in the cross-attention layer.
            query_position_embeddings (`ms.Tensor`, *optional*):
                object_queries that are added to the queries and keys
            in the self-attention layer.
            encoder_hidden_states (`ms.Tensor`):
                cross attention input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_attention_mask (`ms.Tensor`): encoder attention mask of size
                `(batch, 1, target_len, source_len)` where padding elements are indicated by very large negative
                values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # ========== Begin of Self-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.sa_qcontent_proj(
            hidden_states
        )  # target is the input of the first decoder layer. zero by default.
        q_pos = self.sa_qpos_proj(query_position_embeddings)
        k_content = self.sa_kcontent_proj(hidden_states)
        k_pos = self.sa_kpos_proj(query_position_embeddings)
        v = self.sa_v_proj(hidden_states)

        _, num_queries, n_model = q_content.shape

        q = q_content + q_pos
        k = k_content + k_pos
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=q,
            attention_mask=attention_mask,
            key_states=k,
            value_states=v,
            output_attentions=output_attentions,
        )
        # ============ End of Self-Attention =============

        hidden_states = mint.nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.ca_qcontent_proj(hidden_states)
        k_content = self.ca_kcontent_proj(encoder_hidden_states)
        v = self.ca_v_proj(encoder_hidden_states)

        batch_size, num_queries, n_model = q_content.shape
        _, source_len, _ = k_content.shape

        k_pos = self.ca_kpos_proj(object_queries)

        # For the first decoder layer, we concatenate the positional embedding predicted from
        # the object query (the positional embedding) into the original query (key) in DETR.
        if is_first:
            q_pos = self.ca_qpos_proj(query_position_embeddings)
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content

        q = q.view(batch_size, num_queries, self.nhead, n_model // self.nhead)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(batch_size, num_queries, self.nhead, n_model // self.nhead)
        q = mint.cat([q, query_sine_embed], dim=3).view(batch_size, num_queries, n_model * 2)
        k = k.view(batch_size, source_len, self.nhead, n_model // self.nhead)
        k_pos = k_pos.view(batch_size, source_len, self.nhead, n_model // self.nhead)
        k = mint.cat([k, k_pos], dim=3).view(batch_size, source_len, n_model * 2)

        # Cross-Attention Block
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            hidden_states, cross_attn_weights = self.encoder_attn(
                hidden_states=q,
                attention_mask=encoder_attention_mask,
                key_states=k,
                value_states=v,
                output_attentions=output_attentions,
            )

            hidden_states = mint.nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # ============ End of Cross-Attention =============

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


# Copied from transformers.models.detr.modeling_detr.DetrMLPPredictionHead with DetrMLPPredictionHead->MLP
class MLP(nn.Cell):
    """
    Very simple multi-layer perceptron (MLP, also called FFN), used to predict the normalized center coordinates,
    height and width of a bounding box w.r.t. an image.

    Copied from https://github.com/facebookresearch/detr/blob/master/models/detr.py

    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.CellList([mint.nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])])

    def construct(self, x):
        for i, layer in enumerate(self.layers):
            x = mint.nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# Copied from transformers.models.detr.modeling_detr.DetrPreTrainedModel with Detr->ConditionalDetr
class ConditionalDetrPreTrainedModel(PreTrainedModel):
    config_class = ConditionalDetrConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"
    _no_split_modules = [r"ConditionalDetrConvEncoder", r"ConditionalDetrEncoderLayer", r"ConditionalDetrDecoderLayer"]

    def _init_weights(self, module):
        std = self.config.init_std
        xavier_std = self.config.init_xavier_std

        if isinstance(module, ConditionalDetrMHAttentionMap):
            module.k_linear.bias.set_data(
                initializer(Constant(0), module.k_linear.bias.shape, module.k_linear.bias.dtype)
            )
            module.q_linear.bias.set_data(
                initializer(Constant(0), module.q_linear.bias.shape, module.q_linear.bias.dtype)
            )
            module.k_linear.weight.set_data(
                initializer(XavierUniform(xavier_std), module.k_linear.weight.shape, module.k_linear.weight.dtype)
            )
            module.q_linear.weight.set_data(
                initializer(XavierUniform(xavier_std), module.q_linear.weight.shape, module.q_linear.weight.dtype)
            )
        elif isinstance(module, ConditionalDetrLearnedPositionEmbedding):
            module.row_embeddings.weight.set_data(
                initializer(Uniform(), module.row_embeddings.weight.shape, module.row_embeddings.weight.dtype)
            )
            module.column_embeddings.weight.set_data(
                initializer(Uniform(), module.column_embeddings.weight.shape, module.column_embeddings.weight.dtype)
            )
        if isinstance(module, (mint.nn.Linear, mint.nn.Conv2d, mint.nn.BatchNorm2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, mint.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


CONDITIONAL_DETR_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a MindSpore [mindspore.nn.Cell](https://www.mindspore.cn/docs/zh-CN/r2.3.1/api_python/nn/mindspore.nn.Cell.html) subclass.
    Use it as a regular MindSpore Cell and refer to the MindSpore documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`ConditionalDetrConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

CONDITIONAL_DETR_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`ms.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it.

            Pixel values can be obtained using [`AutoImageProcessor`]. See [`ConditionalDetrImageProcessor.__call__`]
            for details.

        pixel_mask (`ms.Tensor` of shape `(batch_size, height, width)`, *optional*):
            Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:

            - 1 for pixels that are real (i.e. **not masked**),
            - 0 for pixels that are padding (i.e. **masked**).

            [What are attention masks?](../glossary#attention-mask)

        decoder_attention_mask (`ms.Tensor` of shape `(batch_size, num_queries)`, *optional*):
            Not used by default. Can be used to mask object queries.
        encoder_outputs (`tuple(tuple(ms.Tensor)`, *optional*):
            Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
            hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
        inputs_embeds (`ms.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing the flattened feature map (output of the backbone + projection layer), you
            can choose to directly pass a flattened representation of an image.
        decoder_inputs_embeds (`ms.Tensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*):
            Optionally, instead of initializing the queries with a tensor of zeros, you can choose to directly pass an
            embedded representation.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


# Copied from transformers.models.detr.modeling_detr.DetrEncoder with Detr->ConditionalDetr,DETR->ConditionalDETR
class ConditionalDetrEncoder(ConditionalDetrPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`ConditionalDetrEncoderLayer`].

    The encoder updates the flattened feature map through multiple self-attention layers.

    Small tweak for ConditionalDETR:

    - object_queries are added to the forward pass.

    Args:
        config: ConditionalDetrConfig
    """

    def __init__(self, config: ConditionalDetrConfig):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        self.layers = nn.CellList([ConditionalDetrEncoderLayer(config) for _ in range(config.encoder_layers)])

        # in the original ConditionalDETR, no layernorm is used at the end of the encoder, as "normalize_before" is set to False by default

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        inputs_embeds=None,
        attention_mask=None,
        object_queries=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            inputs_embeds (`ms.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Flattened feature map (output of the backbone + projection layer) that is passed to the encoder.

            attention_mask (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding pixel features. Mask values selected in `[0, 1]`:

                - 1 for pixel features that are real (i.e. **not masked**),
                - 0 for pixel features that are padding (i.e. **masked**).

                [What are attention masks?](../glossary#attention-mask)

            object_queries (`ms.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Object queries that are added to the queries in each self-attention layer.

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

        hidden_states = inputs_embeds
        hidden_states = mint.nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [batch_size, seq_len] -> [batch_size, 1, target_seq_len, source_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = mint.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None)
            else:
                # we add object_queries as extra input to the encoder_layer
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    object_queries=object_queries,
                    output_attentions=output_attentions,
                )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions)


class ConditionalDetrDecoder(ConditionalDetrPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`ConditionalDetrDecoderLayer`].

    The decoder updates the query embeddings through multiple self-attention and cross-attention layers.

    Some small tweaks for Conditional DETR:

    - object_queries and query_position_embeddings are added to the forward pass.
    - if self.config.auxiliary_loss is set to True, also returns a stack of activations from all decoding layers.

    Args:
        config: ConditionalDetrConfig
    """

    def __init__(self, config: ConditionalDetrConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop

        self.layers = nn.CellList([ConditionalDetrDecoderLayer(config) for _ in range(config.decoder_layers)])
        # in Conditional DETR, the decoder uses layernorm after the last decoder layer output
        self.layernorm = mint.nn.LayerNorm(config.d_model)
        d_model = config.d_model
        self.gradient_checkpointing = False

        # query_scale is the FFN applied on f to generate transformation T
        self.query_scale = MLP(d_model, d_model, d_model, 2)
        self.ref_point_head = MLP(d_model, d_model, 2, 2)
        for layer_id in range(config.decoder_layers - 1):
            self.layers[layer_id + 1].ca_qpos_proj = None

        # Initialize weights and apply final processing
        self.post_init()

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
            # [batch_size, seq_len] -> [batch_size, 1, target_seq_len, source_seq_len]
            encoder_attention_mask = _prepare_4d_attention_mask(
                encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            )

        # optional intermediate hidden states
        intermediate = () if self.config.auxiliary_loss else None

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        reference_points_before_sigmoid = self.ref_point_head(query_position_embeddings)  # [num_queries, batch_size, 2]
        reference_points = reference_points_before_sigmoid.sigmoid().transpose(0, 1)
        obj_center = reference_points[..., :2].transpose(0, 1)
        # get sine embedding for the query vector
        query_sine_embed_before_transformation = gen_sine_position_embeddings(obj_center, self.config.d_model)

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = mint.rand([])
                if dropout_probability < self.layerdrop:
                    continue
            if idx == 0:
                pos_transformation = 1
            else:
                pos_transformation = self.query_scale(hidden_states)
            # apply transformation
            query_sine_embed = query_sine_embed_before_transformation * pos_transformation
            if self.gradient_checkpointing and self.training:
                raise NotImplementedError("Gradient checkpointing is not yet supported.")
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=None,
                    object_queries=object_queries,
                    query_position_embeddings=query_position_embeddings,
                    query_sine_embed=query_sine_embed,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    output_attentions=output_attentions,
                    is_first=(idx == 0),
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
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attns,
                    all_cross_attentions,
                    intermediate,
                    reference_points,
                ]
                if v is not None
            )
        return ConditionalDetrDecoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
            intermediate_hidden_states=intermediate,
            reference_points=reference_points,
        )


@add_start_docstrings(
    """
    The bare Conditional DETR Model (consisting of a backbone and encoder-decoder Transformer) outputting raw
    hidden-states without any specific head on top.
    """,
    CONDITIONAL_DETR_START_DOCSTRING,
)
class ConditionalDetrModel(ConditionalDetrPreTrainedModel):
    def __init__(self, config: ConditionalDetrConfig):
        super().__init__(config)

        # Create backbone + positional encoding
        backbone = ConditionalDetrConvEncoder(config)
        object_queries = build_position_encoding(config)
        self.backbone = ConditionalDetrConvModel(backbone, object_queries)

        # Create projection layer
        self.input_projection = mint.nn.Conv2d(backbone.intermediate_channel_sizes[-1], config.d_model, kernel_size=1)

        self.query_position_embeddings = mint.nn.Embedding(config.num_queries, config.d_model)

        self.encoder = ConditionalDetrEncoder(config)
        self.decoder = ConditionalDetrDecoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def freeze_backbone(self):
        for name, param in self.backbone.conv_encoder.model.named_parameters():
            param.requires_grad_(False)

    def unfreeze_backbone(self):
        for name, param in self.backbone.conv_encoder.model.named_parameters():
            param.requires_grad_(True)

    @add_start_docstrings_to_model_forward(CONDITIONAL_DETR_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ConditionalDetrModelOutput, config_class=_CONFIG_FOR_DOC)
    def construct(
        self,
        pixel_values: ms.Tensor,
        pixel_mask: Optional[ms.Tensor] = None,
        decoder_attention_mask: Optional[ms.Tensor] = None,
        encoder_outputs: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        decoder_inputs_embeds: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[ms.Tensor], ConditionalDetrModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, AutoModel
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("microsoft/conditional-detr-resnet-50")
        >>> model = AutoModel.from_pretrained("microsoft/conditional-detr-resnet-50")

        >>> # prepare image for the model
        >>> inputs = image_processor(images=image, return_tensors="np")
        >>> for k, v in inputs.items():
        ...     inputs[k] = ms.tensor(v)

        >>> # forward pass
        >>> outputs = model(**inputs)

        >>> # the last hidden states are the final query embeddings of the Transformer decoder
        >>> # these are of shape (batch_size, num_queries, hidden_size)
        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 300, 256]
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, num_channels, height, width = pixel_values.shape

        if pixel_mask is None:
            pixel_mask = mint.ones(
                ((batch_size, height, width)),
            )

        # First, sent pixel_values + pixel_mask through Backbone to obtain the features
        # pixel_values should be of shape (batch_size, num_channels, height, width)
        # pixel_mask should be of shape (batch_size, height, width)
        features, object_queries_list = self.backbone(pixel_values, pixel_mask)

        # get final feature map and downsampled mask
        feature_map, mask = features[-1]

        if mask is None:
            raise ValueError("Backbone does not return downsampled pixel mask")

        # Second, apply 1x1 convolution to reduce the channel dimension to d_model (256 by default)
        projected_feature_map = self.input_projection(feature_map)

        # Third, flatten the feature map + object_queries of shape NxCxHxW to NxCxHW, and permute it to NxHWxC
        # In other words, turn their shape into (batch_size, sequence_length, hidden_size)
        flattened_features = projected_feature_map.flatten(2).permute(0, 2, 1)
        object_queries = object_queries_list[-1].flatten(2).permute(0, 2, 1)

        flattened_mask = mask.flatten(1)

        # Fourth, sent flattened_features + flattened_mask + object_queries through encoder
        # flattened_features is a Tensor of shape (batch_size, heigth*width, hidden_size)
        # flattened_mask is a Tensor of shape (batch_size, heigth*width)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                inputs_embeds=flattened_features,
                attention_mask=flattened_mask,
                object_queries=object_queries,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # Fifth, sent query embeddings + object_queries through the decoder (which is conditioned on the encoder output)
        query_position_embeddings = self.query_position_embeddings.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        queries = mint.zeros_like(query_position_embeddings)

        # decoder outputs consists of (dec_features, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            inputs_embeds=queries,
            attention_mask=None,
            object_queries=object_queries,
            query_position_embeddings=query_position_embeddings,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=flattened_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return ConditionalDetrModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            intermediate_hidden_states=decoder_outputs.intermediate_hidden_states,
            reference_points=decoder_outputs.reference_points,
        )


# Copied from transformers.models.detr.modeling_detr.DetrMLPPredictionHead with Detr->ConditionalDetr
class ConditionalDetrMLPPredictionHead(nn.Cell):
    """
    Very simple multi-layer perceptron (MLP, also called FFN), used to predict the normalized center coordinates,
    height and width of a bounding box w.r.t. an image.

    Copied from https://github.com/facebookresearch/detr/blob/master/models/detr.py

    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.CellList([mint.nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])])

    def construct(self, x):
        for i, layer in enumerate(self.layers):
            x = mint.nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@add_start_docstrings(
    """
    CONDITIONAL_DETR Model (consisting of a backbone and encoder-decoder Transformer) with object detection heads on
    top, for tasks such as COCO detection.
    """,
    CONDITIONAL_DETR_START_DOCSTRING,
)
class ConditionalDetrForObjectDetection(ConditionalDetrPreTrainedModel):
    def __init__(self, config: ConditionalDetrConfig):
        super().__init__(config)

        # CONDITIONAL DETR encoder-decoder model
        self.model = ConditionalDetrModel(config)

        # Object detection heads
        self.class_labels_classifier = mint.nn.Linear(
            config.d_model, config.num_labels
        )  # We add one for the "no object" class
        self.bbox_predictor = ConditionalDetrMLPPredictionHead(
            input_dim=config.d_model, hidden_dim=config.d_model, output_dim=4, num_layers=3
        )

        # Initialize weights and apply final processing
        self.post_init()

    # taken from https://github.com/Atten4Vis/conditionalDETR/blob/master/models/conditional_detr.py
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    @add_start_docstrings_to_model_forward(CONDITIONAL_DETR_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ConditionalDetrObjectDetectionOutput, config_class=_CONFIG_FOR_DOC)
    def construct(
        self,
        pixel_values: ms.Tensor,
        pixel_mask: Optional[ms.Tensor] = None,
        decoder_attention_mask: Optional[ms.Tensor] = None,
        encoder_outputs: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        decoder_inputs_embeds: Optional[ms.Tensor] = None,
        labels: Optional[List[dict]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[ms.Tensor], ConditionalDetrObjectDetectionOutput]:
        r"""
        labels (`List[Dict]` of len `(batch_size,)`, *optional*):
            Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
            following 2 keys: 'class_labels' and 'boxes' (the class labels and bounding boxes of an image in the batch
            respectively). The class labels themselves should be a `ms.Tensor` of len `(number of bounding boxes
            in the image,)` and the boxes a `ms.Tensor` of shape `(number of bounding boxes in the image, 4)`.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, AutoModelForObjectDetection
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("microsoft/conditional-detr-resnet-50")
        >>> model = AutoModelForObjectDetection.from_pretrained("microsoft/conditional-detr-resnet-50")

        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)

        >>> # convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
        >>> target_sizes = ms.tensor([image.size[::-1]])
        >>> results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[
        ...     0
        ... ]
        >>> for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        ...     box = [round(i, 2) for i in box.tolist()]
        ...     print(
        ...         f"Detected {model.config.id2label[label.item()]} with confidence "
        ...         f"{round(score.item(), 3)} at location {box}"
        ...     )
        Detected remote with confidence 0.833 at location [38.31, 72.1, 177.63, 118.45]
        Detected cat with confidence 0.831 at location [9.2, 51.38, 321.13, 469.0]
        Detected cat with confidence 0.804 at location [340.3, 16.85, 642.93, 370.95]
        Detected remote with confidence 0.683 at location [334.48, 73.49, 366.37, 190.01]
        Detected couch with confidence 0.535 at location [0.52, 1.19, 640.35, 475.1]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # First, sent images through CONDITIONAL_DETR base model to obtain encoder + decoder outputs
        outputs = self.model(
            pixel_values,
            pixel_mask=pixel_mask,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # class logits + predicted bounding boxes
        logits = self.class_labels_classifier(sequence_output)

        # Index [-2] is valid only if `output_attentions` and `output_hidden_states`
        # are not specified, otherwise it will be another index which is hard to determine.
        # Leave it as is, because it's not a common case to use
        # return_dict=False + output_attentions=True / output_hidden_states=True
        reference = outputs.reference_points if return_dict else outputs[-2]
        reference_before_sigmoid = inverse_sigmoid(reference).transpose(0, 1)

        hs = sequence_output
        tmp = self.bbox_predictor(hs)
        tmp[..., :2] += reference_before_sigmoid
        pred_boxes = tmp.sigmoid()
        # pred_boxes = self.bbox_predictor(sequence_output).sigmoid()

        loss, loss_dict, auxiliary_outputs = None, None, None
        if labels is not None:
            outputs_class, outputs_coord = None, None
            if self.config.auxiliary_loss:
                outputs_coords = []
                intermediate = outputs.intermediate_hidden_states if return_dict else outputs[4]
                outputs_class = self.class_labels_classifier(intermediate)
                for lvl in range(intermediate.shape[0]):
                    tmp = self.bbox_predictor(intermediate[lvl])
                    tmp[..., :2] += reference_before_sigmoid
                    outputs_coord = tmp.sigmoid()
                    outputs_coords.append(outputs_coord)
                outputs_coord = mint.stack(outputs_coords)
            loss, loss_dict, auxiliary_outputs = self.loss_function(
                logits, labels, pred_boxes, self.config, outputs_class, outputs_coord
            )

        if not return_dict:
            if auxiliary_outputs is not None:
                output = (logits, pred_boxes) + auxiliary_outputs + outputs
            else:
                output = (logits, pred_boxes) + outputs
            return ((loss, loss_dict) + output) if loss is not None else output

        return ConditionalDetrObjectDetectionOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_boxes=pred_boxes,
            auxiliary_outputs=auxiliary_outputs,
            last_hidden_state=outputs.last_hidden_state,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


@add_start_docstrings(
    """
    CONDITIONAL_DETR Model (consisting of a backbone and encoder-decoder Transformer) with a segmentation head on top,
    for tasks such as COCO panoptic.

    """,
    CONDITIONAL_DETR_START_DOCSTRING,
)
class ConditionalDetrForSegmentation(ConditionalDetrPreTrainedModel):
    def __init__(self, config: ConditionalDetrConfig):
        super().__init__(config)

        # object detection model
        self.conditional_detr = ConditionalDetrForObjectDetection(config)

        # segmentation head
        hidden_size, number_of_heads = config.d_model, config.encoder_attention_heads
        intermediate_channel_sizes = self.conditional_detr.model.backbone.conv_encoder.intermediate_channel_sizes

        self.mask_head = ConditionalDetrMaskHeadSmallConv(
            hidden_size + number_of_heads, intermediate_channel_sizes[::-1][-3:], hidden_size
        )

        self.bbox_attention = ConditionalDetrMHAttentionMap(
            hidden_size, hidden_size, number_of_heads, dropout=0.0, std=config.init_xavier_std
        )

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(CONDITIONAL_DETR_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ConditionalDetrSegmentationOutput, config_class=_CONFIG_FOR_DOC)
    def construct(
        self,
        pixel_values: ms.Tensor,
        pixel_mask: Optional[ms.Tensor] = None,
        decoder_attention_mask: Optional[ms.Tensor] = None,
        encoder_outputs: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        decoder_inputs_embeds: Optional[ms.Tensor] = None,
        labels: Optional[List[dict]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[ms.Tensor], ConditionalDetrSegmentationOutput]:
        r"""
        labels (`List[Dict]` of len `(batch_size,)`, *optional*):
            Labels for computing the bipartite matching loss, DICE/F-1 loss and Focal loss. List of dicts, each
            dictionary containing at least the following 3 keys: 'class_labels', 'boxes' and 'masks' (the class labels,
            bounding boxes and segmentation masks of an image in the batch respectively). The class labels themselves
            should be a `ms.Tensor` of len `(number of bounding boxes in the image,)`, the boxes a
            `ms.Tensor` of shape `(number of bounding boxes in the image, 4)` and the masks a
            `ms.Tensor` of shape `(number of bounding boxes in the image, height, width)`.

        Returns:

        Examples:

        ```python
        >>> import io
        >>> import requests
        >>> from PIL import Image
        >>> import torch
        >>> import numpy

        >>> from transformers import (
        ...     AutoImageProcessor,
        ...     ConditionalDetrConfig,
        ...     ConditionalDetrForSegmentation,
        ... )
        >>> from transformers.image_transforms import rgb_to_id

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("microsoft/conditional-detr-resnet-50")

        >>> # randomly initialize all weights of the model
        >>> config = ConditionalDetrConfig()
        >>> model = ConditionalDetrForSegmentation(config)

        >>> # prepare image for the model
        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> # forward pass
        >>> outputs = model(**inputs)

        >>> # Use the `post_process_panoptic_segmentation` method of the `image_processor` to retrieve post-processed panoptic segmentation maps
        >>> # Segmentation results are returned as a list of dictionaries
        >>> result = image_processor.post_process_panoptic_segmentation(outputs, target_sizes=[(300, 500)])
        >>> # A tensor of shape (height, width) where each value denotes a segment id, filled with -1 if no segment is found
        >>> panoptic_seg = result[0]["segmentation"]
        >>> # Get prediction score and segment_id to class_id mapping of each segment
        >>> panoptic_segments_info = result[0]["segments_info"]
        ```"""

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, num_channels, height, width = pixel_values.shape

        if pixel_mask is None:
            pixel_mask = mint.ones((batch_size, height, width))

        # First, get list of feature maps and object_queries
        features, object_queries_list = self.conditional_detr.model.backbone(pixel_values, pixel_mask=pixel_mask)

        # Second, apply 1x1 convolution to reduce the channel dimension to d_model (256 by default)
        feature_map, mask = features[-1]
        batch_size, num_channels, height, width = feature_map.shape
        projected_feature_map = self.conditional_detr.model.input_projection(feature_map)

        # Third, flatten the feature map + object_queries of shape NxCxHxW to NxCxHW, and permute it to NxHWxC
        # In other words, turn their shape into (batch_size, sequence_length, hidden_size)
        flattened_features = projected_feature_map.flatten(2).permute(0, 2, 1)
        object_queries = object_queries_list[-1].flatten(2).permute(0, 2, 1)

        flattened_mask = mask.flatten(1)

        # Fourth, sent flattened_features + flattened_mask + object_queries through encoder
        # flattened_features is a Tensor of shape (batch_size, heigth*width, hidden_size)
        # flattened_mask is a Tensor of shape (batch_size, heigth*width)
        if encoder_outputs is None:
            encoder_outputs = self.conditional_detr.model.encoder(
                inputs_embeds=flattened_features,
                attention_mask=flattened_mask,
                object_queries=object_queries,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # Fifth, sent query embeddings + object_queries through the decoder (which is conditioned on the encoder output)
        query_position_embeddings = self.conditional_detr.model.query_position_embeddings.weight.unsqueeze(0).repeat(
            batch_size, 1, 1
        )
        queries = mint.zeros_like(query_position_embeddings)

        # decoder outputs consists of (dec_features, dec_hidden, dec_attn)
        decoder_outputs = self.conditional_detr.model.decoder(
            inputs_embeds=queries,
            attention_mask=None,
            object_queries=object_queries,
            query_position_embeddings=query_position_embeddings,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=flattened_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Sixth, compute logits, pred_boxes and pred_masks
        logits = self.conditional_detr.class_labels_classifier(sequence_output)
        pred_boxes = self.conditional_detr.bbox_predictor(sequence_output).sigmoid()

        memory = encoder_outputs[0].permute(0, 2, 1).view(batch_size, self.config.d_model, height, width)
        mask = flattened_mask.view(batch_size, height, width)

        # FIXME h_boxes takes the last one computed, keep this in mind
        # important: we need to reverse the mask, since in the original implementation the mask works reversed
        # bbox_mask is of shape (batch_size, num_queries, number_of_attention_heads in bbox_attention, height/32, width/32)
        bbox_mask = self.bbox_attention(sequence_output, memory, mask=~mask)

        seg_masks = self.mask_head(projected_feature_map, bbox_mask, [features[2][0], features[1][0], features[0][0]])

        pred_masks = seg_masks.view(
            batch_size, self.conditional_detr.config.num_queries, seg_masks.shape[-2], seg_masks.shape[-1]
        )

        loss, loss_dict, auxiliary_outputs = None, None, None
        if labels is not None:
            outputs_class, outputs_coord = None, None
            if self.config.auxiliary_loss:
                intermediate = decoder_outputs.intermediate_hidden_states if return_dict else decoder_outputs[-1]
                outputs_class = self.conditional_detr.class_labels_classifier(intermediate)
                outputs_coord = self.conditional_detr.bbox_predictor(intermediate).sigmoid()
            loss, loss_dict, auxiliary_outputs = self.loss_function(
                logits, labels, pred_boxes, pred_masks, self.config, outputs_class, outputs_coord
            )

        if not return_dict:
            if auxiliary_outputs is not None:
                output = (logits, pred_boxes, pred_masks) + auxiliary_outputs + decoder_outputs + encoder_outputs
            else:
                output = (logits, pred_boxes, pred_masks) + decoder_outputs + encoder_outputs
            return ((loss, loss_dict) + output) if loss is not None else output

        return ConditionalDetrSegmentationOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_boxes=pred_boxes,
            pred_masks=pred_masks,
            auxiliary_outputs=auxiliary_outputs,
            last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


def _expand(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)


# Copied from transformers.models.detr.modeling_detr.DetrMaskHeadSmallConv with Detr->ConditionalDetr
class ConditionalDetrMaskHeadSmallConv(nn.Cell):
    """
    Simple convolutional head, using group norm. Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()

        if dim % 8 != 0:
            raise ValueError(
                "The hidden_size + number of attention heads must be divisible by 8 as the number of groups in"
                " GroupNorm is set to 8"
            )

        inter_dims = [dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16, context_dim // 64]

        self.lay1 = mint.nn.Conv2d(dim, dim, 3, padding=1)
        self.gn1 = mint.nn.GroupNorm(8, dim)
        self.lay2 = mint.nn.Conv2d(dim, inter_dims[1], 3, padding=1)
        self.gn2 = mint.nn.GroupNorm(min(8, inter_dims[1]), inter_dims[1])
        self.lay3 = mint.nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.gn3 = mint.nn.GroupNorm(min(8, inter_dims[2]), inter_dims[2])
        self.lay4 = mint.nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        self.gn4 = mint.nn.GroupNorm(min(8, inter_dims[3]), inter_dims[3])
        self.lay5 = mint.nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)
        self.gn5 = mint.nn.GroupNorm(min(8, inter_dims[4]), inter_dims[4])
        self.out_lay = mint.nn.Conv2d(inter_dims[4], 1, 3, padding=1)

        self.dim = dim

        self.adapter1 = mint.nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
        self.adapter2 = mint.nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
        self.adapter3 = mint.nn.Conv2d(fpn_dims[2], inter_dims[3], 1)

        for m in self.modules():
            if isinstance(m, mint.nn.Conv2d):
                m.weight.set_data(initializer(HeUniform(1), m.weight.shape, m.weight.dtype))
                m.bias.set_data(initializer(Constant(0), m.bias.shape, m.bias.dtype))

    def construct(self, x: Tensor, bbox_mask: Tensor, fpns: List[Tensor]):
        # here we concatenate x, the projected feature map, of shape (batch_size, d_model, heigth/32, width/32) with
        # the bbox_mask = the attention maps of shape (batch_size, n_queries, n_heads, height/32, width/32).
        # We expand the projected feature map to match the number of heads.
        x = mint.cat([_expand(x, bbox_mask.shape[1]), bbox_mask.flatten(0, 1)], 1)

        x = self.lay1(x)
        x = self.gn1(x)
        x = mint.nn.functional.relu(x)
        x = self.lay2(x)
        x = self.gn2(x)
        x = mint.nn.functional.relu(x)

        cur_fpn = self.adapter1(fpns[0])
        if cur_fpn.shape[0] != x.shape[0]:
            cur_fpn = _expand(cur_fpn, x.shape[0] // cur_fpn.shape[0])
        x = cur_fpn + mint.nn.functional.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay3(x)
        x = self.gn3(x)
        x = mint.nn.functional.relu(x)

        cur_fpn = self.adapter2(fpns[1])
        if cur_fpn.shape[0] != x.shape[0]:
            cur_fpn = _expand(cur_fpn, x.shape[0] // cur_fpn.shape[0])
        x = cur_fpn + mint.nn.functional.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay4(x)
        x = self.gn4(x)
        x = mint.nn.functional.relu(x)

        cur_fpn = self.adapter3(fpns[2])
        if cur_fpn.shape[0] != x.shape[0]:
            cur_fpn = _expand(cur_fpn, x.shape[0] // cur_fpn.shape[0])
        x = cur_fpn + mint.nn.functional.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay5(x)
        x = self.gn5(x)
        x = mint.nn.functional.relu(x)

        x = self.out_lay(x)
        return x


# Copied from transformers.models.detr.modeling_detr.DetrMHAttentionMap with Detr->ConditionalDetr
class ConditionalDetrMHAttentionMap(nn.Cell):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True, std=None):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = mint.nn.Dropout(dropout)

        self.q_linear = mint.nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = mint.nn.Linear(query_dim, hidden_dim, bias=bias)

        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def construct(self, q, k, mask: Optional[Tensor] = None):
        q = self.q_linear(q)
        k = mint.nn.functional.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        queries_per_head = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        keys_per_head = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        weights = mint.einsum("bqnc,bnchw->bqnhw", queries_per_head * self.normalize_fact, keys_per_head)

        if mask is not None:
            weights = weights.masked_fill(mask.unsqueeze(1).unsqueeze(1), dtype_to_max(weights.dtype))
        weights = mint.nn.functional.softmax(weights.flatten(2), dim=-1).view(weights.shape)
        weights = self.dropout(weights)
        return weights


__all__ = [
    "ConditionalDetrForObjectDetection",
    "ConditionalDetrForSegmentation",
    "ConditionalDetrModel",
    "ConditionalDetrPreTrainedModel",
]
