# coding=utf-8
# Copyright 2024 Baidu Inc and The HuggingFace Inc. team.
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
"""PyTorch RT-DETR model."""


from mindspore import mint
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import mindspore as ms
import mindspore.mint.nn.functional as F
from mindspore import Tensor, mint

from ...activations import ACT2CLS, ACT2FN
from ...image_transforms import center_to_corners_format, corners_to_center_format
from ...integrations import use_kernel_forward_from_hub
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import compile_compatible_method_lru_cache

from transformers.utils.generic import ModelOutput

from ...utils import logging
from ...utils.backbone_utils import load_backbone
from transformers import RTDetrConfig
from ...mindspore_adapter import dtype_to_max

from mindspore.common.initializer import (
    initializer,
    XavierUinform,
    Zero,
    Normal,
    Constant,
)


logger = logging.get_logger(__name__)


_CONFIG_FOR_DOC = "RTDetrConfig"
# TODO: Replace all occurrences of the checkpoint with the final one
_CHECKPOINT_FOR_DOC = "PekingU/rtdetr_r50vd"


# Copied from transformers.models.deformable_detr.modeling_deformable_detr.MultiScaleDeformableAttention
class MultiScaleDeformableAttention(ms.nn.Cell):
    def construct(
        self,
        value: Tensor,
        value_spatial_shapes: Tensor,
        value_spatial_shapes_list: List[Tuple],
        level_start_index: Tensor,
        sampling_locations: Tensor,
        attention_weights: Tensor,
        im2col_step: int,
    ):
        batch_size, _, num_heads, hidden_dim = value.shape
        _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
        value_list = value.split([height * width for height, width in value_spatial_shapes_list], dim=1)
        sampling_grids = 2 * sampling_locations - 1
        sampling_value_list = []
        for level_id, (height, width) in enumerate(value_spatial_shapes_list):
            # batch_size, height*width, num_heads, hidden_dim
            # -> batch_size, height*width, num_heads*hidden_dim
            # -> batch_size, num_heads*hidden_dim, height*width
            # -> batch_size*num_heads, hidden_dim, height, width
            value_l_ = (
                value_list[level_id]
                .flatten(2)
                .transpose(1, 2)
                .reshape(batch_size * num_heads, hidden_dim, height, width)
            )
            # batch_size, num_queries, num_heads, num_points, 2
            # -> batch_size, num_heads, num_queries, num_points, 2
            # -> batch_size*num_heads, num_queries, num_points, 2
            sampling_grid_l_ = sampling_grids[:, :, :, level_id].transpose(1, 2).flatten(0, 1)
            # batch_size*num_heads, hidden_dim, num_queries, num_points
            sampling_value_l_ = mint.nn.functional.grid_sample(
                value_l_,
                sampling_grid_l_,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            sampling_value_list.append(sampling_value_l_)
        # (batch_size, num_queries, num_heads, num_levels, num_points)
        # -> (batch_size, num_heads, num_queries, num_levels, num_points)
        # -> (batch_size, num_heads, 1, num_queries, num_levels*num_points)
        attention_weights = attention_weights.transpose(1, 2).reshape(
            batch_size * num_heads, 1, num_queries, num_levels * num_points
        )
        output = (
            (mint.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
            .sum(-1)
            .view(batch_size, num_heads * hidden_dim, num_queries)
        )
        return output.transpose(1, 2).contiguous()


@dataclass
class RTDetrDecoderOutput(ModelOutput):
    """
    Base class for outputs of the RTDetrDecoder. This class adds two attributes to
    BaseModelOutputWithCrossAttentions, namely:
    - a stacked tensor of intermediate decoder hidden states (i.e. the output of each decoder layer)
    - a stacked tensor of intermediate reference points.

    Args:
        last_hidden_state (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        intermediate_hidden_states (`mindspore.Tensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`):
            Stacked intermediate hidden states (output of each layer of the decoder).
        intermediate_logits (`mindspore.Tensor` of shape `(batch_size, config.decoder_layers, sequence_length, config.num_labels)`):
            Stacked intermediate logits (logits of each layer of the decoder).
        intermediate_reference_points (`mindspore.Tensor` of shape `(batch_size, config.decoder_layers, sequence_length, hidden_size)`):
            Stacked intermediate reference points (reference points of each layer of the decoder).
        hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        cross_attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax,
            used to compute the weighted average in the cross-attention heads.
    """

    last_hidden_state: ms.Tensor = None
    intermediate_hidden_states: ms.Tensor = None
    intermediate_logits: ms.Tensor = None
    intermediate_reference_points: ms.Tensor = None
    hidden_states: Optional[Tuple[ms.Tensor]] = None
    attentions: Optional[Tuple[ms.Tensor]] = None
    cross_attentions: Optional[Tuple[ms.Tensor]] = None


@dataclass
class RTDetrModelOutput(ModelOutput):
    """
    Base class for outputs of the RT-DETR encoder-decoder model.

    Args:
        last_hidden_state (`mindspore.Tensor` of shape `(batch_size, num_queries, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.
        intermediate_hidden_states (`mindspore.Tensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`):
            Stacked intermediate hidden states (output of each layer of the decoder).
        intermediate_logits (`mindspore.Tensor` of shape `(batch_size, config.decoder_layers, sequence_length, config.num_labels)`):
            Stacked intermediate logits (logits of each layer of the decoder).
        intermediate_reference_points (`mindspore.Tensor` of shape `(batch_size, config.decoder_layers, num_queries, 4)`):
            Stacked intermediate reference points (reference points of each layer of the decoder).
        decoder_hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, num_queries, hidden_size)`. Hidden-states of the decoder at the output of each layer
            plus the initial embedding outputs.
        decoder_attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, num_queries,
            num_queries)`. Attentions weights of the decoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        cross_attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_queries, num_heads, 4, 4)`.
            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the encoder at the output of each
            layer plus the initial embedding outputs.
        encoder_attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_queries, num_heads, 4, 4)`.
            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        init_reference_points (`mindspore.Tensor` of shape  `(batch_size, num_queries, 4)`):
            Initial reference points sent through the Transformer decoder.
        enc_topk_logits (`mindspore.Tensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            Predicted bounding boxes scores where the top `config.two_stage_num_proposals` scoring bounding boxes are
            picked as region proposals in the encoder stage. Output of bounding box binary classification (i.e.
            foreground and background).
        enc_topk_bboxes (`mindspore.Tensor` of shape `(batch_size, sequence_length, 4)`):
            Logits of predicted bounding boxes coordinates in the encoder stage.
        enc_outputs_class (`mindspore.Tensor` of shape `(batch_size, sequence_length, config.num_labels)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`):
            Predicted bounding boxes scores where the top `config.two_stage_num_proposals` scoring bounding boxes are
            picked as region proposals in the first stage. Output of bounding box binary classification (i.e.
            foreground and background).
        enc_outputs_coord_logits (`mindspore.Tensor` of shape `(batch_size, sequence_length, 4)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`):
            Logits of predicted bounding boxes coordinates in the first stage.
        denoising_meta_values (`dict`):
            Extra dictionary for the denoising related values
    """

    last_hidden_state: ms.Tensor = None
    intermediate_hidden_states: ms.Tensor = None
    intermediate_logits: ms.Tensor = None
    intermediate_reference_points: ms.Tensor = None
    decoder_hidden_states: Optional[Tuple[ms.Tensor]] = None
    decoder_attentions: Optional[Tuple[ms.Tensor]] = None
    cross_attentions: Optional[Tuple[ms.Tensor]] = None
    encoder_last_hidden_state: Optional[ms.Tensor] = None
    encoder_hidden_states: Optional[Tuple[ms.Tensor]] = None
    encoder_attentions: Optional[Tuple[ms.Tensor]] = None
    init_reference_points: ms.Tensor = None
    enc_topk_logits: Optional[ms.Tensor] = None
    enc_topk_bboxes: Optional[ms.Tensor] = None
    enc_outputs_class: Optional[ms.Tensor] = None
    enc_outputs_coord_logits: Optional[ms.Tensor] = None
    denoising_meta_values: Optional[Dict] = None


@dataclass
class RTDetrObjectDetectionOutput(ModelOutput):
    """
    Output type of [`RTDetrForObjectDetection`].

    Args:
        loss (`mindspore.Tensor` of shape `(1,)`, *optional*, returned when `labels` are provided)):
            Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
            bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
            scale-invariant IoU loss.
        loss_dict (`Dict`, *optional*):
            A dictionary containing the individual losses. Useful for logging.
        logits (`mindspore.Tensor` of shape `(batch_size, num_queries, num_classes + 1)`):
            Classification logits (including no-object) for all queries.
        pred_boxes (`mindspore.Tensor` of shape `(batch_size, num_queries, 4)`):
            Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
            values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
            possible padding). You can use [`~RTDetrImageProcessor.post_process_object_detection`] to retrieve the
            unnormalized (absolute) bounding boxes.
        auxiliary_outputs (`list[Dict]`, *optional*):
            Optional, only returned when auxiliary losses are activated (i.e. `config.auxiliary_loss` is set to `True`)
            and labels are provided. It is a list of dictionaries containing the two above keys (`logits` and
            `pred_boxes`) for each decoder layer.
        last_hidden_state (`mindspore.Tensor` of shape `(batch_size, num_queries, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.
        intermediate_hidden_states (`mindspore.Tensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`):
            Stacked intermediate hidden states (output of each layer of the decoder).
        intermediate_logits (`mindspore.Tensor` of shape `(batch_size, config.decoder_layers, num_queries, config.num_labels)`):
            Stacked intermediate logits (logits of each layer of the decoder).
        intermediate_reference_points (`mindspore.Tensor` of shape `(batch_size, config.decoder_layers, num_queries, 4)`):
            Stacked intermediate reference points (reference points of each layer of the decoder).
        decoder_hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, num_queries, hidden_size)`. Hidden-states of the decoder at the output of each layer
            plus the initial embedding outputs.
        decoder_attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, num_queries,
            num_queries)`. Attentions weights of the decoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        cross_attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_queries, num_heads, 4, 4)`.
            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the encoder at the output of each
            layer plus the initial embedding outputs.
        encoder_attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_queries, num_heads, 4, 4)`.
            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        init_reference_points (`mindspore.Tensor` of shape  `(batch_size, num_queries, 4)`):
            Initial reference points sent through the Transformer decoder.
        enc_topk_logits (`mindspore.Tensor` of shape `(batch_size, sequence_length, config.num_labels)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`):
            Logits of predicted bounding boxes coordinates in the encoder.
        enc_topk_bboxes (`mindspore.Tensor` of shape `(batch_size, sequence_length, 4)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`):
            Logits of predicted bounding boxes coordinates in the encoder.
        enc_outputs_class (`mindspore.Tensor` of shape `(batch_size, sequence_length, config.num_labels)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`):
            Predicted bounding boxes scores where the top `config.two_stage_num_proposals` scoring bounding boxes are
            picked as region proposals in the first stage. Output of bounding box binary classification (i.e.
            foreground and background).
        enc_outputs_coord_logits (`mindspore.Tensor` of shape `(batch_size, sequence_length, 4)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`):
            Logits of predicted bounding boxes coordinates in the first stage.
        denoising_meta_values (`dict`):
            Extra dictionary for the denoising related values
    """

    loss: Optional[ms.Tensor] = None
    loss_dict: Optional[Dict] = None
    logits: ms.Tensor = None
    pred_boxes: ms.Tensor = None
    auxiliary_outputs: Optional[List[Dict]] = None
    last_hidden_state: ms.Tensor = None
    intermediate_hidden_states: ms.Tensor = None
    intermediate_logits: ms.Tensor = None
    intermediate_reference_points: ms.Tensor = None
    decoder_hidden_states: Optional[Tuple[ms.Tensor]] = None
    decoder_attentions: Optional[Tuple[ms.Tensor]] = None
    cross_attentions: Optional[Tuple[ms.Tensor]] = None
    encoder_last_hidden_state: Optional[ms.Tensor] = None
    encoder_hidden_states: Optional[Tuple[ms.Tensor]] = None
    encoder_attentions: Optional[Tuple[ms.Tensor]] = None
    init_reference_points: Optional[Tuple[ms.Tensor]] = None
    enc_topk_logits: Optional[ms.Tensor] = None
    enc_topk_bboxes: Optional[ms.Tensor] = None
    enc_outputs_class: Optional[ms.Tensor] = None
    enc_outputs_coord_logits: Optional[ms.Tensor] = None
    denoising_meta_values: Optional[Dict] = None


def _get_clones(partial_module, N):
    return ms.nn.CellList([partial_module() for i in range(N)])


# Copied from transformers.models.conditional_detr.modeling_conditional_detr.inverse_sigmoid
def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return mint.log(x1 / x2)


# Copied from transformers.models.detr.modeling_detr.DetrFrozenBatchNorm2d with Detr->RTDetr
class RTDetrFrozenBatchNorm2d(ms.nn.Cell):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt, without which any other models than
    torchvision.models.resnet[18,34,50,101] produce nans.
    """

    def __init__(self, n):
        super().__init__()
        self.register_buffer("weight", mint.ones(n))
        self.register_buffer("bias", mint.zeros(n))
        self.register_buffer("running_mean", mint.zeros(n))
        self.register_buffer("running_var", mint.ones(n))

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


# Copied from transformers.models.detr.modeling_detr.replace_batch_norm with Detr->RTDetr
def replace_batch_norm(model):
    r"""
    Recursively replace all `mindspore.nn.BatchNorm2d` with `RTDetrFrozenBatchNorm2d`.

    Args:
        model (mindspore.nn.Cell):
            input model
    """
    for name, child_cell in model.name_cells().items():
        if isinstance(child_cell, mint.nn.BatchNorm2d):
            new_module = RTDetrFrozenBatchNorm2d(child_cell.num_features)
            setattr(model, name, new_module)
        else:
            replace_batch_norm(child_cell)


def get_contrastive_denoising_training_group(
    targets,
    num_classes,
    num_queries,
    class_embed,
    num_denoising_queries=100,
    label_noise_ratio=0.5,
    box_noise_scale=1.0,
):
    """
    Creates a contrastive denoising training group using ground-truth samples. It adds noise to labels and boxes.

    Args:
        targets (`List[dict]`):
            The target objects, each containing 'class_labels' and 'boxes' for objects in an image.
        num_classes (`int`):
            Total number of classes in the dataset.
        num_queries (`int`):
            Number of query slots in the transformer.
        class_embed (`callable`):
            A function or a model layer to embed class labels.
        num_denoising_queries (`int`, *optional*, defaults to 100):
            Number of denoising queries.
        label_noise_ratio (`float`, *optional*, defaults to 0.5):
            Ratio of noise applied to labels.
        box_noise_scale (`float`, *optional*, defaults to 1.0):
            Scale of noise applied to bounding boxes.
    Returns:
        `tuple` comprising various elements:
        - **input_query_class** (`mindspore.Tensor`) --
          Class queries with applied label noise.
        - **input_query_bbox** (`mindspore.Tensor`) --
          Bounding box queries with applied box noise.
        - **attn_mask** (`mindspore.Tensor`) --
           Attention mask for separating denoising and reconstruction queries.
        - **denoising_meta_values** (`dict`) --
          Metadata including denoising positive indices, number of groups, and split sizes.
    """

    if num_denoising_queries <= 0:
        return None, None, None, None

    num_ground_truths = [len(t["class_labels"]) for t in targets]

    max_gt_num = max(num_ground_truths)
    if max_gt_num == 0:
        return None, None, None, None

    num_groups_denoising_queries = num_denoising_queries // max_gt_num
    num_groups_denoising_queries = 1 if num_groups_denoising_queries == 0 else num_groups_denoising_queries
    # pad gt to max_num of a batch
    batch_size = len(num_ground_truths)

    input_query_class = ms.mint.full([batch_size, max_gt_num], num_classes, dtype=ms.int32, )
    input_query_bbox = mint.zeros([batch_size, max_gt_num, 4], )
    pad_gt_mask = mint.zeros([batch_size, max_gt_num], dtype=ms.bool_, )

    for i in range(batch_size):
        num_gt = num_ground_truths[i]
        if num_gt > 0:
            input_query_class[i, :num_gt] = targets[i]["class_labels"]
            input_query_bbox[i, :num_gt] = targets[i]["boxes"]
            pad_gt_mask[i, :num_gt] = 1
    # each group has positive and negative queries.
    input_query_class = input_query_class.tile([1, 2 * num_groups_denoising_queries])
    input_query_bbox = input_query_bbox.tile([1, 2 * num_groups_denoising_queries, 1])
    pad_gt_mask = pad_gt_mask.tile([1, 2 * num_groups_denoising_queries])
    # positive and negative mask
    negative_gt_mask = mint.zeros([batch_size, max_gt_num * 2, 1], )
    negative_gt_mask[:, max_gt_num:] = 1
    negative_gt_mask = negative_gt_mask.tile([1, num_groups_denoising_queries, 1])
    positive_gt_mask = 1 - negative_gt_mask
    # contrastive denoising training positive index
    positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask
    denoise_positive_idx = mint.nonzero(positive_gt_mask)[:, 1]
    denoise_positive_idx = mint.split(
        denoise_positive_idx, [n * num_groups_denoising_queries for n in num_ground_truths]
    )
    # total denoising queries
    num_denoising_queries = (max_gt_num * 2 * num_groups_denoising_queries).to(mindspore.int64)

    if label_noise_ratio > 0:
        mask = mint.rand_like(input_query_class, dtype=ms.float32) < (label_noise_ratio * 0.5)
        # randomly put a new one here
        new_label = mint.randint_like(mask, 0, num_classes, dtype=input_query_class.dtype)
        input_query_class = mint.where(mask & pad_gt_mask, new_label, input_query_class)

    if box_noise_scale > 0:
        known_bbox = center_to_corners_format(input_query_bbox)
        diff = mint.tile(input_query_bbox[..., 2:] * 0.5, [1, 1, 2]) * box_noise_scale
        rand_sign = mint.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0
        rand_part = mint.rand_like(input_query_bbox)
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask)
        rand_part *= rand_sign
        known_bbox += rand_part * diff
        known_bbox.clip_(min=0.0, max=1.0)
        input_query_bbox = corners_to_center_format(known_bbox)
        input_query_bbox = inverse_sigmoid(input_query_bbox)

    input_query_class = class_embed(input_query_class)

    target_size = num_denoising_queries + num_queries
    attn_mask = ms.mint.full([target_size, target_size], False, dtype=ms.bool_, )
    # match query cannot see the reconstruction
    attn_mask[num_denoising_queries:, :num_denoising_queries] = True

    # reconstructions cannot see each other
    for i in range(num_groups_denoising_queries):
        idx_block_start = max_gt_num * 2 * i
        idx_block_end = max_gt_num * 2 * (i + 1)
        attn_mask[idx_block_start:idx_block_end, :idx_block_start] = True
        attn_mask[idx_block_start:idx_block_end, idx_block_end:num_denoising_queries] = True

    denoising_meta_values = {
        "dn_positive_idx": denoise_positive_idx,
        "dn_num_group": num_groups_denoising_queries,
        "dn_num_split": [num_denoising_queries, num_queries],
    }

    return input_query_class, input_query_bbox, attn_mask, denoising_meta_values


class RTDetrConvEncoder(ms.nn.Cell):
    """
    Convolutional backbone using the modeling_rt_detr_resnet.py.

    nn.BatchNorm2d layers are replaced by RTDetrFrozenBatchNorm2d as defined above.
    https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetr_pytorch/src/nn/backbone/presnet.py#L142
    """

    def __init__(self, config):
        super().__init__()

        backbone = load_backbone(config)

        if config.freeze_backbone_batch_norms:
            # replace batch norm by frozen batch norm
            replace_batch_norm(backbone)
        self.model = backbone
        self.intermediate_channel_sizes = self.model.channels

    def construct(self, pixel_values: ms.Tensor, pixel_mask: ms.Tensor):
        # send pixel_values through the model to get list of feature maps
        features = self.model(pixel_values).feature_maps

        out = []
        for feature_map in features:
            # downsample pixel_mask to match shape of corresponding feature_map
            mask = mint.nn.functional.interpolate(pixel_mask[None].float(), size=feature_map.shape[-2:]).to(ms.bool_)[0]
            out.append((feature_map, mask))
        return out


class RTDetrConvNormLayer(ms.nn.Cell):
    def __init__(self, config, in_channels, out_channels, kernel_size, stride, padding=None, activation=None):
        super().__init__()
        self.conv = mint.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=False,
        )
        self.norm = mint.nn.BatchNorm2d(out_channels, config.batch_norm_eps)
        self.activation = mint.nn.Identity() if activation is None else ACT2CLS[activation]()

    def construct(self, hidden_state):
        hidden_state = self.conv(hidden_state)
        hidden_state = self.norm(hidden_state)
        hidden_state = self.activation(hidden_state)
        return hidden_state


class RTDetrEncoderLayer(ms.nn.Cell):
    def __init__(self, config: RTDetrConfig):
        super().__init__()
        self.normalize_before = config.normalize_before

        # self-attention
        self.self_attn = RTDetrMultiheadAttention(
            embed_dim=config.encoder_hidden_dim,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
        )
        self.self_attn_layer_norm = mint.nn.LayerNorm(config.encoder_hidden_dim, eps=config.layer_norm_eps)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.encoder_activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = mint.nn.Linear(config.encoder_hidden_dim, config.encoder_ffn_dim)
        self.fc2 = mint.nn.Linear(config.encoder_ffn_dim, config.encoder_hidden_dim)
        self.final_layer_norm = mint.nn.LayerNorm(config.encoder_hidden_dim, eps=config.layer_norm_eps)

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: ms.Tensor,
        position_embeddings: ms.Tensor = None,
        output_attentions: bool = False,
        **kwargs,
    ):
        """
        Args:
            hidden_states (`mindspore.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`mindspore.Tensor`): attention mask of size
                `(batch, 1, target_len, source_len)` where padding elements are indicated by very large negative
                values.
            position_embeddings (`mindspore.Tensor`, *optional*):
                Object queries (also called content embeddings), to be added to the hidden states.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
        )

        hidden_states = mint.nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        if not self.normalize_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        if self.normalize_before:
            hidden_states = self.final_layer_norm(hidden_states)
        residual = hidden_states

        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = mint.nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)

        hidden_states = self.fc2(hidden_states)

        hidden_states = mint.nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = residual + hidden_states
        if not self.normalize_before:
            hidden_states = self.final_layer_norm(hidden_states)

        if self.training:
            if mint.isinf(hidden_states).any() or mint.isnan(hidden_states).any():
                clamp_value = ms.tensor(dtype_to_max(hidden_states.dtype) - 1000, dtype=hidden_states.dtype)
                hidden_states = mint.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class RTDetrRepVggBlock(ms.nn.Cell):
    """
    RepVGG architecture block introduced by the work "RepVGG: Making VGG-style ConvNets Great Again".
    """

    def __init__(self, config: RTDetrConfig):
        super().__init__()

        activation = config.activation_function
        hidden_channels = int(config.encoder_hidden_dim * config.hidden_expansion)
        self.conv1 = RTDetrConvNormLayer(config, hidden_channels, hidden_channels, 3, 1, padding=1)
        self.conv2 = RTDetrConvNormLayer(config, hidden_channels, hidden_channels, 1, 1, padding=0)
        self.activation = mint.nn.Identity() if activation is None else ACT2CLS[activation]()

    def construct(self, x):
        y = self.conv1(x) + self.conv2(x)
        return self.activation(y)


class RTDetrCSPRepLayer(ms.nn.Cell):
    """
    Cross Stage Partial (CSP) network layer with RepVGG blocks.
    """

    def __init__(self, config: RTDetrConfig):
        super().__init__()

        in_channels = config.encoder_hidden_dim * 2
        out_channels = config.encoder_hidden_dim
        num_blocks = 3
        activation = config.activation_function

        hidden_channels = int(out_channels * config.hidden_expansion)
        self.conv1 = RTDetrConvNormLayer(config, in_channels, hidden_channels, 1, 1, activation=activation)
        self.conv2 = RTDetrConvNormLayer(config, in_channels, hidden_channels, 1, 1, activation=activation)
        self.bottlenecks = ms.nn.SequentialCell(*[RTDetrRepVggBlock(config) for _ in range(num_blocks)])
        if hidden_channels != out_channels:
            self.conv3 = RTDetrConvNormLayer(config, hidden_channels, out_channels, 1, 1, activation=activation)
        else:
            self.conv3 = mint.nn.Identity()

    def construct(self, hidden_state):
        hidden_state_1 = self.conv1(hidden_state)
        hidden_state_1 = self.bottlenecks(hidden_state_1)
        hidden_state_2 = self.conv2(hidden_state)
        return self.conv3(hidden_state_1 + hidden_state_2)


# Copied from transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrMultiscaleDeformableAttention with DeformableDetr->RTDetr
class RTDetrMultiscaleDeformableAttention(ms.nn.Cell):
    """
    Multiscale deformable attention as proposed in Deformable DETR.
    """

    def __init__(self, config: RTDetrConfig, num_heads: int, n_points: int):
        super().__init__()

        self.attn = MultiScaleDeformableAttention()

        if config.d_model % num_heads != 0:
            raise ValueError(
                f"embed_dim (d_model) must be divisible by num_heads, but got {config.d_model} and {num_heads}"
            )
        dim_per_head = config.d_model // num_heads
        # check if dim_per_head is power of 2
        if not ((dim_per_head & (dim_per_head - 1) == 0) and dim_per_head != 0):
            warnings.warn(
                "You'd better set embed_dim (d_model) in RTDetrMultiscaleDeformableAttention to make the"
                " dimension of each attention head a power of 2 which is more efficient in the authors' CUDA"
                " implementation."
            )

        self.im2col_step = 64

        self.d_model = config.d_model
        self.n_levels = config.num_feature_levels
        self.n_heads = num_heads
        self.n_points = n_points

        self.sampling_offsets = mint.nn.Linear(config.d_model, num_heads * self.n_levels * n_points * 2)
        self.attention_weights = mint.nn.Linear(config.d_model, num_heads * self.n_levels * n_points)
        self.value_proj = mint.nn.Linear(config.d_model, config.d_model)
        self.output_proj = mint.nn.Linear(config.d_model, config.d_model)

        self.disable_custom_kernels = config.disable_custom_kernels

    def with_pos_embed(self, tensor: ms.Tensor, position_embeddings: Optional[Tensor]):
        return tensor if position_embeddings is None else tensor + position_embeddings

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        position_embeddings: Optional[ms.Tensor] = None,
        reference_points=None,
        spatial_shapes=None,
        spatial_shapes_list=None,
        level_start_index=None,
        output_attentions: bool = False,
    ):
        # add position embeddings to the hidden states before projecting to queries and keys
        if position_embeddings is not None:
            hidden_states = self.with_pos_embed(hidden_states, position_embeddings)

        batch_size, num_queries, _ = hidden_states.shape
        batch_size, sequence_length, _ = encoder_hidden_states.shape
        total_elements = sum(height * width for height, width in spatial_shapes_list)
        if total_elements != sequence_length:
            raise ValueError(
                "Make sure to align the spatial shapes with the sequence length of the encoder hidden states"
            )

        value = self.value_proj(encoder_hidden_states)
        if attention_mask is not None:
            # we invert the attention_mask
            value = value.masked_fill(~attention_mask[..., None], float(0))
        value = value.view(batch_size, sequence_length, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = mint.nn.functional.softmax(attention_weights, -1).view(
            batch_size, num_queries, self.n_heads, self.n_levels, self.n_points
        )
        # batch_size, num_queries, n_heads, n_levels, n_points, 2
        num_coordinates = reference_points.shape[-1]
        if num_coordinates == 2:
            offset_normalizer = mint.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif num_coordinates == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}")

        output = self.attn(
            value,
            spatial_shapes,
            spatial_shapes_list,
            level_start_index,
            sampling_locations,
            attention_weights,
            self.im2col_step,
        )

        output = self.output_proj(output)

        return output, attention_weights


class RTDetrMultiheadAttention(ms.nn.Cell):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.

    Here, we add position embeddings to the queries and keys (as explained in the Deformable DETR paper).
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

    def _reshape(self, tensor: ms.Tensor, seq_len: int, batch_size: int):
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def with_pos_embed(self, tensor: ms.Tensor, position_embeddings: Optional[Tensor]):
        return tensor if position_embeddings is None else tensor + position_embeddings

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        position_embeddings: Optional[ms.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[ms.Tensor, Optional[ms.Tensor], Optional[Tuple[ms.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, target_len, embed_dim = hidden_states.shape
        # add position embeddings to the hidden states before projecting to queries and keys
        if position_embeddings is not None:
            hidden_states_original = hidden_states
            hidden_states = self.with_pos_embed(hidden_states, position_embeddings)

        # get queries, keys and values
        query_states = self.q_proj(hidden_states) * self.scaling
        key_states = self._reshape(self.k_proj(hidden_states), -1, batch_size)
        value_states = self._reshape(self.v_proj(hidden_states_original), -1, batch_size)

        proj_shape = (batch_size * self.num_heads, -1, self.head_dim)
        query_states = self._reshape(query_states, target_len, batch_size).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        source_len = key_states.shape[1]

        attn_weights = mint.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.shape != (batch_size * self.num_heads, target_len, source_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size * self.num_heads, target_len, source_len)}, but is"
                f" {attn_weights.shape}"
            )

        # expand attention_mask
        if attention_mask is not None:
            # [seq_len, seq_len] -> [batch_size, 1, target_seq_len, source_seq_len]
            attention_mask = attention_mask.broadcast_to(batch_size, 1, *attention_mask.shape)

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


class RTDetrDecoderLayer(ms.nn.Cell):
    def __init__(self, config: RTDetrConfig):
        super().__init__()
        # self-attention
        self.self_attn = RTDetrMultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.decoder_activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = mint.nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        # cross-attention
        self.encoder_attn = RTDetrMultiscaleDeformableAttention(
            config,
            num_heads=config.decoder_attention_heads,
            n_points=config.decoder_n_points,
        )
        self.encoder_attn_layer_norm = mint.nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        # feedforward neural networks
        self.fc1 = mint.nn.Linear(config.d_model, config.decoder_ffn_dim)
        self.fc2 = mint.nn.Linear(config.decoder_ffn_dim, config.d_model)
        self.final_layer_norm = mint.nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

    def construct(
        self,
        hidden_states: ms.Tensor,
        position_embeddings: Optional[ms.Tensor] = None,
        reference_points=None,
        spatial_shapes=None,
        spatial_shapes_list=None,
        level_start_index=None,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        encoder_attention_mask: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ):
        """
        Args:
            hidden_states (`mindspore.Tensor`):
                Input to the layer of shape `(seq_len, batch, embed_dim)`.
            position_embeddings (`mindspore.Tensor`, *optional*):
                Position embeddings that are added to the queries and keys in the self-attention layer.
            reference_points (`mindspore.Tensor`, *optional*):
                Reference points.
            spatial_shapes (`mindspore.Tensor`, *optional*):
                Spatial shapes.
            level_start_index (`mindspore.Tensor`, *optional*):
                Level start index.
            encoder_hidden_states (`mindspore.Tensor`):
                cross attention input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_attention_mask (`mindspore.Tensor`): encoder attention mask of size
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
            attention_mask=encoder_attention_mask,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
        )

        hidden_states = mint.nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        second_residual = hidden_states

        # Cross-Attention
        cross_attn_weights = None
        hidden_states, cross_attn_weights = self.encoder_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            output_attentions=output_attentions,
        )

        hidden_states = mint.nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = second_residual + hidden_states

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


class RTDetrPreTrainedModel(PreTrainedModel):
    config_class = RTDetrConfig
    base_model_prefix = "rt_detr"
    main_input_name = "pixel_values"
    _no_split_modules = [r"RTDetrHybridEncoder", r"RTDetrDecoderLayer"]

    def _init_weights(self, cell):
        """Initalize the weights"""

        if isinstance(cell, (RTDetrForObjectDetection, RTDetrDecoder)):
            if hasattr(cell, 'class_embed') and cell.class_embed is not None:
                for layer in cell.class_embed:
                    prior_prob = self.config.initializer_bias_prior_prob or 1 / (self.config.num_labels + 1)
                    bias = -math.log((1 - prior_prob) / prior_prob)
                    layer.weight.set_data(initializer(XavierUniform(), layer.weight.shape, layer.weight.dtype))
                    layer.bias.set_data(initializer(Constant(bias), layer.bias.shape, layer.bias.dtype))

            if hasattr(cell, 'bbox_embed') and cell.bbox_embed is not None:
                for layer in cell.bbox_embed:
                    layer.layers[-1].weight.set_data(
                        initializer(Zero(), layer.layers[-1].weight.shape, layer.layers[-1].weight.dtype))
                    layer.layers[-1].bias.set_data(
                        initializer(Zero(), layer.layers[-1].bias.shape, layer.layers[-1].bias.dtype))

        if isinstance(cell, RTDetrMultiscaleDeformableAttention):
            cell.sampling_offsets.weight.set_data(
                initializer(Zero(), cell.sampling_offsets.weight.shape, cell.sampling_offsets.weight.dtype))

            thetas = mint.arange(cell.n_heads, dtype=ms.float32) * (2.0 * math.pi / cell.n_heads)
            grid_init = mint.stack([mint.cos(thetas), mint.sin(thetas)], -1)

            max_val = grid_init.abs().max(-1, keepdim=True)[0]
            grid_init = (grid_init / max_val)
            grid_init = grid_init.view(cell.n_heads, 1, 1, 2)

            grid_init = mint.tile(grid_init, (1, cell.n_levels, cell.n_points, 1))

            for i in range(cell.n_points):
                grid_init[:, :, i, :] *= (i + 1)

            cell.sampling_offsets.bias = ms.Parameter(grid_init.view(-1), name=cell.sampling_offsets.bias.name)

            cell.attention_weights.weight.set_data(
                initializer(Zero(), cell.attention_weights.weight.shape, cell.attention_weights.weight.dtype))
            cell.attention_weights.bias.set_data(
                initializer(Zero(), cell.attention_weights.bias.shape, cell.attention_weights.bias.dtype))
            cell.value_proj.weight.set_data(
                initializer(XavierUniform(), cell.value_proj.weight.shape, cell.value_proj.weight.dtype))
            cell.value_proj.bias.set_data(initializer(Zero(), cell.value_proj.bias.shape, cell.value_proj.bias.dtype))
            cell.output_proj.weight.set_data(
                initializer(XavierUniform(), cell.output_proj.weight.shape, cell.output_proj.weight.dtype))
            cell.output_proj.bias.set_data(
                initializer(Zero(), cell.output_proj.bias.shape, cell.output_proj.bias.dtype))

        if isinstance(cell, RTDetrModel):
            prior_prob = self.config.initializer_bias_prior_prob or 1 / (self.config.num_labels + 1)
            bias = -math.log((1 - prior_prob) / prior_prob)
            cell.enc_score_head.weight.set_data(
                initializer(XavierUniform(), cell.enc_score_head.weight.shape, cell.enc_score_head.weight.dtype))
            cell.enc_score_head.bias.set_data(
                initializer(Constant(bias), cell.enc_score_head.bias.shape, cell.enc_score_head.bias.dtype))

        if isinstance(cell, (mint.nn.Linear, mint.nn.Conv2d)):
            cell.weight.set_data(
                initializer(Normal(self.config.initializer_range), cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.set_data(initializer(Zero(), cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, mint.nn.BatchNorm2d):
            cell.weight.set_data(initializer(Normal(self.config.initializer_range), cell.weight.shape, cell.weight.dtype))
            cell.bias.set_data(initializer(Zero(), cell.bias.shape, cell.bias.dtype))

        if hasattr(cell, "weight_embedding") and self.config.learn_initial_query:
            cell.weight_embedding.weight.set_data(
                initializer(XavierUniform(), cell.weight_embedding.weight.shape, cell.weight_embedding.weight.dtype))
        if hasattr(cell, "denoising_class_embed") and self.config.num_denoising > 0:
            cell.denoising_class_embed.weight.set_data(
                initializer(XavierUniform(), cell.denoising_class_embed.weight.shape,
                            cell.denoising_class_embed.weight.dtype))


class RTDetrEncoder(ms.nn.Cell):
    def __init__(self, config: RTDetrConfig):
        super().__init__()

        self.layers = ms.nn.CellList([RTDetrEncoderLayer(config) for _ in range(config.encoder_layers)])

    def construct(self, src, src_mask=None, pos_embed=None, output_attentions: bool = False) -> ms.Tensor:
        hidden_states = src
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=src_mask,
                position_embeddings=pos_embed,
                output_attentions=output_attentions,
            )
        return hidden_states


class RTDetrHybridEncoder(ms.nn.Cell):
    """
    Decoder consisting of a projection layer, a set of `RTDetrEncoder`, a top-down Feature Pyramid Network
    (FPN) and a bottom-up Path Aggregation Network (PAN). More details on the paper: https://arxiv.org/abs/2304.08069

    Args:
        config: RTDetrConfig
    """

    def __init__(self, config: RTDetrConfig):
        super().__init__()
        self.config = config
        self.in_channels = config.encoder_in_channels
        self.feat_strides = config.feat_strides
        self.encoder_hidden_dim = config.encoder_hidden_dim
        self.encode_proj_layers = config.encode_proj_layers
        self.positional_encoding_temperature = config.positional_encoding_temperature
        self.eval_size = config.eval_size
        self.out_channels = [self.encoder_hidden_dim for _ in self.in_channels]
        self.out_strides = self.feat_strides
        self.num_fpn_stages = len(self.in_channels) - 1
        self.num_pan_stages = len(self.in_channels) - 1
        activation = config.activation_function

        # encoder transformer
        self.encoder = ms.nn.CellList([RTDetrEncoder(config) for _ in range(len(self.encode_proj_layers))])

        # top-down FPN
        self.lateral_convs = ms.nn.CellList()
        self.fpn_blocks = ms.nn.CellList()
        for _ in range(self.num_fpn_stages):
            lateral_conv = RTDetrConvNormLayer(
                config,
                in_channels=self.encoder_hidden_dim,
                out_channels=self.encoder_hidden_dim,
                kernel_size=1,
                stride=1,
                activation=activation,
            )
            fpn_block = RTDetrCSPRepLayer(config)
            self.lateral_convs.append(lateral_conv)
            self.fpn_blocks.append(fpn_block)

        # bottom-up PAN
        self.downsample_convs = ms.nn.CellList()
        self.pan_blocks = ms.nn.CellList()
        for _ in range(self.num_pan_stages):
            downsample_conv = RTDetrConvNormLayer(
                config,
                in_channels=self.encoder_hidden_dim,
                out_channels=self.encoder_hidden_dim,
                kernel_size=3,
                stride=2,
                activation=activation,
            )
            pan_block = RTDetrCSPRepLayer(config)
            self.downsample_convs.append(downsample_conv)
            self.pan_blocks.append(pan_block)

    @staticmethod
    def build_2d_sincos_position_embedding(
        width, height, embed_dim=256, temperature=10000.0, device="cpu", dtype=ms.float32
    ):
        grid_w = mint.arange(width, ).to(dtype)
        grid_h = mint.arange(height, ).to(dtype)
        grid_w, grid_h = mint.meshgrid(grid_w, grid_h, indexing="ij")
        if embed_dim % 4 != 0:
            raise ValueError("Embed dimension must be divisible by 4 for 2D sin-cos position embedding")
        pos_dim = embed_dim // 4
        omega = mint.arange(pos_dim, ).to(dtype) / pos_dim
        omega = 1.0 / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return mint.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def construct(
        self,
        inputs_embeds=None,
        attention_mask=None,
        position_embeddings=None,
        spatial_shapes=None,
        level_start_index=None,
        valid_ratios=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            inputs_embeds (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Flattened feature map (output of the backbone + projection layer) that is passed to the encoder.
            attention_mask (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding pixel features. Mask values selected in `[0, 1]`:
                - 1 for pixel features that are real (i.e. **not masked**),
                - 0 for pixel features that are padding (i.e. **masked**).
                [What are attention masks?](../glossary#attention-mask)
            position_embeddings (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Position embeddings that are added to the queries and keys in each self-attention layer.
            spatial_shapes (`mindspore.Tensor` of shape `(num_feature_levels, 2)`):
                Spatial shapes of each feature map.
            level_start_index (`mindspore.Tensor` of shape `(num_feature_levels)`):
                Starting index of each feature map.
            valid_ratios (`mindspore.Tensor` of shape `(batch_size, num_feature_levels, 2)`):
                Ratio of valid area in each feature level.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = inputs_embeds

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # encoder
        if self.config.encoder_layers > 0:
            for i, enc_ind in enumerate(self.encode_proj_layers):
                if output_hidden_states:
                    encoder_states = encoder_states + (hidden_states[enc_ind],)
                height, width = hidden_states[enc_ind].shape[2:]
                # flatten [batch, channel, height, width] to [batch, height*width, channel]
                src_flatten = hidden_states[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        width,
                        height,
                        self.encoder_hidden_dim,
                        self.positional_encoding_temperature,
                        dtype=src_flatten.dtype,
                    )
                else:
                    pos_embed = None

                layer_outputs = self.encoder[i](
                    src_flatten,
                    pos_embed=pos_embed,
                    output_attentions=output_attentions,
                )
                hidden_states[enc_ind] = (
                    layer_outputs[0].permute(0, 2, 1).reshape(-1, self.encoder_hidden_dim, height, width).contiguous()
                )

                if output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states[enc_ind],)

        # top-down FPN
        fpn_feature_maps = [hidden_states[-1]]
        for idx, (lateral_conv, fpn_block) in enumerate(zip(self.lateral_convs, self.fpn_blocks)):
            backbone_feature_map = hidden_states[self.num_fpn_stages - idx - 1]
            top_fpn_feature_map = fpn_feature_maps[-1]
            # apply lateral block
            top_fpn_feature_map = lateral_conv(top_fpn_feature_map)
            fpn_feature_maps[-1] = top_fpn_feature_map
            # apply fpn block
            top_fpn_feature_map = mint.nn.functional.interpolate(top_fpn_feature_map, scale_factor=2.0, mode="nearest")
            fused_feature_map = mint.concat([top_fpn_feature_map, backbone_feature_map], dim=1)
            new_fpn_feature_map = fpn_block(fused_feature_map)
            fpn_feature_maps.append(new_fpn_feature_map)

        fpn_feature_maps = fpn_feature_maps[::-1]

        # bottom-up PAN
        pan_feature_maps = [fpn_feature_maps[0]]
        for idx, (downsample_conv, pan_block) in enumerate(zip(self.downsample_convs, self.pan_blocks)):
            top_pan_feature_map = pan_feature_maps[-1]
            fpn_feature_map = fpn_feature_maps[idx + 1]
            downsampled_feature_map = downsample_conv(top_pan_feature_map)
            fused_feature_map = mint.concat([downsampled_feature_map, fpn_feature_map], dim=1)
            new_pan_feature_map = pan_block(fused_feature_map)
            pan_feature_maps.append(new_pan_feature_map)

        if not return_dict:
            return tuple(v for v in [pan_feature_maps, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=pan_feature_maps, hidden_states=encoder_states, attentions=all_attentions
        )


class RTDetrDecoder(RTDetrPreTrainedModel):
    def __init__(self, config: RTDetrConfig):
        super().__init__(config)

        self.dropout = config.dropout
        self.layers = ms.nn.CellList([RTDetrDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.query_pos_head = RTDetrMLPPredictionHead(config, 4, 2 * config.d_model, config.d_model, num_layers=2)

        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        position_embeddings=None,
        reference_points=None,
        spatial_shapes=None,
        spatial_shapes_list=None,
        level_start_index=None,
        valid_ratios=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            inputs_embeds (`mindspore.Tensor` of shape `(batch_size, num_queries, hidden_size)`):
                The query embeddings that are passed into the decoder.
            encoder_hidden_states (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding pixel_values of the encoder. Mask values selected
                in `[0, 1]`:
                - 1 for pixels that are real (i.e. **not masked**),
                - 0 for pixels that are padding (i.e. **masked**).
            position_embeddings (`mindspore.Tensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*):
                Position embeddings that are added to the queries and keys in each self-attention layer.
            reference_points (`mindspore.Tensor` of shape `(batch_size, num_queries, 4)` is `as_two_stage` else `(batch_size, num_queries, 2)` or , *optional*):
                Reference point in range `[0, 1]`, top-left (0,0), bottom-right (1, 1), including padding area.
            spatial_shapes (`mindspore.Tensor` of shape `(num_feature_levels, 2)`):
                Spatial shapes of the feature maps.
            level_start_index (`mindspore.Tensor` of shape `(num_feature_levels)`, *optional*):
                Indexes for the start of each feature level. In range `[0, sequence_length]`.
            valid_ratios (`mindspore.Tensor` of shape `(batch_size, num_feature_levels, 2)`, *optional*):
                Ratio of valid area in each feature level.

            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is not None:
            hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        intermediate = ()
        intermediate_reference_points = ()
        intermediate_logits = ()

        reference_points = mint.nn.functional.sigmoid(reference_points)

        # https://github.com/lyuwenyu/RT-DETR/blob/94f5e16708329d2f2716426868ec89aa774af016/rtdetr_pytorch/src/zoo/rtdetr/rtdetr_decoder.py#L252
        for idx, decoder_layer in enumerate(self.layers):
            reference_points_input = reference_points.unsqueeze(2)
            position_embeddings = self.query_pos_head(reference_points)

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                encoder_hidden_states=encoder_hidden_states,
                reference_points=reference_points_input,
                spatial_shapes=spatial_shapes,
                spatial_shapes_list=spatial_shapes_list,
                level_start_index=level_start_index,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[idx](hidden_states)
                new_reference_points = mint.nn.functional.sigmoid(tmp + inverse_sigmoid(reference_points))
                reference_points = new_reference_points.clone()

            intermediate += (hidden_states,)
            intermediate_reference_points += (
                (new_reference_points,) if self.bbox_embed is not None else (reference_points,)
            )

            if self.class_embed is not None:
                logits = self.class_embed[idx](hidden_states)
                intermediate_logits += (logits,)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # Keep batch_size as first dimension
        intermediate = mint.stack(intermediate, dim=1)
        intermediate_reference_points = mint.stack(intermediate_reference_points, dim=1)
        if self.class_embed is not None:
            intermediate_logits = mint.stack(intermediate_logits, dim=1)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    intermediate,
                    intermediate_logits,
                    intermediate_reference_points,
                    all_hidden_states,
                    all_self_attns,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return RTDetrDecoderOutput(
            last_hidden_state=hidden_states,
            intermediate_hidden_states=intermediate,
            intermediate_logits=intermediate_logits,
            intermediate_reference_points=intermediate_reference_points,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


# taken from https://github.com/facebookresearch/detr/blob/master/models/detr.py
class RTDetrMLPPredictionHead(ms.nn.Cell):
    """
    Very simple multi-layer perceptron (MLP, also called FFN), used to predict the normalized center coordinates,
    height and width of a bounding box w.r.t. an image.

    Copied from https://github.com/facebookresearch/detr/blob/master/models/detr.py
    Origin from https://github.com/lyuwenyu/RT-DETR/blob/94f5e16708329d2f2716426868ec89aa774af016/rtdetr_paddle/ppdet/modeling/transformers/utils.py#L453

    """

    def __init__(self, config, input_dim, d_model, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [d_model] * (num_layers - 1)
        self.layers = ms.nn.CellList([mint.nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])])

    def construct(self, x):
        for i, layer in enumerate(self.layers):
            x = mint.nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class RTDetrModel(RTDetrPreTrainedModel):
    def __init__(self, config: RTDetrConfig):
        super().__init__(config)

        # Create backbone
        self.backbone = RTDetrConvEncoder(config)
        intermediate_channel_sizes = self.backbone.intermediate_channel_sizes

        # Create encoder input projection layers
        # https://github.com/lyuwenyu/RT-DETR/blob/94f5e16708329d2f2716426868ec89aa774af016/rtdetr_pytorch/src/zoo/rtdetr/hybrid_encoder.py#L212
        num_backbone_outs = len(intermediate_channel_sizes)
        encoder_input_proj_list = []
        for _ in range(num_backbone_outs):
            in_channels = intermediate_channel_sizes[_]
            encoder_input_proj_list.append(
                ms.nn.SequentialCell(
                    mint.nn.Conv2d(in_channels, config.encoder_hidden_dim, kernel_size=1, bias=False),
                    mint.nn.BatchNorm2d(config.encoder_hidden_dim),
                )
            )
        self.encoder_input_proj = ms.nn.CellList(encoder_input_proj_list)

        # Create encoder
        self.encoder = RTDetrHybridEncoder(config)

        # denoising part
        if config.num_denoising > 0:
            self.denoising_class_embed = mint.nn.Embedding(
                config.num_labels + 1, config.d_model, padding_idx=config.num_labels
            )

        # decoder embedding
        if config.learn_initial_query:
            self.weight_embedding = mint.nn.Embedding(config.num_queries, config.d_model)

        # encoder head
        self.enc_output = ms.nn.SequentialCell(
            mint.nn.Linear(config.d_model, config.d_model),
            mint.nn.LayerNorm(config.d_model, eps=config.layer_norm_eps),
        )
        self.enc_score_head = mint.nn.Linear(config.d_model, config.num_labels)
        self.enc_bbox_head = RTDetrMLPPredictionHead(config, config.d_model, config.d_model, 4, num_layers=3)

        # init encoder output anchors and valid_mask
        if config.anchor_image_size:
            self.anchors, self.valid_mask = self.generate_anchors(dtype=self.dtype)

        # Create decoder input projection layers
        # https://github.com/lyuwenyu/RT-DETR/blob/94f5e16708329d2f2716426868ec89aa774af016/rtdetr_pytorch/src/zoo/rtdetr/rtdetr_decoder.py#L412
        num_backbone_outs = len(config.decoder_in_channels)
        decoder_input_proj_list = []
        for _ in range(num_backbone_outs):
            in_channels = config.decoder_in_channels[_]
            decoder_input_proj_list.append(
                ms.nn.SequentialCell(
                    mint.nn.Conv2d(in_channels, config.d_model, kernel_size=1, bias=False),
                    mint.nn.BatchNorm2d(config.d_model, config.batch_norm_eps),
                )
            )
        for _ in range(config.num_feature_levels - num_backbone_outs):
            decoder_input_proj_list.append(
                ms.nn.SequentialCell(
                    mint.nn.Conv2d(in_channels, config.d_model, kernel_size=3, stride=2, padding=1, bias=False),
                    mint.nn.BatchNorm2d(config.d_model, config.batch_norm_eps),
                )
            )
            in_channels = config.d_model
        self.decoder_input_proj = ms.nn.CellList(decoder_input_proj_list)

        # decoder
        self.decoder = RTDetrDecoder(config)

        self.post_init()

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad_(False)

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad_(True)

    def generate_anchors(self, spatial_shapes=None, grid_size=0.05, dtype=ms.float32):
        if spatial_shapes is None:
            spatial_shapes = [
                [int(self.config.anchor_image_size[0] / s), int(self.config.anchor_image_size[1] / s)]
                for s in self.config.feat_strides
            ]
        anchors = []
        for level, (height, width) in enumerate(spatial_shapes):
            grid_y, grid_x = mint.meshgrid(
                mint.arange(end=height, ).to(dtype),
                mint.arange(end=width, ).to(dtype),
                indexing="ij",
            )
            grid_xy = mint.stack([grid_x, grid_y], -1)
            grid_xy = grid_xy.unsqueeze(0) + 0.5
            grid_xy[..., 0] /= width
            grid_xy[..., 1] /= height
            wh = mint.ones_like(grid_xy) * grid_size * (2.0**level)
            anchors.append(mint.concat([grid_xy, wh], -1).reshape(-1, height * width, 4))
        # define the valid range for anchor coordinates
        eps = 1e-2
        anchors = mint.concat(anchors, 1)
        valid_mask = ((anchors > eps) * (anchors < 1 - eps)).all(-1, keepdim=True)
        anchors = mint.log(anchors / (1 - anchors))
        anchors = mint.where(valid_mask, anchors, ms.Tensor(dtype_to_max(dtype), dtype=dtype, ))

        return anchors, valid_mask

    def construct(
        self,
        pixel_values: ms.Tensor,
        pixel_mask: Optional[ms.Tensor] = None,
        encoder_outputs: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        decoder_inputs_embeds: Optional[ms.Tensor] = None,
        labels: Optional[List[dict]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[ms.Tensor], RTDetrModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, RTDetrModel
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")
        >>> model = RTDetrModel.from_pretrained("PekingU/rtdetr_r50vd")

        >>> inputs = image_processor(images=image, return_tensors="np")

        >>> outputs = model(**inputs)

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
            pixel_mask = mint.ones(((batch_size, height, width)), )

        features = self.backbone(pixel_values, pixel_mask)

        proj_feats = [self.encoder_input_proj[level](source) for level, (source, mask) in enumerate(features)]

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                proj_feats,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if output_hidden_states else None,
                attentions=encoder_outputs[2]
                if len(encoder_outputs) > 2
                else encoder_outputs[1]
                if output_attentions
                else None,
            )

        # Equivalent to def _get_encoder_input
        # https://github.com/lyuwenyu/RT-DETR/blob/94f5e16708329d2f2716426868ec89aa774af016/rtdetr_pytorch/src/zoo/rtdetr/rtdetr_decoder.py#L412
        sources = []
        for level, source in enumerate(encoder_outputs[0]):
            sources.append(self.decoder_input_proj[level](source))

        # Lowest resolution feature maps are obtained via 3x3 stride 2 convolutions on the final stage
        if self.config.num_feature_levels > len(sources):
            _len_sources = len(sources)
            sources.append(self.decoder_input_proj[_len_sources](encoder_outputs[0])[-1])
            for i in range(_len_sources + 1, self.config.num_feature_levels):
                sources.append(self.decoder_input_proj[i](encoder_outputs[0][-1]))

        # Prepare encoder inputs (by flattening)
        source_flatten = []
        spatial_shapes_list = []
        spatial_shapes = mint.empty((len(sources), 2), dtype=ms.int64)
        for level, source in enumerate(sources):
            height, width = source.shape[-2:]
            spatial_shapes[level, 0] = height
            spatial_shapes[level, 1] = width
            spatial_shapes_list.append((height, width))
            source = source.flatten(2).transpose(1, 2)
            source_flatten.append(source)
        source_flatten = mint.cat(source_flatten, 1)
        level_start_index = mint.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        # prepare denoising training
        if self.training and self.config.num_denoising > 0 and labels is not None:
            (
                denoising_class,
                denoising_bbox_unact,
                attention_mask,
                denoising_meta_values,
            ) = get_contrastive_denoising_training_group(
                targets=labels,
                num_classes=self.config.num_labels,
                num_queries=self.config.num_queries,
                class_embed=self.denoising_class_embed,
                num_denoising_queries=self.config.num_denoising,
                label_noise_ratio=self.config.label_noise_ratio,
                box_noise_scale=self.config.box_noise_scale,
            )
        else:
            denoising_class, denoising_bbox_unact, attention_mask, denoising_meta_values = None, None, None, None

        batch_size = len(source_flatten)
        dtype = source_flatten.dtype

        # prepare input for decoder
        if self.training or self.config.anchor_image_size is None:
            # Pass spatial_shapes as tuple to make it hashable and make sure
            # lru_cache is working for generate_anchors()
            spatial_shapes_tuple = tuple(spatial_shapes_list)
            anchors, valid_mask = self.generate_anchors(spatial_shapes_tuple, dtype=dtype)
        else:
            anchors, valid_mask = self.anchors, self.valid_mask
            anchors, valid_mask = anchors.astype(dtype), valid_mask.astype(dtype)

        # use the valid_mask to selectively retain values in the feature map where the mask is `True`
        memory = valid_mask.to(source_flatten.dtype) * source_flatten

        output_memory = self.enc_output(memory)

        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_logits = self.enc_bbox_head(output_memory) + anchors

        _, topk_ind = mint.topk(mint.max(enc_outputs_class, dim=-1)[0], self.config.num_queries, dim=1)

        reference_points_unact = enc_outputs_coord_logits.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_logits.shape[-1])
        )

        enc_topk_bboxes = mint.nn.functional.sigmoid(reference_points_unact)
        if denoising_bbox_unact is not None:
            reference_points_unact = mint.concat([denoising_bbox_unact, reference_points_unact], 1)

        enc_topk_logits = enc_outputs_class.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1])
        )

        # extract region features
        if self.config.learn_initial_query:
            target = self.weight_embedding.tile([batch_size, 1, 1])
        else:
            target = output_memory.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
            target = target.clone()

        if denoising_class is not None:
            target = mint.concat([denoising_class, target], 1)

        init_reference_points = reference_points_unact.clone()

        # decoder
        decoder_outputs = self.decoder(
            inputs_embeds=target,
            encoder_hidden_states=source_flatten,
            encoder_attention_mask=attention_mask,
            reference_points=init_reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            enc_outputs = tuple(
                value
                for value in [enc_topk_logits, enc_topk_bboxes, enc_outputs_class, enc_outputs_coord_logits]
                if value is not None
            )
            dn_outputs = tuple(value if value is not None else None for value in [denoising_meta_values])
            tuple_outputs = decoder_outputs + encoder_outputs + (init_reference_points,) + enc_outputs + dn_outputs

            return tuple_outputs

        return RTDetrModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            intermediate_hidden_states=decoder_outputs.intermediate_hidden_states,
            intermediate_logits=decoder_outputs.intermediate_logits,
            intermediate_reference_points=decoder_outputs.intermediate_reference_points,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            init_reference_points=init_reference_points,
            enc_topk_logits=enc_topk_logits,
            enc_topk_bboxes=enc_topk_bboxes,
            enc_outputs_class=enc_outputs_class,
            enc_outputs_coord_logits=enc_outputs_coord_logits,
            denoising_meta_values=denoising_meta_values,
        )


class RTDetrForObjectDetection(RTDetrPreTrainedModel):
    # When using clones, all layers > 0 will be clones, but layer 0 *is* required
    _tied_weights_keys = ["bbox_embed", "class_embed"]
    # We can't initialize the model on meta device as some weights are modified during the initialization
    _no_split_modules = None

    def __init__(self, config: RTDetrConfig):
        super().__init__(config)

        # RTDETR encoder-decoder model
        self.model = RTDetrModel(config)

        # Detection heads on top
        self.class_embed = partial(mint.nn.Linear, config.d_model, config.num_labels)
        self.bbox_embed = partial(RTDetrMLPPredictionHead, config, config.d_model, config.d_model, 4, num_layers=3)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = config.decoder_layers
        if config.with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
        else:
            self.class_embed = ms.nn.CellList([self.class_embed() for _ in range(num_pred)])
            self.bbox_embed = ms.nn.CellList([self.bbox_embed() for _ in range(num_pred)])

        # hack implementation for iterative bounding box refinement
        self.model.decoder.class_embed = self.class_embed
        self.model.decoder.bbox_embed = self.bbox_embed

        # Initialize weights and apply final processing
        self.post_init()

    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class, outputs_coord)]

    def construct(
        self,
        pixel_values: ms.Tensor,
        pixel_mask: Optional[ms.Tensor] = None,
        encoder_outputs: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        decoder_inputs_embeds: Optional[ms.Tensor] = None,
        labels: Optional[List[dict]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **loss_kwargs,
    ) -> Union[Tuple[ms.Tensor], RTDetrObjectDetectionOutput]:
        r"""
        labels (`List[Dict]` of len `(batch_size,)`, *optional*):
            Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
            following 2 keys: 'class_labels' and 'boxes' (the class labels and bounding boxes of an image in the batch
            respectively). The class labels themselves should be a `mindspore.Tensor` of len `(number of bounding boxes
            in the image,)` and the boxes a `mindspore.Tensor` of shape `(number of bounding boxes in the image, 4)`.

        Returns:

        Examples:

        ```python
        >>> from transformers import RTDetrImageProcessor, RTDetrForObjectDetection
        >>> from PIL import Image
        >>> import requests
        >>> import mindspore as ms

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")
        >>> model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd")

        >>> # prepare image for the model
        >>> inputs = image_processor(images=image, return_tensors="np")

        >>> # forward pass
        >>> outputs = model(**inputs)

        >>> logits = outputs.logits
        >>> list(logits.shape)
        [1, 300, 80]

        >>> boxes = outputs.pred_boxes
        >>> list(boxes.shape)
        [1, 300, 4]

        >>> # convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
        >>> target_sizes = ms.Tensor([image.size[::-1]])
        >>> results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[
        ...     0
        ... ]

        >>> for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        ...     box = [round(i, 2) for i in box.tolist()]
        ...     print(
        ...         f"Detected {model.config.id2label[label.item()]} with confidence "
        ...         f"{round(score.item(), 3)} at location {box}"
        ...     )
        Detected sofa with confidence 0.97 at location [0.14, 0.38, 640.13, 476.21]
        Detected cat with confidence 0.96 at location [343.38, 24.28, 640.14, 371.5]
        Detected cat with confidence 0.958 at location [13.23, 54.18, 318.98, 472.22]
        Detected remote with confidence 0.951 at location [40.11, 73.44, 175.96, 118.48]
        Detected remote with confidence 0.924 at location [333.73, 76.58, 369.97, 186.99]
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            pixel_values,
            pixel_mask=pixel_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        denoising_meta_values = (
            outputs.denoising_meta_values if return_dict else outputs[-1] if self.training else None
        )

        outputs_class = outputs.intermediate_logits if return_dict else outputs[2]
        outputs_coord = outputs.intermediate_reference_points if return_dict else outputs[3]

        logits = outputs_class[:, -1]
        pred_boxes = outputs_coord[:, -1]

        loss, loss_dict, auxiliary_outputs, enc_topk_logits, enc_topk_bboxes = None, None, None, None, None
        if labels is not None:
            enc_topk_logits = outputs.enc_topk_logits if return_dict else outputs[-5]
            enc_topk_bboxes = outputs.enc_topk_bboxes if return_dict else outputs[-4]
            loss, loss_dict, auxiliary_outputs = self.loss_function(
                logits,
                labels,
                pred_boxes,
                self.config,
                outputs_class,
                outputs_coord,
                enc_topk_logits=enc_topk_logits,
                enc_topk_bboxes=enc_topk_bboxes,
                denoising_meta_values=denoising_meta_values,
                **loss_kwargs,
            )

        if not return_dict:
            if auxiliary_outputs is not None:
                output = (logits, pred_boxes) + (auxiliary_outputs,) + outputs
            else:
                output = (logits, pred_boxes) + outputs
            return ((loss, loss_dict) + output) if loss is not None else output

        return RTDetrObjectDetectionOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_boxes=pred_boxes,
            auxiliary_outputs=auxiliary_outputs,
            last_hidden_state=outputs.last_hidden_state,
            intermediate_hidden_states=outputs.intermediate_hidden_states,
            intermediate_logits=outputs.intermediate_logits,
            intermediate_reference_points=outputs.intermediate_reference_points,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            init_reference_points=outputs.init_reference_points,
            enc_topk_logits=outputs.enc_topk_logits,
            enc_topk_bboxes=outputs.enc_topk_bboxes,
            enc_outputs_class=outputs.enc_outputs_class,
            enc_outputs_coord_logits=outputs.enc_outputs_coord_logits,
            denoising_meta_values=outputs.denoising_meta_values,
        )


__all__ = [
    "RTDetrForObjectDetection",
    "RTDetrModel",
    "RTDetrPreTrainedModel",
]
