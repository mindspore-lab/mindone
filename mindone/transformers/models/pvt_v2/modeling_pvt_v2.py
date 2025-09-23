# coding=utf-8
# Copyright 2024 Authors: Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan,
# Kaitao Song, Ding Liang, Tong Lu, Ping Luo, Ling Shao and The HuggingFace Inc. team.
# All rights reserved.
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
"""MindSpore PVTv2 model."""

import math
from typing import Optional, Tuple, Union

from transformers.models.pvt_v2.configuration_pvt_v2 import PvtV2Config

import mindspore as ms
from mindspore import mint, nn
from mindspore.mint.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...mindspore_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...modeling_outputs import BackboneOutput, BaseModelOutput, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from ...utils.backbone_utils import BackboneMixin

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "PvtV2Config"

_CHECKPOINT_FOR_DOC = "OpenGVLab/pvt_v2_b0"
_EXPECTED_OUTPUT_SHAPE = [1, 256, 7, 7]

_IMAGE_CLASS_CHECKPOINT = "OpenGVLab/pvt_v2_b0"
_IMAGE_CLASS_EXPECTED_OUTPUT = "LABEL_281"  # ImageNet ID for "tabby, tabby cat"


# Copied from transformers.models.beit.modeling_beit.drop_path
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


# Copied from transformers.models.convnext.modeling_convnext.ConvNextDropPath with ConvNext->Pvt
class PvtV2DropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class PvtV2OverlapPatchEmbeddings(nn.Cell):
    """Image to Patch Embedding"""

    def __init__(self, config: PvtV2Config, layer_idx: int):
        super().__init__()
        patch_size = config.patch_sizes[layer_idx]
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        stride = config.strides[layer_idx]
        num_channels = config.num_channels if layer_idx == 0 else config.hidden_sizes[layer_idx - 1]
        hidden_size = config.hidden_sizes[layer_idx]
        self.patch_size = patch_size
        self.proj = mint.nn.Conv2d(
            num_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.layer_norm = mint.nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)

    def construct(self, pixel_values):
        embeddings = self.proj(pixel_values)
        _, _, height, width = embeddings.shape
        embeddings = embeddings.flatten(2).swapaxes(1, 2)
        embeddings = self.layer_norm(embeddings)
        return embeddings, height, width


class PvtV2DepthWiseConv(nn.Cell):
    """
    Depth-wise (DW) convolution to infuse positional information using zero-padding. Depth-wise convolutions
    have an equal number of groups to the number of input channels, meaning one filter per input channel. This
    reduces the overall parameters and compute costs since the key purpose of this layer is position encoding.
    """

    def __init__(self, config: PvtV2Config, dim: int = 768):
        super().__init__()
        self.dwconv = mint.nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def construct(self, hidden_states, height, width):
        batch_size, seq_len, num_channels = hidden_states.shape
        hidden_states = hidden_states.swapaxes(1, 2).view(batch_size, num_channels, height, width)
        hidden_states = self.dwconv(hidden_states)
        hidden_states = hidden_states.flatten(2).swapaxes(1, 2)

        return hidden_states


class PvtV2SelfAttention(nn.Cell):
    """Efficient self-attention mechanism."""

    def __init__(self, config: PvtV2Config, hidden_size: int, num_attention_heads: int, spatial_reduction_ratio: int):
        super().__init__()
        self.linear_attention = config.linear_attention
        self.pruned_heads = set()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({self.num_attention_heads})"
            )

        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = mint.nn.Linear(self.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = mint.nn.Linear(self.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = mint.nn.Linear(self.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.attn_drop = mint.nn.Dropout(config.attention_probs_dropout_prob)
        self.proj = mint.nn.Linear(self.hidden_size, self.hidden_size)
        self.proj_drop = mint.nn.Dropout(config.hidden_dropout_prob)

        self.spatial_reduction_ratio = spatial_reduction_ratio
        if self.linear_attention:
            self.pool = mint.nn.AdaptiveAvgPool2d(7)
            self.spatial_reduction = mint.nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=1, stride=1)
            self.layer_norm = mint.nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
            self.act = mint.nn.GELU()
        elif spatial_reduction_ratio > 1:
            self.spatial_reduction = mint.nn.Conv2d(
                self.hidden_size, self.hidden_size, kernel_size=spatial_reduction_ratio, stride=spatial_reduction_ratio
            )
            self.layer_norm = mint.nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)

    def transpose_for_scores(self, hidden_states) -> ms.Tensor:
        new_shape = hidden_states.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        hidden_states = hidden_states.view(new_shape)
        return hidden_states.permute(0, 2, 1, 3)

    def construct(
        self,
        hidden_states: ms.Tensor,
        height: int,
        width: int,
        output_attentions: bool = False,
    ) -> Tuple[ms.Tensor]:
        batch_size, seq_len, num_channels = hidden_states.shape
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        if self.linear_attention:
            hidden_states = hidden_states.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)
            hidden_states = (
                self.spatial_reduction(self.pool(hidden_states)).reshape(batch_size, num_channels, -1).permute(0, 2, 1)
            )
            hidden_states = self.act(self.layer_norm(hidden_states))
        elif self.spatial_reduction_ratio > 1:
            hidden_states = hidden_states.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)
            hidden_states = self.spatial_reduction(hidden_states).reshape(batch_size, num_channels, -1).permute(0, 2, 1)
            hidden_states = self.layer_norm(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = mint.matmul(query_layer, key_layer.swapaxes(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = mint.nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attn_drop(attention_probs)
        context_layer = (attention_probs @ value_layer).swapaxes(1, 2).reshape(batch_size, seq_len, num_channels)
        context_layer = self.proj(context_layer)
        context_layer = self.proj_drop(context_layer)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.num_attention_heads, self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.query = prune_linear_layer(self.query, index)
        self.key = prune_linear_layer(self.key, index)
        self.value = prune_linear_layer(self.value, index)
        self.proj = prune_linear_layer(self.proj, index, dim=1)

        # Update hyper params and store pruned heads
        self.num_attention_heads = self.num_attention_heads - len(heads)
        self.all_head_size = self.attention_head_size * self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)


class PvtV2ConvFeedForwardNetwork(nn.Cell):
    def __init__(
        self,
        config: PvtV2Config,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
    ):
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        self.dense1 = mint.nn.Linear(in_features, hidden_features)
        self.dwconv = PvtV2DepthWiseConv(config, hidden_features)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        self.dense2 = mint.nn.Linear(hidden_features, out_features)
        self.dropout = mint.nn.Dropout(config.hidden_dropout_prob)
        self.relu = mint.nn.ReLU() if config.linear_attention else mint.nn.Identity()

    def construct(self, hidden_states: ms.Tensor, height, width) -> ms.Tensor:
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.relu(hidden_states)
        hidden_states = self.dwconv(hidden_states, height, width)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class PvtV2BlockLayer(nn.Cell):
    def __init__(self, config: PvtV2Config, layer_idx: int, drop_path: float = 0.0):
        super().__init__()
        hidden_size: int = config.hidden_sizes[layer_idx]
        num_attention_heads: int = config.num_attention_heads[layer_idx]
        spatial_reduction_ratio: int = config.sr_ratios[layer_idx]
        mlp_ratio: float = config.mlp_ratios[layer_idx]
        self.layer_norm_1 = mint.nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.attention = PvtV2SelfAttention(
            config=config,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            spatial_reduction_ratio=spatial_reduction_ratio,
        )
        self.drop_path = PvtV2DropPath(drop_path) if drop_path > 0.0 else mint.nn.Identity()
        self.layer_norm_2 = mint.nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        self.mlp = PvtV2ConvFeedForwardNetwork(config=config, in_features=hidden_size, hidden_features=mlp_hidden_size)

    def construct(self, hidden_states: ms.Tensor, height: int, width: int, output_attentions: bool = False):
        self_attention_outputs = self.attention(
            hidden_states=self.layer_norm_1(hidden_states),
            height=height,
            width=width,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        attention_output = self.drop_path(attention_output)
        hidden_states = attention_output + hidden_states

        mlp_output = self.mlp(self.layer_norm_2(hidden_states), height, width)

        mlp_output = self.drop_path(mlp_output)
        layer_output = hidden_states + mlp_output

        outputs = (layer_output,) + outputs

        return outputs


class PvtV2EncoderLayer(nn.Cell):
    def __init__(self, config: PvtV2Config, layer_idx: int):
        super().__init__()
        self.patch_embedding = PvtV2OverlapPatchEmbeddings(
            config=config,
            layer_idx=layer_idx,
        )
        # Transformer block
        # stochastic depth decay rule
        drop_path_decays = mint.linspace(0, config.drop_path_rate, sum(config.depths)).tolist()
        block_layers = []
        for block_idx in range(config.depths[layer_idx]):
            block_layers.append(
                PvtV2BlockLayer(
                    config=config,
                    layer_idx=layer_idx,
                    drop_path=drop_path_decays[sum(config.depths[:layer_idx]) + block_idx],
                )
            )
        self.blocks = nn.CellList(block_layers)

        # Layer norm
        self.layer_norm = mint.nn.LayerNorm(config.hidden_sizes[layer_idx], eps=config.layer_norm_eps)

    def construct(self, hidden_states, output_attentions):
        all_self_attentions = () if output_attentions else None
        # first, obtain patch embeddings
        hidden_states, height, width = self.patch_embedding(hidden_states)
        # second, send embeddings through blocks
        for block in self.blocks:
            layer_outputs = block(hidden_states, height, width, output_attentions)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions += (layer_outputs[1],)
        # third, apply layer norm
        hidden_states = self.layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (all_self_attentions,)

        return outputs, height, width


class PvtV2Encoder(nn.Cell):
    def __init__(self, config: PvtV2Config):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False

        # encoder layers
        self.layers = nn.CellList([PvtV2EncoderLayer(config, i) for i in range(config.num_encoder_blocks)])

    def construct(
        self,
        pixel_values: ms.Tensor,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        batch_size = pixel_values.shape[0]
        hidden_states = pixel_values
        for idx, layer in enumerate(self.layers):
            layer_output = layer(hidden_states, output_attentions)
            outputs, height, width = layer_output
            hidden_states = outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[1],)
            # reshape back to (batch_size, num_channels, height, width)
            hidden_states = hidden_states.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class PvtV2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = PvtV2Config
    base_model_prefix = "pvt_v2"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module: Union[mint.nn.Linear, mint.nn.Conv2d, mint.nn.LayerNorm]) -> None:
        """Initialize the weights"""
        pass


PVT_V2_START_DOCSTRING = r"""
    This model is a MindSpore
    [ms.nn.Cell](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Cell.html) sub-class. Use
    it as a regular MindSpore Cell and refer to the MindSpore documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`~PvtV2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

PVT_V2_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`ms.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`PvtImageProcessor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class PvtV2Model(PvtV2PreTrainedModel):
    def __init__(self, config: PvtV2Config):
        super().__init__(config)
        self.config = config

        # hierarchical Transformer encoder
        self.encoder = PvtV2Encoder(config)

        # Initialize weights and apply final processing
        self.post_init()

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
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class PvtV2ForImageClassification(PvtV2PreTrainedModel):
    def __init__(self, config: PvtV2Config) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.pvt_v2 = PvtV2Model(config)

        # Classifier head
        self.classifier = (
            mint.nn.Linear(config.hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else mint.nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        pixel_values: Optional[ms.Tensor],
        labels: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`ms.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.pvt_v2(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # convert last hidden states to (batch_size, height*width, hidden_size)
        batch_size = sequence_output.shape[0]
        # (batch_size, num_channels, height, width) -> (batch_size, height, width, num_channels)
        sequence_output = sequence_output.permute(0, 2, 3, 1)
        sequence_output = sequence_output.reshape(batch_size, -1, self.config.hidden_sizes[-1])

        # global average pooling
        sequence_output = sequence_output.mean(dim=1)

        logits = self.classifier(sequence_output)

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
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class PvtV2Backbone(PvtV2Model, BackboneMixin):
    def __init__(self, config: PvtV2Config):
        super().__init__(config)
        super()._init_backbone(config)
        self.num_features = config.hidden_sizes

    def construct(
        self,
        pixel_values: ms.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
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

        >>> processor = AutoImageProcessor.from_pretrained("OpenGVLab/pvt_v2_b0")
        >>> model = AutoBackbone.from_pretrained(
        ...     "OpenGVLab/pvt_v2_b0", out_features=["stage1", "stage2", "stage3", "stage4"]
        ... )

        >>> inputs = processor(image, return_tensors="np")
        >>> inputs = {k: ms.tensor(v) for k, v in inputs.items()}

        >>> outputs = model(**inputs)
        >>> feature_maps = outputs.feature_maps
        >>> list(feature_maps[-1].shape)
        [1, 256, 7, 7]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.encoder(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        hidden_states = outputs.hidden_states

        feature_maps = ()
        for idx, stage in enumerate(self.stage_names):
            if stage in self.out_features:
                feature_maps += (hidden_states[idx],)

        if not return_dict:
            output = (feature_maps,)
            if output_hidden_states:
                output += (outputs.hidden_states,)
            return output

        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=None,
        )


__all__ = ["PvtV2ForImageClassification", "PvtV2Model", "PvtV2PreTrainedModel", "PvtV2Backbone"]
