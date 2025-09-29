# coding=utf-8
# Copyright 2023 MBZUAI and The HuggingFace Inc. team. All rights reserved.
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
"""MindSpore SwiftFormer model."""

import collections.abc
from typing import Optional, Tuple, Union

from transformers.models.swiftformer.configuration_swiftformer import SwiftFormerConfig

import mindspore as ms
from mindspore import mint, nn
from mindspore.mint.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2CLS
from ...modeling_outputs import BaseModelOutputWithNoAttention, ImageClassifierOutputWithNoAttention
from ...modeling_utils import PreTrainedModel
from ...utils import logging

logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "SwiftFormerConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "MBZUAI/swiftformer-xs"
_EXPECTED_OUTPUT_SHAPE = [1, 220, 7, 7]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "MBZUAI/swiftformer-xs"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"


class SwiftFormerPatchEmbedding(nn.Cell):
    """
    Patch Embedding Layer constructed of two 2D convolutional layers.

    Input: tensor of shape `[batch_size, in_channels, height, width]`

    Output: tensor of shape `[batch_size, out_channels, height/4, width/4]`
    """

    def __init__(self, config: SwiftFormerConfig):
        super().__init__()

        in_chs = config.num_channels
        out_chs = config.embed_dims[0]
        self.patch_embedding = nn.SequentialCell(
            mint.nn.Conv2d(in_chs, out_chs // 2, kernel_size=3, stride=2, padding=1),
            mint.nn.BatchNorm2d(out_chs // 2, eps=config.batch_norm_eps),
            mint.nn.ReLU(),
            mint.nn.Conv2d(out_chs // 2, out_chs, kernel_size=3, stride=2, padding=1),
            mint.nn.BatchNorm2d(out_chs, eps=config.batch_norm_eps),
            mint.nn.ReLU(),
        )

    def construct(self, x):
        return self.patch_embedding(x)


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


class SwiftFormerDropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, config: SwiftFormerConfig) -> None:
        super().__init__()
        self.drop_prob = config.drop_path_rate

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class SwiftFormerEmbeddings(nn.Cell):
    """
    Embeddings layer consisting of a single 2D convolutional and batch normalization layer.

    Input: tensor of shape `[batch_size, channels, height, width]`

    Output: tensor of shape `[batch_size, channels, height/stride, width/stride]`
    """

    def __init__(self, config: SwiftFormerConfig, index: int):
        super().__init__()

        patch_size = config.down_patch_size
        stride = config.down_stride
        padding = config.down_pad
        embed_dims = config.embed_dims

        in_chans = embed_dims[index]
        embed_dim = embed_dims[index + 1]

        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        stride = stride if isinstance(stride, collections.abc.Iterable) else (stride, stride)
        padding = padding if isinstance(padding, collections.abc.Iterable) else (padding, padding)

        self.proj = mint.nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = mint.nn.BatchNorm2d(embed_dim, eps=config.batch_norm_eps)

    def construct(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class SwiftFormerConvEncoder(nn.Cell):
    """
    `SwiftFormerConvEncoder` with 3*3 and 1*1 convolutions.

    Input: tensor of shape `[batch_size, channels, height, width]`

    Output: tensor of shape `[batch_size, channels, height, width]`
    """

    def __init__(self, config: SwiftFormerConfig, dim: int):
        super().__init__()
        hidden_dim = int(config.mlp_ratio * dim)

        self.depth_wise_conv = mint.nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = mint.nn.BatchNorm2d(dim, eps=config.batch_norm_eps)
        self.point_wise_conv1 = mint.nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.act = mint.nn.GELU()
        self.point_wise_conv2 = mint.nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.drop_path = mint.nn.Dropout(p=config.drop_conv_encoder_rate)
        self.layer_scale = ms.Parameter(
            mint.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True, name="layer_scale"
        )

    def construct(self, x):
        input = x
        x = self.depth_wise_conv(x)
        x = self.norm(x)
        x = self.point_wise_conv1(x)
        x = self.act(x)
        x = self.point_wise_conv2(x)
        x = input + self.drop_path(self.layer_scale * x)
        return x


class SwiftFormerMlp(nn.Cell):
    """
    MLP layer with 1*1 convolutions.

    Input: tensor of shape `[batch_size, channels, height, width]`

    Output: tensor of shape `[batch_size, channels, height, width]`
    """

    def __init__(self, config: SwiftFormerConfig, in_features: int):
        super().__init__()
        hidden_features = int(in_features * config.mlp_ratio)
        self.norm1 = mint.nn.BatchNorm2d(in_features, eps=config.batch_norm_eps)
        self.fc1 = mint.nn.Conv2d(in_features, hidden_features, 1)
        act_layer = ACT2CLS[config.hidden_act]
        self.act = act_layer()
        self.fc2 = mint.nn.Conv2d(hidden_features, in_features, 1)
        self.drop = mint.nn.Dropout(p=config.drop_mlp_rate)

    def construct(self, x):
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiftFormerEfficientAdditiveAttention(nn.Cell):
    """
    Efficient Additive Attention module for SwiftFormer.

    Input: tensor of shape `[batch_size, channels, height, width]`

    Output: tensor of shape `[batch_size, channels, height, width]`
    """

    def __init__(self, config: SwiftFormerConfig, dim: int = 512):
        super().__init__()

        self.to_query = mint.nn.Linear(dim, dim)
        self.to_key = mint.nn.Linear(dim, dim)

        self.w_g = ms.Parameter(mint.randn(dim, 1), name="w_g")
        self.scale_factor = dim**-0.5
        self.proj = mint.nn.Linear(dim, dim)
        self.final = mint.nn.Linear(dim, dim)

    def construct(self, x):
        query = self.to_query(x)
        key = self.to_key(x)

        query = mint.nn.functional.normalize(query, dim=-1)
        key = mint.nn.functional.normalize(key, dim=-1)

        query_weight = query @ self.w_g
        scaled_query_weight = query_weight * self.scale_factor
        scaled_query_weight = scaled_query_weight.softmax(axis=-1)

        global_queries = mint.sum(scaled_query_weight * query, dim=1)
        global_queries = global_queries.unsqueeze(1).tile((1, key.shape[1], 1))

        out = self.proj(global_queries * key) + query
        out = self.final(out)

        return out


class SwiftFormerLocalRepresentation(nn.Cell):
    """
    Local Representation module for SwiftFormer that is implemented by 3*3 depth-wise and point-wise convolutions.

    Input: tensor of shape `[batch_size, channels, height, width]`

    Output: tensor of shape `[batch_size, channels, height, width]`
    """

    def __init__(self, config: SwiftFormerConfig, dim: int):
        super().__init__()

        self.depth_wise_conv = mint.nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = mint.nn.BatchNorm2d(dim, eps=config.batch_norm_eps)
        self.point_wise_conv1 = mint.nn.Conv2d(dim, dim, kernel_size=1)
        self.act = mint.nn.GELU()
        self.point_wise_conv2 = mint.nn.Conv2d(dim, dim, kernel_size=1)
        self.drop_path = mint.nn.Identity()
        self.layer_scale = ms.Parameter(
            mint.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True, name="layer_scale"
        )

    def construct(self, x):
        input = x
        x = self.depth_wise_conv(x)
        x = self.norm(x)
        x = self.point_wise_conv1(x)
        x = self.act(x)
        x = self.point_wise_conv2(x)
        x = input + self.drop_path(self.layer_scale * x)
        return x


class SwiftFormerEncoderBlock(nn.Cell):
    """
    SwiftFormer Encoder Block for SwiftFormer. It consists of (1) Local representation module, (2)
    SwiftFormerEfficientAdditiveAttention, and (3) MLP block.

    Input: tensor of shape `[batch_size, channels, height, width]`

    Output: tensor of shape `[batch_size, channels,height, width]`
    """

    def __init__(self, config: SwiftFormerConfig, dim: int, drop_path: float = 0.0) -> None:
        super().__init__()

        layer_scale_init_value = config.layer_scale_init_value
        use_layer_scale = config.use_layer_scale

        self.local_representation = SwiftFormerLocalRepresentation(config, dim=dim)
        self.attn = SwiftFormerEfficientAdditiveAttention(config, dim=dim)
        self.linear = SwiftFormerMlp(config, in_features=dim)
        self.drop_path = SwiftFormerDropPath(config) if drop_path > 0.0 else mint.nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = ms.Parameter(
                layer_scale_init_value * mint.ones(dim).unsqueeze(-1).unsqueeze(-1),
                requires_grad=True,
                name="layer_scale_1",
            )
            self.layer_scale_2 = ms.Parameter(
                layer_scale_init_value * mint.ones(dim).unsqueeze(-1).unsqueeze(-1),
                requires_grad=True,
                name="layer_scale_2",
            )

    def construct(self, x):
        x = self.local_representation(x)
        batch_size, channels, height, width = x.shape
        res = self.attn(x.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels))
        res = res.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 * res)
            x = x + self.drop_path(self.layer_scale_2 * self.linear(x))
        else:
            x = x + self.drop_path(res)
            x = x + self.drop_path(self.linear(x))
        return x


class SwiftFormerStage(nn.Cell):
    """
    A Swiftformer stage consisting of a series of `SwiftFormerConvEncoder` blocks and a final
    `SwiftFormerEncoderBlock`.

    Input: tensor in shape `[batch_size, channels, height, width]`

    Output: tensor in shape `[batch_size, channels, height, width]`
    """

    def __init__(self, config: SwiftFormerConfig, index: int) -> None:
        super().__init__()

        layer_depths = config.depths
        dim = config.embed_dims[index]
        depth = layer_depths[index]

        blocks = []
        for block_idx in range(depth):
            block_dpr = config.drop_path_rate * (block_idx + sum(layer_depths[:index])) / (sum(layer_depths) - 1)

            if depth - block_idx <= 1:
                blocks.append(SwiftFormerEncoderBlock(config, dim=dim, drop_path=block_dpr))
            else:
                blocks.append(SwiftFormerConvEncoder(config, dim=dim))

        self.blocks = nn.CellList(blocks)

    def construct(self, input):
        for block in self.blocks:
            input = block(input)
        return input


class SwiftFormerEncoder(nn.Cell):
    def __init__(self, config: SwiftFormerConfig) -> None:
        super().__init__()
        self.config = config

        embed_dims = config.embed_dims
        downsamples = config.downsamples
        layer_depths = config.depths

        # Transformer model
        network = []
        for i in range(len(layer_depths)):
            stage = SwiftFormerStage(config=config, index=i)
            network.append(stage)
            if i >= len(layer_depths) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                # downsampling between two stages
                network.append(SwiftFormerEmbeddings(config, index=i))
        self.network = nn.CellList(network)

        self.gradient_checkpointing = False

    def construct(
        self,
        hidden_states: ms.Tensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, BaseModelOutputWithNoAttention]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        all_hidden_states = (hidden_states,) if output_hidden_states else None

        for block in self.network:
            hidden_states = block(hidden_states)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )


class SwiftFormerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SwiftFormerConfig
    base_model_prefix = "swiftformer"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["SwiftFormerEncoderBlock"]

    def _init_weights(self, module: Union[mint.nn.Linear, mint.nn.Conv2d, mint.nn.LayerNorm]) -> None:
        """Initialize the weights"""
        pass


SWIFTFORMER_START_DOCSTRING = r"""
    This model is a MindSpore
    [ms.nn.Cell](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Cell.html) subclass. Use it
    as a regular MindSpore Cell and refer to the MindSpore documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`SwiftFormerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

SWIFTFORMER_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`ms.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`ViTImageProcessor.__call__`]
            for details.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class SwiftFormerModel(SwiftFormerPreTrainedModel):
    def __init__(self, config: SwiftFormerConfig):
        super().__init__(config)
        self.config = config

        self.patch_embed = SwiftFormerPatchEmbedding(config)
        self.encoder = SwiftFormerEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        pixel_values: Optional[ms.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithNoAttention]:
        r""" """

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        embedding_output = self.patch_embed(pixel_values)
        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return tuple(v for v in encoder_outputs if v is not None)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=encoder_outputs.last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
        )


class SwiftFormerForImageClassification(SwiftFormerPreTrainedModel):
    def __init__(self, config: SwiftFormerConfig) -> None:
        super().__init__(config)

        embed_dims = config.embed_dims

        self.num_labels = config.num_labels
        self.swiftformer = SwiftFormerModel(config)

        # Classifier head
        self.norm = mint.nn.BatchNorm2d(embed_dims[-1], eps=config.batch_norm_eps)
        self.head = mint.nn.Linear(embed_dims[-1], self.num_labels) if self.num_labels > 0 else mint.nn.Identity()
        self.dist_head = mint.nn.Linear(embed_dims[-1], self.num_labels) if self.num_labels > 0 else mint.nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        pixel_values: Optional[ms.Tensor] = None,
        labels: Optional[ms.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutputWithNoAttention]:
        r"""
        labels (`ms.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # run base model
        outputs = self.swiftformer(
            pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs.last_hidden_state if return_dict else outputs[0]

        # run classification head
        sequence_output = self.norm(sequence_output)
        sequence_output = sequence_output.flatten(2).mean(-1)
        cls_out = self.head(sequence_output)
        distillation_out = self.dist_head(sequence_output)
        logits = (cls_out + distillation_out) / 2

        # calculate loss
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

        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )


__all__ = ["SwiftFormerForImageClassification", "SwiftFormerModel", "SwiftFormerPreTrainedModel"]
