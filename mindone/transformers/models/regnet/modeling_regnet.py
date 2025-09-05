# coding=utf-8
# Copyright 2022 Meta Platforms, Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""MindSpore RegNet model."""

from typing import Optional

from transformers.models.regnet.configuration_regnet import RegNetConfig

import mindspore as ms
from mindspore import Tensor, mint, nn
from mindspore.mint.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithNoAttention,
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention,
)
from ...modeling_utils import PreTrainedModel
from ...utils import logging

logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "RegNetConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "facebook/regnet-y-040"
_EXPECTED_OUTPUT_SHAPE = [1, 1088, 7, 7]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "facebook/regnet-y-040"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"


class RegNetConvLayer(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        activation: Optional[str] = "relu",
    ):
        super().__init__()
        self.convolution = mint.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=groups,
            bias=False,
        )
        self.normalization = mint.nn.BatchNorm2d(out_channels)
        self.activation = ACT2FN[activation] if activation is not None else mint.nn.Identity()

    def construct(self, hidden_state):
        hidden_state = self.convolution(hidden_state)
        hidden_state = self.normalization(hidden_state)
        hidden_state = self.activation(hidden_state)
        return hidden_state


class RegNetEmbeddings(nn.Cell):
    """
    RegNet Embeddings (stem) composed of a single aggressive convolution.
    """

    def __init__(self, config: RegNetConfig):
        super().__init__()
        self.embedder = RegNetConvLayer(
            config.num_channels, config.embedding_size, kernel_size=3, stride=2, activation=config.hidden_act
        )
        self.num_channels = config.num_channels

    def construct(self, pixel_values):
        num_channels = pixel_values.shape[1]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        hidden_state = self.embedder(pixel_values)
        return hidden_state


# Copied from transformers.models.resnet.modeling_resnet.ResNetShortCut with ResNet->RegNet
class RegNetShortCut(nn.Cell):
    """
    RegNet shortcut, used to project the residual features to the correct size. If needed, it is also used to
    downsample the input using `stride=2`.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.convolution = mint.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.normalization = mint.nn.BatchNorm2d(out_channels)

    def construct(self, input: Tensor) -> Tensor:
        hidden_state = self.convolution(input)
        hidden_state = self.normalization(hidden_state)
        return hidden_state


class RegNetSELayer(nn.Cell):
    """
    Squeeze and Excitation layer (SE) proposed in [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507).
    """

    def __init__(self, in_channels: int, reduced_channels: int):
        super().__init__()

        self.pooler = mint.nn.AdaptiveAvgPool2d((1, 1))
        self.attention = nn.SequentialCell(
            mint.nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            mint.nn.ReLU(),
            mint.nn.Conv2d(reduced_channels, in_channels, kernel_size=1),
            mint.nn.Sigmoid(),
        )

    def construct(self, hidden_state):
        # b c h w -> b c 1 1
        pooled = self.pooler(hidden_state)
        attention = self.attention(pooled)
        hidden_state = hidden_state * attention
        return hidden_state


class RegNetXLayer(nn.Cell):
    """
    RegNet's layer composed by three `3x3` convolutions, same as a ResNet bottleneck layer with reduction = 1.
    """

    def __init__(self, config: RegNetConfig, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        should_apply_shortcut = in_channels != out_channels or stride != 1
        groups = max(1, out_channels // config.groups_width)
        self.shortcut = (
            RegNetShortCut(in_channels, out_channels, stride=stride) if should_apply_shortcut else mint.nn.Identity()
        )
        self.layer = nn.SequentialCell(
            RegNetConvLayer(in_channels, out_channels, kernel_size=1, activation=config.hidden_act),
            RegNetConvLayer(out_channels, out_channels, stride=stride, groups=groups, activation=config.hidden_act),
            RegNetConvLayer(out_channels, out_channels, kernel_size=1, activation=None),
        )
        self.activation = ACT2FN[config.hidden_act]

    def construct(self, hidden_state):
        residual = hidden_state
        hidden_state = self.layer(hidden_state)
        residual = self.shortcut(residual)
        hidden_state += residual
        hidden_state = self.activation(hidden_state)
        return hidden_state


class RegNetYLayer(nn.Cell):
    """
    RegNet's Y layer: an X layer with Squeeze and Excitation.
    """

    def __init__(self, config: RegNetConfig, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        should_apply_shortcut = in_channels != out_channels or stride != 1
        groups = max(1, out_channels // config.groups_width)
        self.shortcut = (
            RegNetShortCut(in_channels, out_channels, stride=stride) if should_apply_shortcut else mint.nn.Identity()
        )
        self.layer = nn.SequentialCell(
            RegNetConvLayer(in_channels, out_channels, kernel_size=1, activation=config.hidden_act),
            RegNetConvLayer(out_channels, out_channels, stride=stride, groups=groups, activation=config.hidden_act),
            RegNetSELayer(out_channels, reduced_channels=int(round(in_channels / 4))),
            RegNetConvLayer(out_channels, out_channels, kernel_size=1, activation=None),
        )
        self.activation = ACT2FN[config.hidden_act]

    def construct(self, hidden_state):
        residual = hidden_state
        hidden_state = self.layer(hidden_state)
        residual = self.shortcut(residual)
        hidden_state += residual
        hidden_state = self.activation(hidden_state)
        return hidden_state


class RegNetStage(nn.Cell):
    """
    A RegNet stage composed by stacked layers.
    """

    def __init__(
        self,
        config: RegNetConfig,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        depth: int = 2,
    ):
        super().__init__()

        layer = RegNetXLayer if config.layer_type == "x" else RegNetYLayer

        self.layers = nn.SequentialCell(
            # downsampling is done in the first layer with stride of 2
            layer(
                config,
                in_channels,
                out_channels,
                stride=stride,
            ),
            *[layer(config, out_channels, out_channels) for _ in range(depth - 1)],
        )

    def construct(self, hidden_state):
        hidden_state = self.layers(hidden_state)
        return hidden_state


class RegNetEncoder(nn.Cell):
    def __init__(self, config: RegNetConfig):
        super().__init__()
        self.stages = []
        # based on `downsample_in_first_stage`, the first layer of the first stage may or may not downsample the input
        self.stages.append(
            RegNetStage(
                config,
                config.embedding_size,
                config.hidden_sizes[0],
                stride=2 if config.downsample_in_first_stage else 1,
                depth=config.depths[0],
            )
        )
        in_out_channels = zip(config.hidden_sizes, config.hidden_sizes[1:])
        for (in_channels, out_channels), depth in zip(in_out_channels, config.depths[1:]):
            self.stages.append(RegNetStage(config, in_channels, out_channels, depth=depth))
        self.stages = nn.CellList(self.stages)

    def construct(
        self, hidden_state: Tensor, output_hidden_states: bool = False, return_dict: bool = True
    ) -> BaseModelOutputWithNoAttention:
        hidden_states = () if output_hidden_states else None

        for stage_module in self.stages:
            if output_hidden_states:
                hidden_states = hidden_states + (hidden_state,)

            hidden_state = stage_module(hidden_state)

        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state,)

        if not return_dict:
            return tuple(v for v in [hidden_state, hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(last_hidden_state=hidden_state, hidden_states=hidden_states)


class RegNetPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = RegNetConfig
    base_model_prefix = "regnet"
    main_input_name = "pixel_values"
    _no_split_modules = ["RegNetYLayer"]

    # Copied from transformers.models.resnet.modeling_resnet.ResNetPreTrainedModel._init_weights
    def _init_weights(self, module):
        pass


REGNET_START_DOCSTRING = r"""
    This model is a MindSpore
    [ms.nn.Cell](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Cell.html) subclass. Use it
    as a regular MindSpore Cell and refer to the MindSpore documentation for all matters related to general usage and
    behavior.

    Parameters:
        config ([`RegNetConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

REGNET_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`ms.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`ConvNextImageProcessor.__call__`] for details.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""


# Copied from transformers.models.resnet.modeling_resnet.ResNetModel with RESNET->REGNET,ResNet->RegNet
class RegNetModel(RegNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embedder = RegNetEmbeddings(config)
        self.encoder = RegNetEncoder(config)
        self.pooler = mint.nn.AdaptiveAvgPool2d((1, 1))
        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self, pixel_values: Tensor, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None
    ) -> BaseModelOutputWithPoolingAndNoAttention:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        embedding_output = self.embedder(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output, output_hidden_states=output_hidden_states, return_dict=return_dict
        )

        last_hidden_state = encoder_outputs[0]

        pooled_output = self.pooler(last_hidden_state)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )


# Copied from transformers.models.resnet.modeling_resnet.ResNetForImageClassification with RESNET->REGNET,ResNet->RegNet,resnet->regnet
class RegNetForImageClassification(RegNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.regnet = RegNetModel(config)
        # classification head
        self.classifier = nn.SequentialCell(
            mint.nn.Flatten(),
            mint.nn.Linear(config.hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else mint.nn.Identity(),
        )
        # initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        pixel_values: Optional[ms.Tensor] = None,
        labels: Optional[ms.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> ImageClassifierOutputWithNoAttention:
        r"""
        labels (`ms.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.regnet(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

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
            return (loss,) + output if loss is not None else output

        return ImageClassifierOutputWithNoAttention(loss=loss, logits=logits, hidden_states=outputs.hidden_states)


__all__ = ["RegNetForImageClassification", "RegNetModel", "RegNetPreTrainedModel"]
