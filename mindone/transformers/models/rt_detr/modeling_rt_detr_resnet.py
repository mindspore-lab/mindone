# coding=utf-8
# Copyright 2024 Microsoft Research, Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""
PyTorch RTDetr specific ResNet model. The main difference between hugginface ResNet model is that this RTDetrResNet
model forces to use shortcut at the first layer in the resnet-18/34 models.
See https://github.com/lyuwenyu/RT-DETR/blob/5b628eaa0a2fc25bdafec7e6148d5296b144af85/rtdetr_pytorch/src/nn/backbone/
presnet.py#L126 for details.
"""

import math
from typing import Optional

from transformers import RTDetrResNetConfig

import mindspore
from mindspore import Tensor, mint
from mindspore.common.initializer import (
    HeNormal,
    HeUniform,
    One,
    Uniform,
    Zero,
    _calculate_fan_in_and_fan_out,
    initializer,
)

from ...activations import ACT2FN
from ...modeling_outputs import BackboneOutput, BaseModelOutputWithNoAttention
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from ...utils.backbone_utils import BackboneMixin

logger = logging.get_logger(__name__)


# Copied from transformers.models.resnet.modeling_resnet.ResNetConvLayer -> RTDetrResNetConvLayer
class RTDetrResNetConvLayer(mindspore.nn.Cell):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, activation: str = "relu"
    ):
        super().__init__()
        self.convolution = mint.nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=False
        )
        self.normalization = mint.nn.BatchNorm2d(out_channels)
        self.activation = ACT2FN[activation] if activation is not None else mint.nn.Identity()

    def construct(self, input: Tensor) -> Tensor:
        hidden_state = self.convolution(input)
        hidden_state = self.normalization(hidden_state).to(input.dtype)
        hidden_state = self.activation(hidden_state)
        return hidden_state


class RTDetrResNetEmbeddings(mindspore.nn.Cell):
    """
    ResNet Embeddings (stem) composed of a deep aggressive convolution.
    """

    def __init__(self, config: RTDetrResNetConfig):
        super().__init__()
        self.embedder = mindspore.nn.SequentialCell(
            RTDetrResNetConvLayer(
                config.num_channels,
                config.embedding_size // 2,
                kernel_size=3,
                stride=2,
                activation=config.hidden_act,
            ),
            RTDetrResNetConvLayer(
                config.embedding_size // 2,
                config.embedding_size // 2,
                kernel_size=3,
                stride=1,
                activation=config.hidden_act,
            ),
            RTDetrResNetConvLayer(
                config.embedding_size // 2,
                config.embedding_size,
                kernel_size=3,
                stride=1,
                activation=config.hidden_act,
            ),
        )
        self.pooler = mint.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.num_channels = config.num_channels

    def construct(self, pixel_values: Tensor) -> Tensor:
        num_channels = pixel_values.shape[1]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        embedding = self.embedder(pixel_values)
        embedding = self.pooler(embedding.to(mindspore.float32)).to(pixel_values.dtype)
        return embedding


# Copied from transformers.models.resnet.modeling_resnet.ResNetShortCut -> RTDetrResNetChortCut
class RTDetrResNetShortCut(mindspore.nn.Cell):
    """
    ResNet shortcut, used to project the residual features to the correct size. If needed, it is also used to
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


class RTDetrResNetBasicLayer(mindspore.nn.Cell):
    """
    A classic ResNet's residual layer composed by two `3x3` convolutions.
    See https://github.com/lyuwenyu/RT-DETR/blob/5b628eaa0a2fc25bdafec7e6148d5296b144af85/rtdetr_pytorch/src/nn/
    backbone/presnet.py#L34.
    """

    def __init__(
        self,
        config: RTDetrResNetConfig,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        should_apply_shortcut: bool = False,
    ):
        super().__init__()
        if in_channels != out_channels:
            self.shortcut = (
                mindspore.nn.SequentialCell(
                    *[
                        mint.nn.AvgPool2d(2, 2, 0, ceil_mode=True),
                        RTDetrResNetShortCut(in_channels, out_channels, stride=1),
                    ]
                )
                if should_apply_shortcut
                else mint.nn.Identity()
            )
        else:
            self.shortcut = (
                RTDetrResNetShortCut(in_channels, out_channels, stride=stride)
                if should_apply_shortcut
                else mint.nn.Identity()
            )
        self.layer = mindspore.nn.SequentialCell(
            RTDetrResNetConvLayer(in_channels, out_channels, stride=stride),
            RTDetrResNetConvLayer(out_channels, out_channels, activation=None),
        )
        self.activation = ACT2FN[config.hidden_act]

    def construct(self, hidden_state):
        residual = hidden_state
        hidden_state = self.layer(hidden_state)
        residual = self.shortcut(residual)
        hidden_state += residual
        hidden_state = self.activation(hidden_state)
        return hidden_state


class RTDetrResNetBottleNeckLayer(mindspore.nn.Cell):
    """
    A classic RTDetrResNet's bottleneck layer composed by three `3x3` convolutions.

    The first `1x1` convolution reduces the input by a factor of `reduction` in order to make the second `3x3`
    convolution faster. The last `1x1` convolution remaps the reduced features to `out_channels`. If
    `downsample_in_bottleneck` is true, downsample will be in the first layer instead of the second layer.
    """

    def __init__(
        self,
        config: RTDetrResNetConfig,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ):
        super().__init__()
        reduction = 4
        should_apply_shortcut = in_channels != out_channels or stride != 1
        reduces_channels = out_channels // reduction
        if stride == 2:
            self.shortcut = mindspore.nn.SequentialCell(
                *[
                    mint.nn.AvgPool2d(2, 2, 0, ceil_mode=True),
                    RTDetrResNetShortCut(in_channels, out_channels, stride=1)
                    if should_apply_shortcut
                    else mint.nn.Identity(),
                ]
            )
        else:
            self.shortcut = (
                RTDetrResNetShortCut(in_channels, out_channels, stride=stride)
                if should_apply_shortcut
                else mint.nn.Identity()
            )
        self.layer = mindspore.nn.SequentialCell(
            RTDetrResNetConvLayer(
                in_channels, reduces_channels, kernel_size=1, stride=stride if config.downsample_in_bottleneck else 1
            ),
            RTDetrResNetConvLayer(
                reduces_channels, reduces_channels, stride=stride if not config.downsample_in_bottleneck else 1
            ),
            RTDetrResNetConvLayer(reduces_channels, out_channels, kernel_size=1, activation=None),
        )
        self.activation = ACT2FN[config.hidden_act]

    def construct(self, hidden_state):
        residual = hidden_state
        hidden_state = self.layer(hidden_state)
        residual = self.shortcut(residual)
        hidden_state += residual
        hidden_state = self.activation(hidden_state)
        return hidden_state


class RTDetrResNetStage(mindspore.nn.Cell):
    """
    A RTDetrResNet stage composed by stacked layers.
    """

    def __init__(
        self,
        config: RTDetrResNetConfig,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        depth: int = 2,
    ):
        super().__init__()

        layer = RTDetrResNetBottleNeckLayer if config.layer_type == "bottleneck" else RTDetrResNetBasicLayer

        if config.layer_type == "bottleneck":
            first_layer = layer(
                config,
                in_channels,
                out_channels,
                stride=stride,
            )
        else:
            first_layer = layer(config, in_channels, out_channels, stride=stride, should_apply_shortcut=True)
        self.layers = mindspore.nn.SequentialCell(
            first_layer, *[layer(config, out_channels, out_channels) for _ in range(depth - 1)]
        )

    def construct(self, input: Tensor) -> Tensor:
        hidden_state = input
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


# Copied from transformers.models.resnet.modeling_resnet.ResNetEncoder with ResNet->RTDetrResNet
class RTDetrResNetEncoder(mindspore.nn.Cell):
    def __init__(self, config: RTDetrResNetConfig):
        super().__init__()
        self.stages = mindspore.nn.CellList([])
        # based on `downsample_in_first_stage` the first layer of the first stage may or may not downsample the input
        self.stages.append(
            RTDetrResNetStage(
                config,
                config.embedding_size,
                config.hidden_sizes[0],
                stride=2 if config.downsample_in_first_stage else 1,
                depth=config.depths[0],
            )
        )
        in_out_channels = zip(config.hidden_sizes, config.hidden_sizes[1:])
        for (in_channels, out_channels), depth in zip(in_out_channels, config.depths[1:]):
            self.stages.append(RTDetrResNetStage(config, in_channels, out_channels, depth=depth))

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

        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_state,
            hidden_states=hidden_states,
        )


# Copied from transformers.models.resnet.modeling_resnet.ResNetPreTrainedModel with ResNet->RTDetrResNet
class RTDetrResNetPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = RTDetrResNetConfig
    base_model_prefix = "resnet"
    main_input_name = "pixel_values"
    _no_split_modules = ["RTDetrResNetConvLayer", "RTDetrResNetShortCut"]

    def _init_weights(self, cell):
        """Initializes weights using MindSpore initializers."""

        if isinstance(cell, mint.nn.Conv2d):
            cell.weight.set_data(
                initializer(HeNormal(mode="fan_out", nonlinearity="relu"), cell.weight.shape, cell.weight.dtype)
            )

        elif isinstance(cell, mint.nn.Linear):
            cell.weight.set_data(
                initializer(HeUniform(negative_slope=math.sqrt(5)), cell.weight.shape, cell.weight.dtype)
            )
            if cell.bias is not None:
                fan_in, _ = _calculate_fan_in_and_fan_out(cell.weight.shape)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                cell.bias.set_data(initializer(Uniform(scale=bound), cell.bias.shape, cell.bias.dtype))

        elif isinstance(cell, (mint.nn.BatchNorm2d, mint.nn.GroupNorm)):
            cell.weight.set_data(initializer(One(), cell.weight.shape, cell.weight.dtype))
            cell.bias.set_data(initializer(Zero(), cell.bias.shape, cell.bias.dtype))


class RTDetrResNetBackbone(RTDetrResNetPreTrainedModel, BackboneMixin):
    def __init__(self, config):
        super().__init__(config)
        super()._init_backbone(config)

        self.num_features = [config.embedding_size] + config.hidden_sizes
        self.embedder = RTDetrResNetEmbeddings(config)
        self.encoder = RTDetrResNetEncoder(config)

        # initialize weights and apply final processing
        self.post_init()

    def construct(
        self, pixel_values: Tensor, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None
    ) -> BackboneOutput:
        """
        Returns:

        Examples:

        ```python
        >>> from transformers import RTDetrResNetConfig, RTDetrResNetBackbone
        >>> import mindspore as ms

        >>> config = RTDetrResNetConfig()
        >>> model = RTDetrResNetBackbone(config)

        >>> pixel_values = mindspore.ops.randn(1, 3, 224, 224)

        >>> outputs = model(pixel_values)

        >>> feature_maps = outputs.feature_maps
        >>> list(feature_maps[-1].shape)
        [1, 2048, 7, 7]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        embedding_output = self.embedder(pixel_values)

        outputs = self.encoder(embedding_output, output_hidden_states=True, return_dict=True)

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


__all__ = [
    "RTDetrResNetBackbone",
    "RTDetrResNetPreTrainedModel",
]
