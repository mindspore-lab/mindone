# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Optional, Tuple

import mindspore as ms
from mindspore import Parameter, mint, nn
from mindspore.common.initializer import Constant, One, TruncatedNormal, Zero, initializer

from mindone.utils import WeightNorm


class ConvNeXtBlock(nn.Cell):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
            None means non-conditional LayerNorm. Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: float,
        condition_dim: Optional[int] = None,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=7,
            padding=3,
            group=dim,
            pad_mode="pad",
            has_bias=True,
        )  # depthwise conv
        self.adanorm = condition_dim is not None
        if condition_dim:
            self.norm = AdaLayerNorm(condition_dim, dim, eps=1e-6)
        else:
            self.norm = mint.nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = mint.nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = mint.nn.Linear(intermediate_dim, dim)
        self.gamma = (
            Parameter(layer_scale_init_value * mint.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def construct(self, x: ms.Tensor, cond_embedding_id: Optional[ms.Tensor] = None) -> ms.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        if self.adanorm:
            assert cond_embedding_id is not None
            x = self.norm(x, cond_embedding_id)
        else:
            x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x


class AdaLayerNorm(nn.Cell):
    """
    Adaptive Layer Normalization module with learnable embeddings per `num_embeddings` classes

    Args:
        condition_dim (int): Dimension of the condition.
        embedding_dim (int): Dimension of the embeddings.
    """

    def __init__(self, condition_dim: int, embedding_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = embedding_dim
        self.scale = mint.nn.Linear(condition_dim, embedding_dim)
        self.shift = mint.nn.Linear(condition_dim, embedding_dim)
        self.scale.weight.set_data(
            initializer(
                One(),
                shape=self.scale.weight.shape,
                dtype=self.scale.weight.dtype,
            )
        )
        self.shift.weight.set_data(
            initializer(
                Zero(),
                shape=self.shift.weight.shape,
                dtype=self.shift.weight.dtype,
            )
        )

    def construct(self, x: ms.Tensor, cond_embedding: ms.Tensor) -> ms.Tensor:
        scale = self.scale(cond_embedding)
        shift = self.shift(cond_embedding)
        x = mint.nn.functional.layer_norm(x, (self.dim,), eps=self.eps)
        x = x * scale.unsqueeze(1) + shift.unsqueeze(1)
        return x


class ResBlock1(nn.Cell):
    """
    ResBlock adapted from HiFi-GAN V1 (https://github.com/jik876/hifi-gan) with dilated 1D convolutions,
    but without upsampling layers.

    Args:
        dim (int): Number of input channels.
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
        dilation (tuple[int], optional): Dilation factors for the dilated convolutions.
            Defaults to (1, 3, 5).
        lrelu_slope (float, optional): Negative slope of the LeakyReLU activation function.
            Defaults to 0.1.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        dilation: Tuple[int, int, int] = (1, 3, 5),
        lrelu_slope: float = 0.1,
        layer_scale_init_value: Optional[float] = None,
    ):
        super().__init__()
        self.lrelu_slope = lrelu_slope
        self.convs1 = nn.CellList(
            [
                WeightNorm(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=self.get_padding(kernel_size, dilation[0]),
                        pad_mode="pad",
                        has_bias=True,
                    )
                ),
                WeightNorm(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=self.get_padding(kernel_size, dilation[1]),
                        pad_mode="pad",
                        has_bias=True,
                    )
                ),
                WeightNorm(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=self.get_padding(kernel_size, dilation[2]),
                        pad_mode="pad",
                        has_bias=True,
                    )
                ),
            ]
        )

        self.convs2 = nn.CellList(
            [
                WeightNorm(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=self.get_padding(kernel_size, 1),
                        pad_mode="pad",
                        has_bias=True,
                    )
                ),
                WeightNorm(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=self.get_padding(kernel_size, 1),
                        pad_mode="pad",
                        has_bias=True,
                    )
                ),
                WeightNorm(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=self.get_padding(kernel_size, 1),
                        pad_mode="pad",
                        has_bias=True,
                    )
                ),
            ]
        )

        self.gamma = [
            (
                Parameter(layer_scale_init_value * mint.ones((dim, 1)), requires_grad=True)
                if layer_scale_init_value is not None
                else None
            ),
            (
                Parameter(layer_scale_init_value * mint.ones((dim, 1)), requires_grad=True)
                if layer_scale_init_value is not None
                else None
            ),
            (
                Parameter(layer_scale_init_value * mint.ones((dim, 1)), requires_grad=True)
                if layer_scale_init_value is not None
                else None
            ),
        ]

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        for c1, c2, gamma in zip(self.convs1, self.convs2, self.gamma):
            xt = mint.nn.functional.leaky_relu(x, negative_slope=self.lrelu_slope)
            xt = c1(xt)
            xt = mint.nn.functional.leaky_relu(xt, negative_slope=self.lrelu_slope)
            xt = c2(xt)
            if gamma is not None:
                xt = gamma * xt
            x = xt + x
        return x

    @staticmethod
    def get_padding(kernel_size: int, dilation: int = 1) -> int:
        return int((kernel_size * dilation - dilation) / 2)


class Backbone(nn.Cell):
    """Base class for the generator's backbone. It preserves the same temporal resolution across all layers."""

    def construct(self, x: ms.Tensor, **kwargs) -> ms.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, C, L), where B is the batch size,
                        C denotes output features, and L is the sequence length.

        Returns:
            Tensor: Output of shape (B, L, H), where B is the batch size, L is the sequence length,
                    and H denotes the model dimension.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class VocosBackbone(Backbone):
    """
    Vocos backbone module built with ConvNeXt blocks. Supports additional conditioning with Adaptive Layer Normalization

    Args:
        input_channels (int): Number of input features channels.
        dim (int): Hidden dimension of the model.
        intermediate_dim (int): Intermediate dimension used in ConvNeXtBlock.
        num_layers (int): Number of ConvNeXtBlock layers.
        layer_scale_init_value (float, optional): Initial value for layer scaling. Defaults to `1 / num_layers`.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
                                                None means non-conditional model. Defaults to None.
    """

    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        layer_scale_init_value: Optional[float] = None,
        condition_dim: Optional[int] = None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embed = nn.Conv1d(
            input_channels,
            dim,
            kernel_size=7,
            padding=3,
            pad_mode="pad",
            has_bias=True,
        )
        self.adanorm = condition_dim is not None
        if condition_dim:
            self.norm = AdaLayerNorm(condition_dim, dim, eps=1e-6)
        else:
            self.norm = mint.nn.LayerNorm(dim, eps=1e-6)
        layer_scale_init_value = layer_scale_init_value or 1 / num_layers
        self.convnext = nn.CellList(
            [
                ConvNeXtBlock(
                    dim=dim,
                    intermediate_dim=intermediate_dim,
                    layer_scale_init_value=layer_scale_init_value,
                    condition_dim=condition_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = mint.nn.LayerNorm(dim, eps=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, mint.nn.Linear)):
            m.weight.set_data(initializer(TruncatedNormal(sigma=0.02), shape=m.weight.shape, dtype=m.weight.dtype))
            m.bias.set_data(initializer(Constant(0), shape=m.bias.shape, dtype=m.bias.dtype))

    def construct(self, x: ms.Tensor, condition: ms.Tensor = None) -> ms.Tensor:
        x = self.embed(x)
        if self.adanorm:
            assert condition is not None
            x = self.norm(x.transpose(1, 2), condition)
        else:
            x = self.norm(x.transpose(1, 2))
        x = x.transpose(1, 2)
        for conv_block in self.convnext:
            x = conv_block(x, condition)
        x = self.final_layer_norm(x.transpose(1, 2))
        return x


class VocosResNetBackbone(Backbone):
    """
    Vocos backbone module built with ResBlocks.

    Args:
        input_channels (int): Number of input features channels.
        dim (int): Hidden dimension of the model.
        num_blocks (int): Number of ResBlock1 blocks.
        layer_scale_init_value (float, optional): Initial value for layer scaling. Defaults to None.
    """

    def __init__(
        self,
        input_channels,
        dim,
        num_blocks,
        layer_scale_init_value=None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embed = WeightNorm(nn.Conv1d(input_channels, dim, kernel_size=3, padding=1, pad_mode="pad", has_bias=True))
        layer_scale_init_value = layer_scale_init_value or 1 / num_blocks / 3
        self.resnet = nn.SequentialCell(
            *[ResBlock1(dim=dim, layer_scale_init_value=layer_scale_init_value) for _ in range(num_blocks)]
        )

    def construct(self, x: ms.Tensor, **kwargs) -> ms.Tensor:
        x = self.embed(x)
        x = self.resnet(x)
        x = x.transpose(1, 2)
        return x
