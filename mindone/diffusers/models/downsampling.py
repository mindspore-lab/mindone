# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from typing import Optional

import mindspore as ms
from mindspore import nn

from .normalization import LayerNorm


class Downsample2D(nn.Cell):
    """A 2D downsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        padding (`int`, default `1`):
            padding for the convolution.
        name (`str`, default `conv`):
            name of the downsampling 2D layer.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: Optional[int] = None,
        padding: int = 1,
        name: str = "conv",
        kernel_size=3,
        norm_type=None,
        eps=None,
        elementwise_affine=None,
        bias=True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name
        conv_cls = nn.Conv2d

        if norm_type == "ln_norm":
            self.norm = LayerNorm(channels, eps, elementwise_affine)
        elif norm_type == "rms_norm":
            raise NotImplementedError("RMSNorm is not implemented")
        elif norm_type is None:
            self.norm = None
        else:
            raise ValueError(f"unknown norm_type: {norm_type}")

        if use_conv:
            conv = conv_cls(
                self.channels,
                self.out_channels,
                kernel_size=kernel_size,
                stride=stride,
                pad_mode="pad",
                padding=padding,
                has_bias=bias,
            )
            if padding == 0:
                self.pad = nn.Pad(paddings=((0, 0), (0, 0), (0, 1), (0, 1)))
            else:
                self.pad = nn.Identity()
        else:
            assert self.channels == self.out_channels
            conv = nn.AvgPool2d(kernel_size=stride, stride=stride)
            self.pad = nn.Identity()

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if name == "conv":
            self.conv = conv
        elif name == "Conv2d_0":
            self.Conv2d_0 = conv
        else:
            self.conv = conv

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        assert hidden_states.shape[1] == self.channels

        if self.norm is not None:
            hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if self.use_conv and self.padding == 0:
            hidden_states = self.pad(hidden_states)

        assert hidden_states.shape[1] == self.channels

        hidden_states = self.conv(hidden_states)

        return hidden_states
