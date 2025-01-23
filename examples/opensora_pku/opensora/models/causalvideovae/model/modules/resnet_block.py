import math

import mindspore as ms
from mindspore import nn
from mindspore.common.initializer import HeUniform, Uniform

try:
    from opensora.npu_config import npu_config
except ImportError:
    npu_config = None
from .conv import CausalConv3d
from .normalize import Normalize
from .ops import nonlinearity, video_to_image


class ResnetBlock3D(nn.Cell):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        norm_type,
        dtype=ms.float32,
        upcast_sigmoid=False,
    ):
        super().__init__()
        self.dtype = dtype
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut
        self.upcast_sigmoid = upcast_sigmoid

        # FIXME: GroupNorm precision mismatch with PT.
        self.norm1 = Normalize(in_channels, norm_type=norm_type)
        self.conv1 = CausalConv3d(in_channels, out_channels, 3, padding=1)
        self.norm2 = Normalize(out_channels, norm_type=norm_type)
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = CausalConv3d(out_channels, out_channels, 3, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = CausalConv3d(in_channels, out_channels, 3, padding=1)
            else:
                self.nin_shortcut = CausalConv3d(in_channels, out_channels, 1, padding=0)

    def construct(self, x):
        h = x
        h = npu_config.run_group_norm(self.norm1, h)
        h = nonlinearity(h, self.upcast_sigmoid)
        h = self.conv1(h)
        h = npu_config.run_group_norm(self.norm2, h)
        h = nonlinearity(h, self.upcast_sigmoid)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


# pku opensora v1.1
class ResnetBlock2D(nn.Cell):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        norm_type,
        dropout,
        dtype=ms.float32,
        upcast_sigmoid=False,
    ):
        super().__init__()
        self.dtype = dtype
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.upcast_sigmoid = upcast_sigmoid

        self.norm1 = Normalize(in_channels, norm_type=norm_type)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            pad_mode="pad",
            padding=1,
            has_bias=True,
            weight_init=HeUniform(negative_slope=math.sqrt(5)),
            bias_init=Uniform(scale=1 / math.sqrt(in_channels * 3 * 3)),
        ).to_float(dtype)
        self.norm2 = Normalize(out_channels, norm_type=norm_type)
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            pad_mode="pad",
            padding=1,
            has_bias=True,
            weight_init=HeUniform(negative_slope=math.sqrt(5)),
            bias_init=Uniform(scale=1 / math.sqrt(out_channels * 3 * 3)),
        ).to_float(dtype)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    pad_mode="pad",
                    padding=1,
                    has_bias=True,
                    weight_init=HeUniform(negative_slope=math.sqrt(5)),
                    bias_init=Uniform(scale=1 / math.sqrt(in_channels * 3 * 3)),
                ).to_float(dtype)
            else:
                self.nin_shortcut = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    pad_mode="valid",
                    has_bias=True,
                    weight_init=HeUniform(negative_slope=math.sqrt(5)),
                    bias_init=Uniform(scale=1 / math.sqrt(in_channels * 3 * 3)),
                ).to_float(dtype)

    @video_to_image
    def construct(self, x):
        h = x

        h = npu_config.run_group_norm(self.norm1, h)
        h = nonlinearity(h, self.upcast_sigmoid)
        h = self.conv1(h)

        h = npu_config.run_group_norm(self.norm2, h)
        h = nonlinearity(h, self.upcast_sigmoid)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        x = x + h
        return x
