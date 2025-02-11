import math
from collections import deque
from typing import Tuple, Union

from opensora.npu_config import npu_config

import mindspore as ms
from mindspore import mint, nn, ops
from mindspore.common.initializer import HeUniform, Uniform

from .conv import CausalConv3d
from .ops import cast_tuple, video_to_image


class ResizeNearestNeighbor(nn.Cell):
    def construct(self, x, size, scale_factor=None):
        return ops.interpolate(x, size=size, scale_factor=scale_factor, mode="nearest")


class Upsample(nn.Cell):
    def __init__(self, in_channels, out_channels, with_conv=True, dtype=ms.float32):
        super().__init__()
        self.dtype = dtype
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                pad_mode="pad",
                padding=1,
                has_bias=True,
                weight_init=HeUniform(negative_slope=math.sqrt(5)),
                bias_init=Uniform(scale=1 / math.sqrt(in_channels * 3 * 3)),
            ).to_float(self.dtype)
        self.resize = ResizeNearestNeighbor()

    @video_to_image
    def construct(self, x):
        in_shape = x.shape[-2:]
        out_shape = tuple(2 * x for x in in_shape)
        x = npu_config.run_interpolate(self.resize, x, size=out_shape)
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Cell):
    def __init__(self, in_channels, out_channels, undown=False, dtype=ms.float32):
        super().__init__()
        self.with_conv = True
        self.undown = undown
        self.dtype = dtype
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            if self.undown:
                self.conv = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    pad_mode="pad",
                    has_bias=True,
                    weight_init=HeUniform(negative_slope=math.sqrt(5)),
                    bias_init=Uniform(scale=1 / math.sqrt(in_channels * 3 * 3)),
                ).to_float(self.dtype)
            else:
                self.conv = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=0,
                    pad_mode="pad",
                    has_bias=True,
                    weight_init=HeUniform(negative_slope=math.sqrt(5)),
                    bias_init=Uniform(scale=1 / math.sqrt(in_channels * 3 * 3)),
                ).to_float(self.dtype)

    @video_to_image
    def construct(self, x):
        if self.with_conv:
            if self.undown:
                x = self.conv(x)
            else:
                pad = (0, 1, 0, 1)
                x = mint.nn.functional.pad(x, pad, mode="constant", value=0)
                x = self.conv(x)
        else:
            x = npu_config.run_pool_2d(
                mint.nn.functional.avg_pool2d, kernel_size=2, stride=2
            )  # avgpool does not support bf16, but only fp32 and fp16
        return x


class SpatialDownsample2x(nn.Cell):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int]] = (3, 3),
        stride: Union[int, Tuple[int]] = (2, 2),
        dtype=ms.float32,
    ):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 2)
        stride = cast_tuple(stride, 2)
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.kernel_size = kernel_size
        # TODO: no need to use CausalConv3d, can reshape to spatial (bt, c, h, w) and use conv 2d
        self.conv = CausalConv3d(
            self.chan_in,
            self.chan_out,
            (1,) + self.kernel_size,
            stride=(1,) + stride,
            padding=0,
        )

        # no asymmetric padding, must do it ourselves
        # order (width_pad, width_pad, height_pad, height_pad, time_pad, 0)
        # self.padding = (0,1,0,1,0,0) # not compatible for ms2.2
        self.pad = ops.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 1), (0, 1)))

    def construct(self, x):
        # x shape: (b c t h w)
        pad = (0, 1, 0, 1, 0, 0)
        x = mint.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class SpatialUpsample2x(nn.Cell):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int]] = (3, 3),
        stride: Union[int, Tuple[int]] = (1, 1),
        dtype=ms.float32,
    ):
        super().__init__()
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.kernel_size = kernel_size
        self.conv = CausalConv3d(
            self.chan_in,
            self.chan_out,
            (1,) + self.kernel_size,
            stride=(1,) + stride,
            padding=1,
        )

    def construct(self, x):
        b, c, t, h, w = x.shape

        # x = rearrange(x, "b c t h w -> b (c t) h w")
        x = ops.reshape(x, (b, c * t, h, w))

        hw_in = x.shape[-2:]
        scale_factor = 2
        hw_out = tuple(scale_factor * s_ for s_ in hw_in)
        x = ops.ResizeNearestNeighbor(hw_out)(x)

        # x = ops.interpolate(x, scale_factor=(2.,2.), mode="nearest") # 4D not supported
        # x = rearrange(x, "b (c t) h w -> b c t h w", t=t)
        x = ops.reshape(x, (b, c, t, h * scale_factor, w * scale_factor))

        x = self.conv(x)
        return x


class TimeDownsample2x(nn.Cell):
    def __init__(
        self,
        kernel_size: int = 3,
        replace_avgpool3d: bool = True,  # FIXME: currently, ms+910b does not support nn.AvgPool3d
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.replace_avgpool3d = replace_avgpool3d
        if not replace_avgpool3d:
            self.conv = nn.AvgPool3d((kernel_size, 1, 1), stride=(2, 1, 1))
        else:
            self.conv = nn.AvgPool2d((kernel_size, 1), stride=(2, 1))
        # print('D--: replace avgpool3d', replace_avgpool3d)
        self.time_pad = self.kernel_size - 1

    def construct(self, x):
        first_frame = x[:, :, :1, :, :]
        # first_frame_pad = ops.repeat_interleave(first_frame, self.time_pad, axis=2)
        first_frame_pad = mint.cat([first_frame] * self.time_pad, dim=2)
        x = mint.cat((first_frame_pad, x), dim=2)

        if not self.replace_avgpool3d:
            return self.conv(x)
        else:
            # FIXME: only work when h, w stride is 1
            b, c, t, h, w = x.shape
            x = ops.reshape(x, (b, c, t, h * w))
            x = self.conv(x)
            x = ops.reshape(x, (b, c, -1, h, w))
            return x


class TimeUpsample2x(nn.Cell):
    def __init__(self, exclude_first_frame=True):
        super().__init__()
        self.exclude_first_frame = exclude_first_frame

    def construct(self, x):
        if x.shape[2] > 1:
            if self.exclude_first_frame:
                x, x_ = x[:, :, :1], x[:, :, 1:]
                # FIXME: ms2.2.10 cannot support trilinear on 910b
                x_ = ops.interpolate(x_, scale_factor=(2.0, 1.0, 1.0), mode="trilinear")
                x = mint.cat([x, x_], dim=2)
            else:
                x = ops.interpolate(x, scale_factor=(2.0, 1.0, 1.0), mode="trilinear")

        return x


class TimeDownsampleRes2x(nn.Cell):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: int = 3,
        mix_factor: float = 2.0,
        replace_avgpool3d: bool = True,  # FIXME: currently, ms+910b does not support nn.AvgPool3d
    ):
        super().__init__()
        self.kernel_size = cast_tuple(kernel_size, 3)
        self.replace_avgpool3d = replace_avgpool3d
        if not replace_avgpool3d:
            self.avg_pool = nn.AvgPool3d((kernel_size, 1, 1), stride=(2, 1, 1))
        else:
            self.avg_pool = nn.AvgPool2d((kernel_size, 1), stride=(2, 1))
        self.time_pad = self.kernel_size[0] - 1

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=(2, 1, 1),
            pad_mode="pad",
            padding=(0, 0, 1, 1, 1, 1),
            has_bias=True,
        )

        self.mix_factor = ms.Parameter(ms.Tensor([mix_factor]), requires_grad=True)

    def construct(self, x):
        alpha = mint.sigmoid(self.mix_factor)

        first_frame = x[:, :, :1, :, :]
        # first_frame_pad = ops.repeat_interleave(first_frame, self.time_pad, axis=2)
        first_frame_pad = mint.cat([first_frame] * self.time_pad, dim=2)
        x = mint.cat((first_frame_pad, x), dim=2)
        if npu_config is not None and npu_config.on_npu:
            x_dtype = x.dtype
            conv_out = npu_config.run_conv3d(self.conv, x, x_dtype)
        else:
            conv_out = self.conv(x)

        # avg pool
        if not self.replace_avgpool3d:
            pool_out = self.avg_pool(x)
        else:
            # FIXME: only work when h, w stride is 1
            b, c, t, h, w = x.shape
            x = ops.reshape(x, (b, c, t, h * w))
            x = self.avg_pool(x)
            pool_out = ops.reshape(x, (b, c, -1, h, w))

        return alpha * pool_out + (1 - alpha) * conv_out


class TimeUpsampleRes2x(nn.Cell):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: int = 3,
        mix_factor: float = 2.0,
    ):
        super().__init__()
        self.conv = CausalConv3d(in_channels, out_channels, kernel_size, padding=1)
        self.mix_factor = ms.Parameter(ms.Tensor([mix_factor]), requires_grad=True)
        self.interpolate = TrilinearInterpolate()

    def construct(self, x):
        alpha = mint.sigmoid(self.mix_factor)
        if x.shape[2] > 1:
            x, x_ = x[:, :, :1], x[:, :, 1:]
            ori_dtype = x.dtype
            # FIXME: ms2.2.10 cannot support trilinear on 910b
            x_ = self.interpolate(x_, scale_factor=(2.0, 1.0, 1.0)).to(ori_dtype)
            x = mint.cat([x, x_], dim=2)

        return alpha * x + (1 - alpha) * self.conv(x)


class TrilinearInterpolate(nn.Cell):
    def construct(self, x, scale_factor, size=None):
        return ops.interpolate(x, scale_factor=scale_factor, size=size, mode="trilinear")


class Spatial2xTime2x3DUpsample(nn.Cell):
    def __init__(
        self,
        in_channels,
        out_channels,
        dtype=ms.float32,
        t_interpolation="trilinear",
        enable_cached=False,
    ):
        super().__init__()
        self.dtype = dtype
        self.t_interpolation = t_interpolation
        assert self.t_interpolation == "trilinear", "only support trilinear interpolation now"
        self.conv = CausalConv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.interpolate = TrilinearInterpolate()
        self.enable_cached = enable_cached
        self.causal_cached = deque()

    def construct(self, x):
        if x.shape[2] > 1 or len(self.causal_cached) > 0:
            if self.enable_cached and len(self.causal_cached) > 0:
                x = mint.cat([self.causal_cached.popleft(), x], dim=2)
                self.causal_cached.append(x[:, :, -2:-1].copy())
                x = npu_config.run_interpolate(self.interpolate, x, scale_factor=(2.0, 1.0, 1.0))
                x = x[:, :, 2:]
                x = npu_config.run_interpolate(self.interpolate, x, scale_factor=(1.0, 2.0, 2.0))
            else:
                if self.enable_cached:
                    self.causal_cached.append(x[:, :, -1:].copy())
                x, x_ = x[:, :, :1], x[:, :, 1:]
                x_ = npu_config.run_interpolate(self.interpolate, x_, scale_factor=(2.0, 1.0, 1.0))
                x_ = npu_config.run_interpolate(self.interpolate, x_, scale_factor=(1.0, 2.0, 2.0))
                x = npu_config.run_interpolate(self.interpolate, x, scale_factor=(1.0, 2.0, 2.0))
                x = mint.cat([x, x_], dim=2)
        else:
            if self.enable_cached:
                self.causal_cached.append(x[:, :, -1:].copy())
            x = npu_config.run_interpolate(self.interpolate, x, scale_factor=(1.0, 2.0, 2.0))
        return self.conv(x)


class Spatial2xTime2x3DDownsample(nn.Cell):
    def __init__(self, in_channels, out_channels, dtype=ms.float32):
        super().__init__()
        self.dtype = dtype
        self.conv = CausalConv3d(in_channels, out_channels, kernel_size=3, padding=0, stride=2)

    def construct(self, x):
        pad = (0, 1, 0, 1, 0, 0)
        x = mint.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x
