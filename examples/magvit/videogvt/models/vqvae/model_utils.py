import functools
import numpy as np
from typing import Tuple, Union

import mindspore as ms
from mindspore import nn, ops


def _get_selected_flags(total_len: int, select_len: int, suffix: bool):
    assert select_len <= total_len
    selected = np.zeros(total_len, dtype=bool)
    if not suffix:
        selected[:select_len] = True
    else:
        selected[-select_len:] = True
    return selected


def get_norm_layer(norm_type, dtype):
    if norm_type == "LN":
        norm_fn = functools.partial(nn.LayerNorm, dtype=dtype)
    elif norm_type == "GN":
        norm_fn = functools.partial(nn.GroupNorm, dtype=dtype)
    elif norm_type is None:
        norm_fn = lambda: (lambda x: x)
    else:
        raise NotImplementedError(f"norm_type: {norm_type}")
    return norm_fn


def pad_at_dim(t, pad, dim=-1, value=0.0):
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = [(0, 0)] * dims_from_right
    pad_op = ops.Pad(tuple(zeros + [pad] + [(0, 0)] * 2))
    return pad_op(t)


def divisible_by(num, den):
    return (num % den) == 0


def is_odd(n):
    return not divisible_by(n, 2)


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


def nonlinearity(x, upcast=False):
    # swish
    ori_dtype = x.dtype
    if upcast:
        return x * (ops.sigmoid(x.astype(ms.float32))).astype(ori_dtype)
    else:
        return x * (ops.sigmoid(x))


def default(v, d):
    return v if v is not None else d


class GroupNormExtend(nn.GroupNorm):
    # GroupNorm supporting tensors with more than 4 dim
    def construct(self, x):
        x_shape = x.shape
        if x.ndim >= 5:
            x = x.view(x_shape[0], x_shape[1], x_shape[2], -1)
        y = super().construct(x)
        return y.view(x_shape)


class CausalConv3d(nn.Cell):
    """
    Temporal padding: Padding with the first frame, by repeating K_t-1 times.
    Spatial padding: follow standard conv3d, determined by pad mode and padding
    Ref: opensora plan

    Args:
        kernel_size: order (T, H, W)
        stride: order (T, H, W)
        padding: int, controls the amount of spatial padding applied to the input on both sides
    """

    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int, int, int]],
        padding: int = 0,
        dtype=ms.float32,
        **kwargs,
    ):
        super().__init__()
        assert isinstance(padding, int)
        kernel_size = cast_tuple(kernel_size, 3)
        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)

        dilation = kwargs.pop("dilation", 1)
        stride = kwargs.pop("stride", 1)
        stride = cast_tuple(stride, 3)  # (stride, 1, 1)
        dilation = cast_tuple(dilation, 3)  # (dilation, 1, 1)

        # pad temporal dimension by k-1, manually
        time_pad = dilation[0] * (time_kernel_size - 1) + (1 - stride[0])
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        self.time_pad = time_pad
        self.time_causal_padding = (
            (0, 0),
            (0, 0),
            (time_pad, 0),
            (height_pad, height_pad),
            (width_pad, width_pad),
        )

        # pad h,w dimensions if used, by conv3d API
        # diff from torch: bias, pad_mode

        self.conv = nn.Conv3d(
            chan_in,
            chan_out,
            kernel_size,
            stride=stride,
            dilation=dilation,
            has_bias=True,
            pad_mode="pad",
            **kwargs,
        ).to_float(dtype)

    def construct(self, x):
        # x: (bs, Cin, T, H, W )
        op_pad = ops.Pad(self.time_causal_padding)
        x = op_pad(x)
        x = self.conv(x)

        return x


def Normalize(in_channels, num_groups=32, extend=False, dtype=ms.float32):
    if extend:
        return GroupNormExtend(
            num_groups=num_groups,
            num_channels=in_channels,
            eps=1e-6,
            affine=True,
            dtype=dtype,
        )
    else:
        return nn.GroupNorm(
            num_groups=num_groups,
            num_channels=in_channels,
            eps=1e-6,
            affine=True,
            dtype=dtype,
        )


def SameConv2d(dim_in, dim_out, kernel_size):
    kernel_size = cast_tuple(kernel_size, 2)
    kernel_size_extend = (
        kernel_size[0],
        kernel_size[0],
        kernel_size[1],
        kernel_size[1],
    )
    padding = tuple([k // 2 for k in kernel_size_extend])
    return nn.Conv2d(
        dim_in,
        dim_out,
        kernel_size=kernel_size,
        padding=padding,
        pad_mode="pad",
        has_bias=True,
    )


def Avgpool3d(x):
    # ops.AvgPool3D(strides=(2, 2, 2))
    b, c, h, w, d = x.shape
    x = x.reshape(b * c, h, w, d)
    x = ops.AvgPool(kernel_size=1, strides=2)(x)
    x = ops.permute(x, (0, 2, 3, 1))
    x = ops.AvgPool(kernel_size=1, strides=(1, 2))(x)
    x = ops.permute(x, (0, 3, 1, 2))
    h, w, d = x.shape[-3:]
    x = x.reshape(b, c, h, w, d)
    return x


class Upsample3D(nn.Cell):
    def __init__(self, in_channels, with_conv, scale_factor, dtype=ms.float32):
        super().__init__()
        self.dtype = dtype
        self.with_conv = with_conv
        self.scale_factor = scale_factor
        if self.with_conv:
            self.conv = nn.Conv3d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                pad_mode="pad",
                padding=1,
                has_bias=True,
            ).to_float(self.dtype)

    def construct(self, x):
        b, c, t, h, w = x.shape

        x = ops.reshape(x, (b, c * t, h, w))

        # spatial upsample
        hw_in = x.shape[-2:]
        hw_out = tuple(int(f_ * s_) for s_, f_ in zip(hw_in, self.scale_factor[-2:]))
        x = ops.ResizeNearestNeighbor(hw_out)(x)

        # spatial upsample
        hw_size = int(h * self.scale_factor[1] * w * self.scale_factor[2])
        x = ops.reshape(x, (b, c, t, hw_size))
        hw_out = tuple([int(self.scale_factor[0] * t), hw_size])
        x = ops.ResizeNearestNeighbor(hw_out)(x)
        x = ops.reshape(
            x,
            (
                b,
                c,
                int(t * self.scale_factor[0]),
                int(h * self.scale_factor[1]),
                int(w * self.scale_factor[2]),
            ),
        )

        if self.with_conv:
            x = self.conv(x)
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

        self.conv = nn.Conv2d(
            self.chan_in,
            self.chan_out,
            self.kernel_size,
            stride=stride,
            padding=0,
            has_bias=True,
        ).to_float(dtype)

    def construct(self, x):
        # x shape: (b c t h w)

        b, c, t, h, w = x.shape

        x = ops.permute(x, (0, 2, 1, 3, 4))
        x = x.reshape(b * t, c, h, w)

        x = self.conv(x)

        _, c, h, w = x.shape
        x = x.reshape(b, t, c, h, w)
        x = ops.permute(x, (0, 2, 1, 3, 4))

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
        self.conv = nn.Conv2d(
            self.chan_in,
            self.chan_out,
            self.kernel_size,
            stride=stride,
            padding=1,
            pad_mode="pad",
            has_bias=True,
            dtype=dtype,
        ).to_float(dtype)

    def construct(self, x):
        b, c, t, h, w = x.shape

        # x = rearrange(x, "b c t h w -> b (c t) h w")
        x = ops.permute(x, (0, 2, 1, 3, 4))
        x = ops.reshape(x, (-1, c, h, w))

        hw_in = x.shape[-2:]
        scale_factor = 2
        hw_out = tuple(scale_factor * s_ for s_ in hw_in)
        x = ops.ResizeNearestNeighbor(hw_out)(x)

        x = self.conv(x)

        x = ops.reshape(x, (b, t, c, hw_out[0], hw_out[1]))
        x = ops.permute(x, (0, 2, 1, 3, 4))

        return x


class TimeDownsample2x(nn.Cell):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: int = 3,
        dtype=ms.float32,
    ):
        super().__init__()
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.kernel_size = kernel_size
        self.conv = CausalConv3d(chan_in, chan_out, kernel_size, stride=2).to_float(
            dtype
        )

    def construct(self, x):
        x = self.conv(x)
        return x


class TimeUpsample2x(nn.Cell):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: int = 3,
        dtype=ms.float32,
    ):
        super().__init__()
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.kernel_size = kernel_size
        self.conv = CausalConv3d(
            chan_in, chan_out, kernel_size, stride=1, dtype=dtype
        ).to_float(dtype)

    def construct(self, x):
        x = ops.permute(x, (0, 1, 3, 4, 2))

        b, c, h, w, t = x.shape

        x = x.reshape(b, c, -1, t)

        in_shape = x.shape[-2:]
        out_shape = (in_shape[0], in_shape[1] * 2)
        x = ops.ResizeNearestNeighbor(out_shape)(x)

        x = x.reshape(b, c, h, w, -1)
        x = ops.permute(x, (0, 1, 4, 2, 3))
        x = self.conv(x)

        return x

class ResnetBlock3D(nn.Cell):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.1,
        dtype=ms.float32,
        upcast_sigmoid=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut
        self.upcast_sigmoid = upcast_sigmoid

        # FIXME: GroupNorm precision mismatch with PT.
        self.norm1 = GroupNormExtend(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True, dtype=dtype
        )
        self.conv1 = CausalConv3d(in_channels, out_channels, 3, padding=1, dtype=dtype)
        self.norm2 = GroupNormExtend(
            num_groups=32, num_channels=out_channels, eps=1e-6, affine=True, dtype=dtype
        )
        self.conv2 = CausalConv3d(out_channels, out_channels, 3, padding=1, dtype=dtype)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = CausalConv3d(
                    in_channels, out_channels, 3, padding=1, dtype=dtype
                )
            else:
                self.nin_shortcut = CausalConv3d(
                    in_channels, out_channels, 1, padding=0, dtype=dtype
                )

    def construct(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h, self.upcast_sigmoid)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nonlinearity(h, self.upcast_sigmoid)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h
