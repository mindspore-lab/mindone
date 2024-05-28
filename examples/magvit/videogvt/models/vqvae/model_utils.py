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


class Downsample(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        with_conv: bool,
        include_t_dim: bool = True,
        factor: int = 2,
        dtype=ms.float32,
    ):
        super().__init__()
        self.dtype = dtype
        self.with_conv = with_conv
        self.include_t_dim = include_t_dim
        self.factor = factor
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=2,
                pad_mode="valid",
                padding=0,
                has_bias=True,
            ).to_float(self.dtype)

    def construct(self, x):
        if self.with_conv:
            pad = ((0, 0), (0, 0), (0, 1), (0, 1))
            x = nn.Pad(paddings=pad)(x)
            x = self.conv(x)
        else:
            t_factor = self.factor if self.include_t_dim else 1
            shape = (t_factor, self.factor, self.factor)
            x = ops.AvgPool3D(kernel_size=shape, strides=shape)(x)
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
        # x = ops.pad(x, self.padding, mode="constant", value=0)
        x = self.pad(x)
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
        first_frame_pad = ops.repeat_interleave(first_frame, self.time_pad, axis=2)
        x = ops.concat((first_frame_pad, x), axis=2)

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
                x = ops.concat([x, x_], axis=2)
            else:
                x = ops.interpolate(x, scale_factor=(2.0, 1.0, 1.0), mode="trilinear")

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


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((3, 40, 40, 200))
    x = ms.Tensor(x, ms.float32)
    # test Residual Layer
    res = ResidualLayer(40, 40, 20)
    res_out = res(x)
    print("Res Layer out shape:", res_out.shape)
    # test res stack
    res_stack = ResidualStack(40, 40, 20, 3)
    res_stack_out = res_stack(x)
    print("Res Stack out shape:", res_stack_out.shape)
