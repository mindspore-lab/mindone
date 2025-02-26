from typing import Tuple, Union

import mindspore as ms
from mindspore import nn, ops


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


def divisible_by(num, den):
    return (num % den) == 0


def is_odd(n):
    return not divisible_by(n, 2)


def pad_at_dim(t, pad, dim=-1, value=0.0):
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = [(0, 0)] * dims_from_right
    pad_op = ops.Pad(tuple(zeros + [pad] + [(0, 0)] * 2))
    return pad_op(t)


def exists(v):
    return v is not None


def get_activation_fn(activation):
    if activation == "relu":
        activation_fn = nn.ReLU
    elif activation == "swish":
        activation_fn = nn.SiLU
    else:
        raise NotImplementedError
    return activation_fn


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
        strides=None,
        pad_mode="valid",
        dtype=ms.float32,
        **kwargs,
    ):
        super().__init__()

        kernel_size = cast_tuple(kernel_size, 3)
        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)

        dilation = kwargs.pop("dilation", 1)
        stride = strides[0] if strides is not None else kwargs.pop("stride", 1)

        # pad temporal dimension by k-1, manually
        time_pad = dilation * (time_kernel_size - 1) + (1 - stride)
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

        stride = strides if strides is not None else (stride, 1, 1)
        dilation = (dilation, 1, 1)

        self.conv = nn.Conv3d(
            chan_in,
            chan_out,
            kernel_size,
            stride=stride,
            dilation=dilation,
            pad_mode=pad_mode,
            dtype=dtype,
            **kwargs,
        ).to_float(dtype)

    def construct(self, x):
        # x: (bs, Cin, T, H, W )
        op_pad = ops.Pad(self.time_causal_padding)
        x = op_pad(x)
        x = self.conv(x)
        return x


class TimeDownsample2x(nn.Cell):
    def __init__(
        self,
        dim,
        dim_out,
        kernel_size=3,
        stride=1,
        dtype=ms.float32,
    ):
        super().__init__()
        self.time_causal_padding = (kernel_size - 1, 0)
        self.conv = nn.Conv1d(dim, dim_out, kernel_size, stride=stride, pad_mode="valid", dtype=dtype).to_float(dtype)

    def construct(self, x):
        b, c, t, h, w = x.shape
        x = x.permute(0, 3, 4, 1, 2)
        x = x.reshape(-1, c, t)

        x = ops.pad(x, self.time_causal_padding)
        x = self.conv(x)

        x = x.reshape(b, h, w, c, -1)
        x = x.permute(0, 3, 4, 1, 2)

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
            dtype=dtype,
        ).to_float(dtype)

    def construct(self, x):
        # x shape: (b c t h w)

        b, c_in, t_in, h_in, w_in = x.shape

        x = ops.permute(x, (0, 2, 1, 3, 4))
        x = x.reshape(b * t_in, c_in, h_in, w_in)

        x = self.conv(x)

        _, c_out, h_out, w_out = x.shape
        x = x.reshape(b, t_in, c_out, h_out, w_out)
        x = ops.permute(x, (0, 2, 1, 3, 4))

        return x


class TimeUpsample2x(nn.Cell):
    def __init__(
        self,
        dim,
        dim_out,
        kernel_size=3,
        dtype=ms.float32,
    ):
        super().__init__()

        self.conv = nn.Conv1d(dim, dim_out * 2, kernel_size, dtype=dtype).to_float(dtype)
        self.activate = nn.SiLU()

    def construct(self, x):
        b, c, t, h, w = x.shape
        x = x.permute(0, 3, 4, 1, 2)
        x = x.reshape(-1, c, t)

        x = self.conv(x)

        x = x.reshape(b, h, w, -1, t * 2)
        x = ops.permute(x, (0, 3, 4, 1, 2))

        x = self.activate(x)

        return x


class SpatialUpsample2x(nn.Cell):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int]] = (3, 3),
        dtype=ms.float32,
    ):
        super().__init__()
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.kernel_size = kernel_size

        self.conv = nn.Conv2d(
            self.chan_in,
            self.chan_out * 4,
            self.kernel_size,
            dtype=dtype,
        ).to_float(dtype)

        self.activate = nn.SiLU()

    def construct(self, x):
        b, c_in, t_in, h_in, w_in = x.shape

        x = ops.permute(x, (0, 2, 1, 3, 4))
        x = ops.reshape(x, (b * t_in, c_in, h_in, w_in))

        x = self.conv(x)

        x = ops.reshape(x, (b, t_in, self.chan_out, h_in * 2, w_in * 2))
        x = ops.permute(x, (0, 2, 1, 3, 4))

        x = self.activate(x)

        return x
