import logging
from typing import Tuple, Union

import numpy as np

import mindspore as ms
from mindspore import nn, ops

_logger = logging.getLogger(__name__)


def divisible_by(num, den):
    return (num % den) == 0


def is_odd(n):
    return not divisible_by(n, 2)


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


class Conv2d(nn.Conv2d):
    """
    Conv2d for video input (B C T H W)
    """

    def rearrange_in(self, x):
        # b c f h w -> b f c h w
        B, C, F, H, W = x.shape
        x = ops.transpose(x, (0, 2, 1, 3, 4))
        # -> (b*f c h w)
        x = ops.reshape(x, (-1, C, H, W))

        return x

    def rearrange_out(self, x, F):
        BF, D, H_, W_ = x.shape
        # (b*f D h w) -> (b f D h w)
        x = ops.reshape(x, (BF // F, F, D, H_, W_))
        # -> (b D f h w)
        x = ops.transpose(x, (0, 2, 1, 3, 4))

        return x

    def construct(self, x):
        # import pdb; pdb.set_trace()
        # x: (b c f h w)
        F = x.shape[-3]
        x = self.rearrange_in(x)

        x = super().construct(x)

        x = self.rearrange_out(x, F)

        return x


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

        """
        if isinstance(padding, str):
            if padding == 'same':
                height_pad = height_kernel_size // 2
                width_pad = width_kernel_size // 2
            elif padding == 'valid':
                height_pad = 0
                width_pad = 0
            else:
                raise ValueError
        else:
            padding = list(cast_tuple(padding, 3))
        """

        # pad temporal dimension by k-1, manually
        self.time_pad = dilation[0] * (time_kernel_size - 1) + (1 - stride[0])
        if self.time_pad >= 1:
            self.temporal_padding = True
        else:
            self.temporal_padding = False

        # pad h,w dimensions if used, by conv3d API
        # diff from torch: bias, pad_mode

        # TODO: why not use HeUniform init?
        weight_init_value = 1.0 / (np.prod(kernel_size) * chan_in)
        if padding == 0:
            self.conv = nn.Conv3d(
                chan_in,
                chan_out,
                kernel_size,
                stride=stride,
                dilation=dilation,
                has_bias=True,
                pad_mode="valid",
                weight_init=weight_init_value,
                bias_init="zeros",
                **kwargs,
            ).to_float(dtype)
        else:
            # axis order (t0, t1, h0 ,h1, w0, w2)
            padding = list(cast_tuple(padding, 6))
            padding[0] = 0
            padding[1] = 0
            padding = tuple(padding)
            self.conv = nn.Conv3d(
                chan_in,
                chan_out,
                kernel_size,
                stride=stride,
                dilation=dilation,
                has_bias=True,
                pad_mode="pad",
                padding=padding,
                weight_init=weight_init_value,
                bias_init="zeros",
                **kwargs,
            ).to_float(dtype)

    def construct(self, x):
        # x: (bs, Cin, T, H, W )
        if self.temporal_padding:
            first_frame = x[:, :, :1, :, :]
            first_frame_pad = ops.repeat_interleave(first_frame, self.time_pad, axis=2)
            x = ops.concat((first_frame_pad, x), axis=2)

        return self.conv(x)
