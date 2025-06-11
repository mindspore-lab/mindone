"""
Modified from https://github.com/CompVis/taming-transformers/blob/master/taming/modules/diffusionmodules/model.py#L34
"""

from typing import Tuple, Union

import numpy as np

import mindspore as ms
import mindspore.common.initializer as initializer
import mindspore.mint.nn.functional as F
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import mint


def nonlinearity(x):
    # swish
    return x * mint.sigmoid(x)


def Normalize(in_channels):
    return mint.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Cell):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = mint.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def construct(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class DepthToSpaceUpsample(nn.Cell):
    def __init__(
        self,
        in_channels,
    ):
        super().__init__()
        conv = mint.nn.Conv2d(in_channels, in_channels * 4, 1)

        self.net = nn.SequentialCell(
            conv,
            mint.nn.SiLU(),
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = ms.Tensor(np.empty((o // 4, i, h, w)), dtype=ms.float32)
        conv_weight = initializer("HeUniform", conv_weight.shape, conv_weight.dtype)
        conv_weight = ops.repeat_interleave(conv_weight, 4, axis=0)

        conv.weight.set_data(conv_weight)
        conv.bias.set_data(mint.zeros_like(conv.bias))

    def construct(self, x):
        out = self.net(x)
        # Rearrange("b (c p1 p2) h w -> b c (h p1) (w p2)", p1=2, p2=2),
        b, c, h, w = out.shape
        out = out.reshape((b, c // 4, 2, 2, h, w))
        out = out.transpose(0, 1, 4, 2, 5, 3).reshape((b, c // 4, h * 2, w * 2))
        return out


class Downsample(nn.Cell):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = mint.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def construct(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


def unpack_time(t, batch):
    _, c, w, h = t.shape
    out = ops.reshape(t, (batch, -1, c, w, h))
    # b t c h w -> b c t h w
    out = ops.transpose(out, (0, 2, 1, 3, 4))
    return out


def pack_time(t):
    # b c t h w -> b t c h w
    out = ops.transpose(t, (0, 2, 1, 3, 4))
    _, _, c, w, h = out.shape
    return ops.reshape(out, (-1, c, w, h))


class TimeDownsample2x(nn.Cell):
    def __init__(
        self,
        dim,
        dim_out=None,
        kernel_size=3,
    ):
        super().__init__()
        if dim_out is None:
            dim_out = dim
        self.time_causal_padding = (kernel_size - 1, 0)
        self.conv = mint.nn.Conv1d(dim, dim_out, kernel_size, stride=2)

    def construct(self, x):
        # b c t h w -> b h w c t
        x = ops.transpose(x, (0, 3, 4, 1, 2))
        b, h, w, c, t = x.shape
        x = ops.reshape(x, (-1, c, t))

        x = F.pad(x, self.time_causal_padding)
        out = self.conv(x)

        out = ops.reshape(out, (b, h, w, c, t))
        # b h w c t -> b c t h w
        out = ops.transpose(out, (0, 3, 4, 1, 2))

        return out


class TimeUpsample2x(nn.Cell):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        if dim_out is None:
            dim_out = dim
        conv = mint.nn.Conv1d(dim, dim_out * 2, 1)

        self.net = nn.SequentialCell(
            mint.nn.SiLU(),
            conv,
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, t = conv.weight.shape
        conv_weight = ms.Tensor(np.empty((o // 2, i, t)), dtype=ms.float32)
        conv_weight = initializer("HeUniform", conv_weight.shape, conv_weight.dtype)
        conv_weight = ops.repeat_interleave(conv_weight, 2, axis=0)

        conv.weight.set_data(conv_weight)
        conv.bias.set_data(ops.zeros_like(conv.bias))

    def construct(self, x):
        # b c t h w -> b h w c t
        x = ops.transpose(x, (0, 3, 4, 1, 2))
        b, h, w, c, t = x.shape
        x = ops.reshape(x, (-1, c, t))

        out = self.net(x)
        # Rearrange("b (c p) t -> b c (t p)", p=2)
        out = ops.reshape(out, (b, -1, 2, t)).transpose((0, 1, 3, 2)).reshape((b, -1, t * 2))
        out = out[:, :, 1:]

        out = ops.reshape(out, (b, h, w, c, t))
        # b h w c t -> b c t h w
        out = ops.transpose(out, (0, 3, 4, 1, 2))
        return out


class AttnBlock(nn.Cell):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = mint.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = mint.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = mint.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = mint.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def construct(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = ops.reshape(q, (b, c, h * w))
        q = ops.transpose(q, (0, 2, 1))  # b,hw,c
        k = ops.reshape(k, (b, c, h * w))  # b,c,hw
        w_ = mint.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = ops.reshape(v, (b, c, h * w))
        w_ = ops.transpose(w_, (0, 2, 1))  # b,hw,hw (first hw of k, second of q)
        h_ = ops.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = ops.reshape(h_, (b, c, h, w))

        h_ = self.proj_out(h_)

        return x + h_


class TimeAttention(AttnBlock):
    def construct(self, x, *args, **kwargs):
        # b c t h w -> b h w t c
        x = ops.transpose(x, (0, 3, 4, 2, 1))
        b, h, w, t, c = x.shape
        x = ops.reshape(x, (-1, t, c))

        x = super().construct(x, *args, **kwargs)

        x = ops.reshape(x, (b, h, w, t, c))
        # b h w t c -> b c t h w
        x = ops.transpose(x, (0, 4, 3, 1, 2))
        return x


class Residual(nn.Cell):
    def __init__(self, fn: nn.Cell):
        super().__init__()
        self.fn = fn

    def construct(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


class CausalConv3d(nn.Cell):
    def __init__(self, chan_in, chan_out, kernel_size: Union[int, Tuple[int, int, int]], pad_mode="constant", **kwargs):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        dilation = kwargs.pop("dilation", 1)
        stride = kwargs.pop("stride", 1)

        self.pad_mode = pad_mode
        time_pad = dilation * (time_kernel_size - 1) + (1 - stride)
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        self.time_pad = time_pad
        self.time_causal_padding = (
            width_pad,
            width_pad,
            height_pad,
            height_pad,
            time_pad,
            0,
        )

        stride = (stride, 1, 1)
        dilation = (dilation, 1, 1)
        self.conv = mint.nn.Conv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)

    def construct(self, x):
        pad_mode = self.pad_mode if self.time_pad < x.shape[2] else "constant"

        x = F.pad(x, self.time_causal_padding, mode=pad_mode)
        return self.conv(x)


def ResnetBlockCausal3D(dim, kernel_size: Union[int, Tuple[int, int, int]], pad_mode: str = "constant"):
    net = nn.SequentialCell(
        Normalize(dim),
        mint.nn.SiLU(),
        CausalConv3d(dim, dim, kernel_size, pad_mode),
        Normalize(dim),
        mint.nn.SiLU(),
        CausalConv3d(dim, dim, kernel_size, pad_mode),
    )
    return Residual(net)


class ResnetBlock(nn.Cell):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = mint.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = mint.nn.Linear(temb_channels, out_channels)
        else:
            self.temb_proj = None
        self.norm2 = Normalize(out_channels)
        self.dropout = mint.nn.Dropout(dropout)
        self.conv2 = mint.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = mint.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = mint.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def construct(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h
