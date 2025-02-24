# Copyright 2025 StepFun Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# ==============================================================================

from typing import Optional

import numpy as np
from safetensors import safe_open
from stepvideo.mindspore_adapter.pynative_utils import pynative_x_to_dtype
from stepvideo.mindspore_adapter.scaled_dot_product_attn import scaled_dot_product_attention

import mindspore as ms
from mindspore import Parameter, Tensor, mint, nn, ops
from mindspore.communication.management import get_rank


class Base_group_norm(nn.Cell):
    def __init__(self, norm_layer: mint.nn.GroupNorm, spatial=False):
        super().__init__(auto_prefix=False)
        self.norm_layer = norm_layer
        self.spatial = spatial

    def construct(self, x, act_silu=False, channel_last=False):
        if self.spatial:
            # assert channel_last == True
            x_shape = x.shape
            x = x.flatten(0, 1)
            if channel_last:
                # Permute to NCHW format
                x = mint.permute(x, (0, 3, 1, 2))

            out = mint.nn.functional.group_norm(
                x, self.norm_layer.num_groups, self.norm_layer.weight, self.norm_layer.bias, self.norm_layer.eps
            )
            if act_silu:
                out = mint.nn.functional.silu(out)

            if channel_last:
                # Permute back to NHWC format
                out = mint.permute(out, (0, 2, 3, 1))

            out = out.view(x_shape)
        else:
            if channel_last:
                # Permute to NCHW format
                x = mint.permute(x, (0, 3, 1, 2))
            out = mint.nn.functional.group_norm(
                x, self.norm_layer.num_groups, self.norm_layer.weight, self.norm_layer.bias, self.norm_layer.eps
            )
            if act_silu:
                out = mint.nn.functional.silu(out)
            if channel_last:
                # Permute back to NHWC format
                out = mint.permute(out, (0, 2, 3, 1))
        return out


class Base_group_norm_with_zero_pad(nn.Cell):
    def __init__(self, norm_layer: mint.nn.GroupNorm, spatial=False):
        super().__init__(auto_prefix=False)
        self.norm_layer = norm_layer
        self.base_group_norm = Base_group_norm(norm_layer, spatial=spatial)

    def construct(self, x, act_silu=True, pad_size=2):
        out_shape = list(x.shape)
        out_shape[1] += pad_size

        # FIXME: @jit bug
        out = mint.zeros(out_shape, dtype=x.dtype)
        # _out = ()
        # for i in range(out_shape[0]):
        #     _out += (ops.zeros(out_shape[1:], dtype=x.dtype),)
        # out = ops.stack(_out, axis=0)

        out[:, pad_size:] = self.base_group_norm(x, act_silu=act_silu, channel_last=True)
        out[:, :pad_size] = 0
        return out


class Base_conv2d(nn.Cell):
    def __init__(self, conv_layer):
        super().__init__(auto_prefix=False)
        self.conv_layer = conv_layer

    def construct(self, x, channel_last=False, residual=None):
        if channel_last:
            x = mint.permute(x, (0, 3, 1, 2))  # NHWC to NCHW
        out = mint.nn.functional.conv2d(
            x,
            self.conv_layer.weight,
            self.conv_layer.bias,
            stride=self.conv_layer.stride,
            padding=self.conv_layer.padding,
        )
        if residual is not None:
            if channel_last:
                residual = mint.permute(residual, (0, 3, 1, 2))  # NHWC to NCHW
            out += residual
        if channel_last:
            out = mint.permute(out, (0, 2, 3, 1))  # NCHW to NHWC
        return out


class Base_conv3d(nn.Cell):
    def __init__(self, conv_layer):
        super().__init__(auto_prefix=False)
        self.conv_layer = conv_layer

    def construct(self, x, channel_last=False, residual=None, only_return_output=False):
        if only_return_output:
            size = cal_outsize(
                x.shape, self.conv_layer.weight.shape, self.conv_layer.stride, padding=self.conv_layer.padding
            )
            return mint.zeros(size, dtype=x.dtype)
        if channel_last:
            x = mint.permute(x, (0, 4, 1, 2, 3))  # NDHWC to NCDHW
        out = mint.nn.functional.conv3d(
            x,
            self.conv_layer.weight,
            self.conv_layer.bias,
            stride=self.conv_layer.stride,
            padding=self.conv_layer.padding,
        )
        if residual is not None:
            if channel_last:
                residual = mint.permute(residual, (0, 4, 1, 2, 3))  # NDHWC to NCDHW
            out += residual
        if channel_last:
            out = mint.permute(out, (0, 2, 3, 4, 1))  # NCDHW to NDHWC
        return out


class Base_conv3d_channel_last(nn.Cell):
    def __init__(self, conv_layer):
        super().__init__(auto_prefix=False)
        self.conv_layer = conv_layer
        self.base_conv3d = Base_conv3d(conv_layer)

    def construct(self, x, residual=None):
        in_numel = x.numel()
        out_numel = int(x.numel() * self.conv_layer.out_channels / self.conv_layer.in_channels)
        if (in_numel >= 2**30) or (out_numel >= 2**30):
            # assert self.conv_layer.stride[0] == 1, "time split asks time stride = 1"

            B, T, H, W, C = x.shape
            K = self.conv_layer.kernel_size[0]

            chunks = 4
            chunk_size = T // chunks

            if residual is None:
                out_nhwc = self.base_conv3d(x, channel_last=True, residual=residual, only_return_output=True)
            else:
                out_nhwc = residual

            assert B == 1
            # outs = []
            for i in range(chunks):
                if i == chunks - 1:
                    xi = x[:1, chunk_size * i :]
                    out_nhwci = out_nhwc[:1, chunk_size * i :]
                else:
                    xi = x[:1, chunk_size * i : chunk_size * (i + 1) + K - 1]
                    out_nhwci = out_nhwc[:1, chunk_size * i : chunk_size * (i + 1)]
                if residual is not None:
                    if i == chunks - 1:
                        ri = residual[:1, chunk_size * i :]
                    else:
                        ri = residual[:1, chunk_size * i : chunk_size * (i + 1)]
                else:
                    ri = None
                out_nhwci.copy_(self.base_conv3d(xi, channel_last=True, residual=ri))
                # out_nhwci = self.base_conv3d(xi, channel_last=True, residual=ri)  # error
        else:
            out_nhwc = self.base_conv3d(x, channel_last=True, residual=residual)
        return out_nhwc


def cal_outsize(input_sizes, kernel_sizes, stride, padding):
    stride_d, stride_h, stride_w = stride
    padding_d, padding_h, padding_w = padding
    dilation_d, dilation_h, dilation_w = 1, 1, 1

    in_d = input_sizes[1]
    in_h = input_sizes[2]
    in_w = input_sizes[3]
    # in_channel = input_sizes[4]

    kernel_d = kernel_sizes[2]
    kernel_h = kernel_sizes[3]
    kernel_w = kernel_sizes[4]
    out_channels = kernel_sizes[0]

    out_d = calc_out_(in_d, padding_d, dilation_d, kernel_d, stride_d)
    out_h = calc_out_(in_h, padding_h, dilation_h, kernel_h, stride_h)
    out_w = calc_out_(in_w, padding_w, dilation_w, kernel_w, stride_w)
    size = [input_sizes[0], out_d, out_h, out_w, out_channels]
    return size


def calc_out_(in_size, padding, dilation, kernel, stride):
    return (in_size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


class Upsample2D(nn.Cell):
    def __init__(self, channels, use_conv=False, use_conv_transpose=False, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose

        if use_conv:
            self.conv = mint.nn.Conv2d(self.channels, self.out_channels, 3, padding=1, bias=True)
        else:
            assert "Not Supported"
            self.conv = mint.nn.ConvTranspose2d(channels, self.out_channels, 4, 2, 1, bias=True)

        self.base_conv2d = Base_conv2d(self.conv)

    def construct(self, x, output_size=None):
        # assert x.shape[-1] == self.channels

        if self.use_conv_transpose:
            return self.conv(x)

        if output_size is None:
            h, w = x.shape[1] * 2, x.shape[2] * 2
            output_size = (h, w)

        x = mint.permute(
            mint.nn.functional.interpolate(mint.permute(x, (0, 3, 1, 2)), size=output_size, mode="nearest"),
            (0, 2, 3, 1),
        ).contiguous()
        # x = self.conv(x)
        x = self.base_conv2d(x, channel_last=True)

        return x


class Downsample2D(nn.Cell):
    def __init__(self, channels, use_conv=False, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2

        if use_conv:
            self.conv = mint.nn.Conv2d(self.channels, self.out_channels, 3, stride=stride, padding=padding, bias=True)
        else:
            assert self.channels == self.out_channels
            self.conv = nn.AvgPool2d(kernel_size=stride, stride=stride, pad_mode="pad")

        self.base_conv2d = Base_conv2d(self.conv)

    def construct(self, x):
        # assert x.shape[-1] == self.channels
        if self.use_conv and self.padding == 0:
            pad = (0, 0, 0, 1, 0, 1)
            x = mint.nn.functional.pad(x, pad, mode="constant", value=0)

        # assert x.shape[-1] == self.channels
        # x = self.conv(x)
        x = self.base_conv2d(x, channel_last=True)
        return x


class CausalConv(nn.Cell):
    def __init__(self, chan_in, chan_out, kernel_size, **kwargs):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = kernel_size if isinstance(kernel_size, tuple) else ((kernel_size,) * 3)
        elif isinstance(kernel_size, list):
            kernel_size = tuple(kernel_size)

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        self.dilation = kwargs.pop("dilation", 1)
        self.stride = kwargs.pop("stride", 1)
        if isinstance(self.stride, int):
            self.stride = (self.stride, 1, 1)
        time_pad = self.dilation * (time_kernel_size - 1) + max((1 - self.stride[0]), 0)
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2
        self.time_causal_padding = (width_pad, width_pad, height_pad, height_pad, time_pad, 0)
        self.time_uncausal_padding = (width_pad, width_pad, height_pad, height_pad, 0, 0)

        if "bias" not in kwargs:
            kwargs["bias"] = True
        if "padding_mode" not in kwargs:
            kwargs["padding_mode"] = "zeros"

        self.conv = mint.nn.Conv3d(chan_in, chan_out, kernel_size, stride=self.stride, dilation=self.dilation, **kwargs)
        # self.is_first_run = True

    def construct(self, x, is_init=True, residual=None):
        x = mint.nn.functional.pad(x, self.time_causal_padding if is_init else self.time_uncausal_padding)

        x = self.conv(x)
        if residual is not None:
            x = x + residual
        return x


class ChannelDuplicatingPixelUnshuffleUpSampleLayer3D(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        assert out_channels * factor**3 % in_channels == 0
        self.repeats = out_channels * factor**3 // in_channels

    def construct(self, x: Tensor, is_init=True) -> Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.view(
            x.shape[0], self.out_channels, self.factor, self.factor, self.factor, x.shape[2], x.shape[3], x.shape[4]
        )
        x = mint.permute(x, (0, 1, 5, 2, 6, 3, 7, 4))
        x = x.view(
            x.shape[0], self.out_channels, x.shape[2] * self.factor, x.shape[4] * self.factor, x.shape[6] * self.factor
        )
        x = x[:, :, self.factor - 1 :, :, :]
        return x


class ConvPixelShuffleUpSampleLayer3D(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        factor: int,
    ):
        super().__init__()
        self.factor = factor
        out_ratio = factor**3
        self.conv = CausalConv(in_channels, out_channels * out_ratio, kernel_size=kernel_size)

    def construct(self, x: Tensor, is_init=True) -> Tensor:
        x = self.conv(x, is_init)
        x = self.pixel_shuffle_3d(x, self.factor)
        return x

    @staticmethod
    def pixel_shuffle_3d(x: Tensor, factor: int) -> Tensor:
        batch_size, channels, depth, height, width = x.shape
        new_channels = channels // (factor**3)
        new_depth = depth * factor
        new_height = height * factor
        new_width = width * factor

        x = x.view(batch_size, new_channels, factor, factor, factor, depth, height, width)
        x = mint.permute(x, (0, 1, 5, 2, 6, 3, 7, 4))
        x = x.view(batch_size, new_channels, new_depth, new_height, new_width)
        x = x[:, :, factor - 1 :, :, :]
        return x


class ConvPixelUnshuffleDownSampleLayer3D(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        factor: int,
    ):
        super().__init__()
        self.factor = factor
        out_ratio = factor**3
        assert out_channels % out_ratio == 0
        self.conv = CausalConv(in_channels, out_channels // out_ratio, kernel_size=kernel_size)

    def construct(self, x: Tensor, is_init=True) -> Tensor:
        x = self.conv(x, is_init)
        x = self.pixel_unshuffle_3d(x, self.factor)
        return x

    @staticmethod
    def pixel_unshuffle_3d(x: Tensor, factor: int) -> Tensor:
        pad = (0, 0, 0, 0, factor - 1, 0)  # (left, right, top, bottom, front, back)
        x = mint.nn.functional.pad(x, pad)
        B, C, D, H, W = x.shape
        x = x.view(B, C, D // factor, factor, H // factor, factor, W // factor, factor)
        x = mint.permute(x, (0, 1, 3, 5, 7, 2, 4, 6))
        x = x.view(B, C * factor**3, D // factor, H // factor, W // factor)
        return x


class PixelUnshuffleChannelAveragingDownSampleLayer3D(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        assert in_channels * factor**3 % out_channels == 0
        self.group_size = in_channels * factor**3 // out_channels

    def construct(self, x: Tensor, is_init=True) -> Tensor:
        pad = (0, 0, 0, 0, self.factor - 1, 0)  # (left, right, top, bottom, front, back)
        x = mint.nn.functional.pad(x, pad)
        B, C, D, H, W = x.shape
        x = x.view(B, C, D // self.factor, self.factor, H // self.factor, self.factor, W // self.factor, self.factor)
        x = mint.permute(x, (0, 1, 3, 5, 7, 2, 4, 6))
        x = x.view(B, C * self.factor**3, D // self.factor, H // self.factor, W // self.factor)
        x = x.view(B, self.out_channels, self.group_size, D // self.factor, H // self.factor, W // self.factor)
        x = x.mean(axis=2)
        return x


class CausalConvChannelLast(CausalConv):
    def __init__(self, chan_in, chan_out, kernel_size, **kwargs):
        super().__init__(chan_in, chan_out, kernel_size, **kwargs)

        self.time_causal_padding = (0, 0) + self.time_causal_padding
        self.time_uncausal_padding = (0, 0) + self.time_uncausal_padding

        self.base_conv3d_channel_last = Base_conv3d_channel_last(self.conv)

    def construct(self, x, is_init=True, residual=None):
        # if self.is_first_run:
        #     self.is_first_run = False
        #     # self.conv.weight = Parameter(self.conv.weight.permute(0,2,3,4,1).contiguous())

        x = mint.nn.functional.pad(x, self.time_causal_padding if is_init else self.time_uncausal_padding)

        x = self.base_conv3d_channel_last(x, residual=residual)
        return x


class CausalConvAfterNorm(CausalConv):
    def __init__(self, chan_in, chan_out, kernel_size, **kwargs):
        super().__init__(chan_in, chan_out, kernel_size, **kwargs)

        if "bias" not in kwargs:
            kwargs["bias"] = True
        if "padding_mode" not in kwargs:
            kwargs["padding_mode"] = "zeros"

        kernel_size = tuple(kernel_size) if isinstance(kernel_size, list) else kernel_size
        if self.time_causal_padding == (1, 1, 1, 1, 2, 0):
            self.conv = mint.nn.Conv3d(
                chan_in, chan_out, kernel_size, stride=self.stride, dilation=self.dilation, padding=(0, 1, 1), **kwargs
            )
        else:
            self.conv = mint.nn.Conv3d(
                chan_in, chan_out, kernel_size, stride=self.stride, dilation=self.dilation, **kwargs
            )
        # self.is_first_run = True

        self.base_conv3d_channel_last = Base_conv3d_channel_last(self.conv)

    def construct(self, x, is_init=True, residual=None):
        # if self.is_first_run:
        #     self.is_first_run = False

        if self.time_causal_padding == (1, 1, 1, 1, 2, 0):
            pass
        else:
            x = mint.nn.functional.pad(x, self.time_causal_padding).contiguous()

        x = self.base_conv3d_channel_last(x, residual=residual)
        return x


class AttnBlock(nn.Cell):
    def __init__(self, in_channels, spatial):
        super().__init__()

        self.norm = mint.nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.q = CausalConvChannelLast(in_channels, in_channels, kernel_size=1)
        self.k = CausalConvChannelLast(in_channels, in_channels, kernel_size=1)
        self.v = CausalConvChannelLast(in_channels, in_channels, kernel_size=1)
        self.proj_out = CausalConvChannelLast(in_channels, in_channels, kernel_size=1)

        self.base_group_norm = Base_group_norm(self.norm, spatial=spatial)

    def attention(self, x, is_init=True):
        x = self.base_group_norm(x, act_silu=False, channel_last=True)
        q = self.q(x, is_init)
        k = self.k(x, is_init)
        v = self.v(x, is_init)

        b, t, h, w, c = q.shape

        # q, k, v = map(lambda x: rearrange(x, "b t h w c -> b 1 (t h w) c"), (q, k, v))
        q = q.view(b, 1, t * h * w, c)
        k = k.view(b, 1, t * h * w, c)
        v = v.view(b, 1, t * h * w, c)

        x = scaled_dot_product_attention(q, k, v, is_causal=True)

        # x = rearrange(x, "b 1 (t h w) c -> b t h w c", t=t, h=h, w=w)
        x = x.view(b, t, h, w, -1)

        return x

    def construct(self, x):
        x = mint.permute(x, (0, 2, 3, 4, 1))
        h = self.attention(x)
        x = self.proj_out(h, residual=x)
        x = mint.permute(x, (0, 4, 1, 2, 3))
        return x


class Resnet3DBlock(nn.Cell):
    def __init__(self, in_channels, out_channels=None, temb_channels=512, conv_shortcut=False, spatial=False):
        super().__init__()

        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = mint.nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv1 = CausalConvAfterNorm(in_channels, out_channels, kernel_size=3)
        if temb_channels > 0:
            self.temb_proj = mint.nn.Linear(temb_channels, out_channels)

        self.norm2 = mint.nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.conv2 = CausalConvAfterNorm(out_channels, out_channels, kernel_size=3)

        assert conv_shortcut is False
        self.use_conv_shortcut = conv_shortcut
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = CausalConvAfterNorm(in_channels, out_channels, kernel_size=3)
            else:
                self.nin_shortcut = CausalConvAfterNorm(in_channels, out_channels, kernel_size=1)

        self.base_group_norm_with_zero_pad_1 = Base_group_norm_with_zero_pad(self.norm1, spatial=spatial)
        self.base_group_norm_with_zero_pad_2 = Base_group_norm_with_zero_pad(self.norm2, spatial=spatial)

    def construct(self, x, temb=None, is_init=True):
        x = mint.permute(x, (0, 2, 3, 4, 1))

        h = self.base_group_norm_with_zero_pad_1(x, act_silu=True, pad_size=2)
        h = self.conv1(h)
        if temb is not None:
            h = h + self.temb_proj(mint.nn.functional.silu(temb))[:, :, None, None]

        x = self.nin_shortcut(x) if self.in_channels != self.out_channels else x

        h = self.base_group_norm_with_zero_pad_2(h, act_silu=True, pad_size=2)
        x = self.conv2(h, residual=x)

        x = mint.permute(x, (0, 4, 1, 2, 3))
        return x


class Downsample3D(nn.Cell):
    def __init__(self, in_channels, with_conv, stride):
        super().__init__()

        self.with_conv = with_conv
        if with_conv:
            self.conv = CausalConv(in_channels, in_channels, kernel_size=3, stride=stride)

    def construct(self, x, is_init=True):
        if self.with_conv:
            x = self.conv(x, is_init)
        else:
            x = ops.avg_pool3d(x, kernel_size=2, stride=2)
        return x


class VideoEncoder(nn.Cell):
    def __init__(
        self,
        ch=32,
        ch_mult=(4, 8, 16, 16),
        num_res_blocks=2,
        in_channels=3,
        z_channels=16,
        double_z=True,
        down_sampling_layer=[1, 2],
        resamp_with_conv=True,
        version=1,
        spatial=False,
    ):
        super().__init__()

        temb_ch = 0

        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        # downsampling
        self.conv_in = CausalConv(in_channels, ch, kernel_size=3)
        self.down_sampling_layer = down_sampling_layer

        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.CellList()
        for i_level in range(self.num_resolutions):
            block = nn.CellList()
            attn = nn.CellList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    Resnet3DBlock(in_channels=block_in, out_channels=block_out, temb_channels=temb_ch, spatial=spatial)
                )
                block_in = block_out
            down = nn.Cell()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                if i_level in self.down_sampling_layer:
                    down.downsample = Downsample3D(block_in, resamp_with_conv, stride=(2, 2, 2))
                else:
                    down.downsample = Downsample2D(block_in, resamp_with_conv, padding=0)  # DIFF
            self.down.append(down)

        # middle
        self.mid = nn.Cell()
        self.mid.block_1 = Resnet3DBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=temb_ch, spatial=spatial
        )
        self.mid.attn_1 = AttnBlock(block_in, spatial=spatial)
        self.mid.block_2 = Resnet3DBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=temb_ch, spatial=spatial
        )

        # end
        self.norm_out = mint.nn.GroupNorm(num_groups=32, num_channels=block_in)
        self.version = version
        if version == 2:
            channels = 4 * z_channels * 2**3
            self.conv_patchify = ConvPixelUnshuffleDownSampleLayer3D(block_in, channels, kernel_size=3, factor=2)
            self.shortcut_pathify = PixelUnshuffleChannelAveragingDownSampleLayer3D(block_in, channels, 2)
            self.shortcut_out = PixelUnshuffleChannelAveragingDownSampleLayer3D(
                channels, 2 * z_channels if double_z else z_channels, 1
            )
            self.conv_out = CausalConvChannelLast(channels, 2 * z_channels if double_z else z_channels, kernel_size=3)
        else:
            self.conv_out = CausalConvAfterNorm(block_in, 2 * z_channels if double_z else z_channels, kernel_size=3)

        self.base_group_norm = Base_group_norm(self.norm_out, spatial=spatial)
        self.base_group_norm_with_zero_pad = Base_group_norm_with_zero_pad(self.norm_out, spatial=spatial)

    # @inference_mode()
    def construct(self, x, video_frame_num, is_init=True):
        # timestep embedding
        temb = None

        # t = video_frame_num

        # downsampling
        h = self.conv_in(x, is_init)

        # make it real channel last, but behave like normal layout
        h = mint.permute(mint.permute(h, (0, 2, 3, 4, 1)), (0, 4, 1, 2, 3))

        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h, temb, is_init)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)

            if i_level != self.num_resolutions - 1:
                if isinstance(self.down[i_level].downsample, Downsample2D):
                    # _, _, t, _, _ = h.shape
                    # h = rearrange(h, "b c t h w -> (b t) h w c", t=t)
                    _b, _c, _t, _h, _w = h.shape
                    h = h.transpose(0, 2, 3, 4, 1).view(_b * _t, _h, _w, _c)

                    h = self.down[i_level].downsample(h)
                    # h = rearrange(h, "(b t) h w c -> b c t h w", t=t)
                    _, _h, _w, _c = h.shape
                    h = h.view(-1, _t, _h, _w, _c).transpose(0, 4, 1, 2, 3)
                else:
                    h = self.down[i_level].downsample(h, is_init)

        h = self.mid.block_1(h, temb, is_init)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb, is_init)

        h = mint.permute(h, (0, 2, 3, 4, 1))  # b c l h w -> b l h w c
        if self.version == 2:
            h = self.base_group_norm(h, act_silu=True, channel_last=True)
            h = mint.permute(h, (0, 4, 1, 2, 3))
            shortcut = self.shortcut_pathify(h, is_init)
            h = self.conv_patchify(h, is_init)
            h = h + shortcut
            shortcut = mint.permute(self.shortcut_out(h, is_init), (0, 2, 3, 4, 1))
            h = self.conv_out(mint.permute(h, (0, 2, 3, 4, 1)), is_init)
            h = h + shortcut
        else:
            h = self.base_group_norm_with_zero_pad(h, act_silu=True, pad_size=2)
            h = self.conv_out(h, is_init)
        h = mint.permute(h, (0, 4, 1, 2, 3))  # b l h w c -> b c l h w

        # h = rearrange(h, "b c t h w -> b t c h w")
        h = mint.swapaxes(h, 1, 2)

        return h


class Res3DBlockUpsample(nn.Cell):
    def __init__(self, input_filters, num_filters, down_sampling_stride, down_sampling=False, spatial=False):
        super().__init__()

        self.input_filters = input_filters
        self.num_filters = num_filters

        self.act_ = mint.nn.SiLU()

        self.conv1 = CausalConvChannelLast(num_filters, num_filters, kernel_size=(3, 3, 3))
        self.norm1 = mint.nn.GroupNorm(32, num_filters)

        self.conv2 = CausalConvChannelLast(num_filters, num_filters, kernel_size=(3, 3, 3))
        self.norm2 = mint.nn.GroupNorm(32, num_filters)

        self.down_sampling = down_sampling
        if down_sampling:
            self.down_sampling_stride = down_sampling_stride
        else:
            self.down_sampling_stride = [1, 1, 1]

        if num_filters != input_filters or down_sampling:
            self.conv3 = CausalConvChannelLast(
                input_filters, num_filters, kernel_size=[1, 1, 1], stride=self.down_sampling_stride
            )
            self.norm3 = mint.nn.GroupNorm(32, num_filters)

        self.base_group_norm_1 = Base_group_norm(self.norm1, spatial=spatial)
        self.base_group_norm_2 = Base_group_norm(self.norm2, spatial=spatial)
        if num_filters != input_filters or down_sampling:
            self.base_group_norm_3 = Base_group_norm(self.norm3, spatial=spatial)

    def construct(self, x, is_init=False):
        x = mint.permute(x, (0, 2, 3, 4, 1))

        residual = x

        h = self.conv1(x, is_init)
        h = self.base_group_norm_1(h, act_silu=True, channel_last=True)

        h = self.conv2(h, is_init)
        h = self.base_group_norm_2(h, act_silu=False, channel_last=True)

        if self.down_sampling or self.num_filters != self.input_filters:
            x = self.conv3(x, is_init)
            x = self.base_group_norm_3(x, act_silu=False, channel_last=True)

        h = h + x
        h = self.act_(h)
        if residual is not None:
            h = h + residual

        h = mint.permute(h, (0, 4, 1, 2, 3))
        return h


class Upsample3D(nn.Cell):
    def __init__(self, in_channels, scale_factor=2, spatial=False):
        super().__init__()

        self.scale_factor = float(scale_factor)
        self.conv3d = Res3DBlockUpsample(
            input_filters=in_channels,
            num_filters=in_channels,
            down_sampling_stride=(1, 1, 1),
            down_sampling=False,
            spatial=spatial,
        )

    def construct(self, x, is_init=True, is_split=True):
        b, c, t, h, w = x.shape

        # for interpolate op
        _dtype = x.dtype
        x = x.to(ms.float32)

        # x = x.permute(0,2,3,4,1).contiguous().permute(0,4,1,2,3).to(memory_format=torch.channels_last_3d)
        if is_split:
            split_size = c // 8
            x_slices = mint.split(x, split_size, dim=1)
            x = [mint.nn.functional.interpolate(x, scale_factor=self.scale_factor) for x in x_slices]
            x = mint.cat(x, dim=1)
        else:
            x = mint.nn.functional.interpolate(x, scale_factor=self.scale_factor)
            # x = ops.interpolate(x, size=(int(x.shape[-3]*self.scale_factor), int(x.shape[-2]*self.scale_factor), int(x.shape[-1]*self.scale_factor)))

        x = x.to(_dtype)

        x = self.conv3d(x, is_init)
        return x


class VideoDecoder(nn.Cell):
    def __init__(
        self,
        ch=128,
        z_channels=16,
        out_channels=3,
        ch_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        temporal_up_layers=[2, 3],
        temporal_downsample=4,
        resamp_with_conv=True,
        version=1,
        spatial=False,
    ):
        super().__init__()

        temb_ch = 0

        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.temporal_downsample = temporal_downsample

        block_in = ch * ch_mult[self.num_resolutions - 1]
        self.version = version
        if version == 2:
            channels = 4 * z_channels * 2**3
            self.conv_in = CausalConv(z_channels, channels, kernel_size=3)
            self.shortcut_in = ChannelDuplicatingPixelUnshuffleUpSampleLayer3D(z_channels, channels, 1)
            self.conv_unpatchify = ConvPixelShuffleUpSampleLayer3D(channels, block_in, kernel_size=3, factor=2)
            self.shortcut_unpathify = ChannelDuplicatingPixelUnshuffleUpSampleLayer3D(channels, block_in, 2)
        else:
            self.conv_in = CausalConv(z_channels, block_in, kernel_size=3)

        # middle
        self.mid = nn.Cell()
        self.mid.block_1 = Resnet3DBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=temb_ch, spatial=spatial
        )
        self.mid.attn_1 = AttnBlock(block_in, spatial=spatial)
        self.mid.block_2 = Resnet3DBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=temb_ch, spatial=spatial
        )

        # upsampling
        self.up_id = len(temporal_up_layers)
        self.video_frame_num = 1
        self.cur_video_frame_num = self.video_frame_num // 2**self.up_id + 1
        self.up = nn.CellList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.CellList()
            attn = nn.CellList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    Resnet3DBlock(in_channels=block_in, out_channels=block_out, temb_channels=temb_ch, spatial=spatial)
                )
                block_in = block_out
            up = nn.Cell()
            up.block = block
            up.attn = attn
            if i_level != 0:
                if i_level in temporal_up_layers:
                    up.upsample = Upsample3D(block_in, spatial=spatial)
                    self.cur_video_frame_num = self.cur_video_frame_num * 2
                else:
                    up.upsample = Upsample2D(block_in, resamp_with_conv)
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = mint.nn.GroupNorm(num_groups=32, num_channels=block_in)
        self.conv_out = CausalConvAfterNorm(block_in, out_channels, kernel_size=3)

        self.base_group_norm_with_zero_pad = Base_group_norm_with_zero_pad(self.norm_out, spatial=spatial)

    # @inference_mode()
    def construct(self, z, is_init=True):
        # z = rearrange(z, "b t c h w -> b c t h w")
        z = mint.swapaxes(z, 1, 2)

        h = self.conv_in(z, is_init=is_init)
        if self.version == 2:
            shortcut = self.shortcut_in(z, is_init=is_init)
            h = h + shortcut
            shortcut = self.shortcut_unpathify(h, is_init=is_init)
            h = self.conv_unpatchify(h, is_init=is_init)
            h = h + shortcut

        temb = None

        h = mint.permute(mint.permute(h, (0, 2, 3, 4, 1)), (0, 4, 1, 2, 3))
        h = self.mid.block_1(h, temb, is_init=is_init)
        h = self.mid.attn_1(h)
        h = mint.permute(mint.permute(h, (0, 2, 3, 4, 1)), (0, 4, 1, 2, 3))
        h = self.mid.block_2(h, temb, is_init=is_init)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = mint.permute(mint.permute(h, (0, 2, 3, 4, 1)), (0, 4, 1, 2, 3))
                h = self.up[i_level].block[i_block](h, temb, is_init=is_init)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                if isinstance(self.up[i_level].upsample, Upsample2D):
                    B = h.shape[0]
                    h = mint.permute(h, (0, 2, 3, 4, 1)).flatten(0, 1)
                    h = self.up[i_level].upsample(h)
                    # h = h.unflatten(0, (B, -1)).permute(0,4,1,2,3)
                    h = nn.Unflatten(0, (B, -1))(h)
                    h = mint.permute(h, (0, 4, 1, 2, 3))
                else:
                    h = self.up[i_level].upsample(h, is_init=is_init)

        # end
        h = mint.permute(h, (0, 2, 3, 4, 1))  # b c l h w -> b l h w c
        h = self.base_group_norm_with_zero_pad(h, act_silu=True, pad_size=2)
        h = self.conv_out(h)
        h = mint.permute(h, (0, 4, 1, 2, 3))

        if is_init:
            h = h[:, :, (self.temporal_downsample - 1) :]
        return h


def rms_norm(input, normalized_shape, eps=1e-6):
    dtype = input.dtype
    input = input.to(ms.float32)
    variance = input.pow(2).flatten(-len(normalized_shape)).mean(-1)[(...,) + (None,) * len(normalized_shape)]
    input = input * mint.rsqrt(variance + eps)
    return input.to(dtype)


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False, rms_norm_mean=False, only_return_mean=False):
        self.parameters = parameters
        self.mean, self.logvar = mint.chunk(parameters, 2, dim=-3)  # N,[X],C,H,W
        self.logvar = mint.clamp(self.logvar, -30.0, 20.0)
        self.std = mint.exp(0.5 * self.logvar)
        self.var = mint.exp(self.logvar)
        self.deterministic = deterministic
        if self.deterministic:
            self.var = self.std = mint.zeros_like(self.mean, dtype=self.parameters.dtype)
        if rms_norm_mean:
            self.mean = rms_norm(self.mean, self.mean.shape[1:])
        self.only_return_mean = only_return_mean

    def sample(self, generator=None):
        # as the parameters and has same dtype
        sample = mint.randn(self.mean.shape)
        sample = sample.to(dtype=self.parameters.dtype)
        x = self.mean + self.std * sample
        if self.only_return_mean:
            return self.mean
        else:
            return x


class AutoencoderKL(nn.Cell):
    # @with_empty_init
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        z_channels=16,
        num_res_blocks=2,
        model_path=None,
        weight_dict={},
        world_size=1,
        version=1,
    ):
        super().__init__()

        self.frame_len = 17
        self.latent_len = 3 if version == 2 else 5

        spatial = True if version == 2 else False

        self.encoder = VideoEncoder(
            in_channels=in_channels,
            z_channels=z_channels,
            num_res_blocks=num_res_blocks,
            version=version,
            spatial=spatial,
        )

        self.decoder = VideoDecoder(
            z_channels=z_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            version=version,
            spatial=spatial,
        )

        # FXIME: comment for test
        if model_path is not None:
            weight_dict = self.init_from_ckpt(model_path)
            ms.load_param_into_net(self, weight_dict)
            # self.load_from_dict(weight_dict)

        self.convert_channel_last()

        self.world_size = world_size

    def init_from_ckpt(self, model_path):
        # 1. original
        # from safetensors import safe_open
        # p = {}
        # with safe_open(model_path, framework="pt", device="cpu") as f:
        #     for k in f.keys():
        #         tensor = f.get_tensor(k)
        #         if k.startswith("decoder.conv_out."):
        #             k = k.replace("decoder.conv_out.", "decoder.conv_out.conv.")
        #         p[k] = tensor
        # return p

        # 2. fix
        p = {}
        with safe_open(model_path, framework="np", device="cpu") as f:
            for k in f.keys():
                tensor = f.get_tensor(k)
                if k.startswith("decoder.conv_out."):
                    k = k.replace("decoder.conv_out.", "decoder.conv_out.conv.")
                assert isinstance(tensor, np.ndarray)
                p[k] = Parameter(tensor)
        return p

        # 3. old
        # from transformers.utils import is_safetensors_available
        # from mindone.safetensors.mindspore import load_file as safe_load_file
        # if model_path.endswith(".safetensors") and is_safetensors_available():
        #     # Check format of the archive
        #     with safe_open(model_path, framework="np") as f:
        #         metadata = f.metadata()
        #     if metadata.get("format") not in ["pt", "tf", "flax", "np"]:
        #         raise OSError(
        #             f"The safetensors archive passed at {model_path} does not contain the valid metadata. Make sure "
        #             "you save your model with the `save_pretrained` method."
        #         )
        #     state_dict = safe_load_file(model_path)

        #     # filter `decoder.conv_out`
        #     for k in state_dict.keys().copy():
        #         if k.startswith("decoder.conv_out."):
        #             new_k = k.replace("decoder.conv_out.", "decoder.conv_out.conv.")
        #             state_dict[new_k] = state_dict.pop(k)

        #     return state_dict

        # else:
        #     raise NotImplementedError(
        #         f"Only supports deserialization of weights file in safetensors format, but got {checkpoint_file}"
        #     )

    # def load_from_dict(self, state_dict, start_prefix=""):

    #     from mindone.transformers.modeling_utils import _convert_state_dict

    #     state_dict_ms = _convert_state_dict(self, state_dict, prefix="")

    #     local_state = {start_prefix + k: v for k, v in self.parameters_and_names()}
    #     for k, v in state_dict.items():
    #         if k in local_state:
    #             v.set_dtype(local_state[k].dtype)
    #         else:
    #             pass  # unexpect key keeps origin dtype
    #     ms.load_param_into_net(self, state_dict_ms, strict_load=True)

    def convert_channel_last(self):
        # Conv2d NCHW->NHWC
        pass

    def naive_encode(self, x, is_init_image=True):
        b, l, c, h, w = x.shape
        # x = rearrange(x, 'b l c h w -> b c l h w').contiguous()
        x = mint.swapaxes(x, 1, 2)
        z = self.encoder(x, l, True)  # 下采样[1, 4, 8, 16, 16]
        return z

    # @inference_mode()
    def encode(self, x):
        # b (nc cf) c h w -> (b nc) cf c h w -> encode -> (b nc) cf c h w -> b (nc cf) c h w
        chunks = list(x.split(self.frame_len, axis=1))
        for i in range(len(chunks)):
            chunks[i] = self.naive_encode(chunks[i], True)
        z = mint.cat(chunks, dim=1)

        posterior = DiagonalGaussianDistribution(z)  # FIXME: adapte to static graph
        return posterior.sample()

    def decode_naive(self, z, is_init=True):
        # z = z.to(next(self.decoder.parameters()).dtype)
        z = z.to(self.decoder.conv_in.conv.weight.dtype)
        dec = self.decoder(z, is_init)
        return dec

    # @inference_mode()
    # @ms.jit
    def decode(self, z):
        # b (nc cf) c h w -> (b nc) cf c h w -> decode -> (b nc) c cf h w -> b (nc cf) c h w
        chunks = list(z.split(self.latent_len, axis=1))

        # for @jit
        max_num_per_rank = None
        chunks_total_num = None

        if self.world_size > 1:
            chunks_total_num = len(chunks)
            max_num_per_rank = (chunks_total_num + self.world_size - 1) // self.world_size
            rank = get_rank()
            chunks_ = chunks[max_num_per_rank * rank : max_num_per_rank * (rank + 1)]
            if len(chunks_) < max_num_per_rank:
                chunks_.extend(chunks[: max_num_per_rank - len(chunks_)])
            chunks = chunks_

        for i in range(len(chunks)):
            chunks[i] = mint.permute(self.decode_naive(chunks[i], True), (0, 2, 1, 3, 4))
        x = mint.cat(chunks, dim=1)

        if self.world_size > 1:
            # x_ = ops.zeros([x.shape[0], (self.world_size * max_num_per_rank) * self.frame_len, *x.shape[2:]], dtype=x.dtype)
            # torch.distributed.all_gather_into_tensor(x_, x)
            x_ = ops.AllGather()(x).view(
                [x.shape[0], (self.world_size * max_num_per_rank) * self.frame_len, *x.shape[2:]]
            )
            x = x_[:, : chunks_total_num * self.frame_len]

        x = self.mix(x)

        # x = x.to(ms.float32)
        x = pynative_x_to_dtype(x, ms.float32)

        return x

    def mix(self, x):
        remain_scale = 0.6
        mix_scale = 1.0 - remain_scale
        front = slice(self.frame_len - 1, x.shape[1] - 1, self.frame_len)
        back = slice(self.frame_len, x.shape[1], self.frame_len)
        x[:, back] = x[:, back] * remain_scale + x[:, front] * mix_scale
        x[:, front] = x[:, front] * remain_scale + x[:, back] * mix_scale
        return x

    def to(self, dtype: Optional[ms.Type] = None):
        for p in self.get_parameters():
            p.set_dtype(dtype)
        return self
