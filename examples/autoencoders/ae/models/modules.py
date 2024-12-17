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


class CausalConv3d(nn.Cell):
    """
    Temporal padding: Padding with the first frame, by repeating K_t-1 times.
    Spatial padding: follow standard conv3d, determined by pad mode and padding
    Ref: opensora plan

    Args:
        padding: controls the amount of padding applied to the input.
        int or a tuple of ints giving the amount of implicit padding applied on both sides
    """

    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int, int, int]],
        padding: Union[int, Tuple[int, int, int]] = 0,
        dtype=ms.float32,
        **kwargs,
    ):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)
        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)

        dilation = kwargs.pop("dilation", 1)
        stride = kwargs.pop("stride", 1)

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
        self.time_pad = dilation * (time_kernel_size - 1) + (1 - stride)

        # pad h,w dimensions if used, by conv3d API
        # diff from torch: bias, pad_mode
        stride = cast_tuple(stride, 3)  # (stride, 1, 1)
        dilation = cast_tuple(dilation, 3)  # (dilation, 1, 1)

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
        first_frame = x[:, :, :1, :, :]
        first_frame_pad = ops.repeat_interleave(first_frame, self.time_pad, axis=2)

        x = ops.concat((first_frame_pad, x), axis=2)

        return self.conv(x)


class ResnetBlock3D(nn.Cell):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut

        # FIXME: GroupNorm precision mismatch with PT.
        self.norm1 = Normalize(in_channels, extend=True)
        self.conv1 = CausalConv3d(in_channels, out_channels, 3, padding=1)
        self.norm2 = Normalize(out_channels, extend=True)
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = CausalConv3d(out_channels, out_channels, 3, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = CausalConv3d(in_channels, out_channels, 3, padding=1)
            else:
                self.nin_shortcut = CausalConv3d(in_channels, out_channels, 1, padding=0)

    def construct(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
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


class CausalConv3dZeroPad(nn.Cell):
    """
    Temporal Padding: pading with constant values (zero) by repeating t-1 times.
    Spatial Padding: same padding, filled with zeros
    Ref: magvit-v2 torch reproduction
    """

    def __init__(self, chan_in, chan_out, kernel_size: Union[int, Tuple[int, int, int]], pad_mode="constant", **kwargs):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)

        dilation = kwargs.pop("dilation", 1)
        stride = kwargs.pop("stride", 1)

        self.pad_mode = pad_mode
        time_pad = dilation * (time_kernel_size - 1) + (1 - stride)
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2
        self.time_pad = time_pad
        self.time_causal_padding = (width_pad, width_pad, height_pad, height_pad, time_pad, 0)

        stride = (stride, 1, 1)
        dilation = (dilation, 1, 1)
        # diff from torch: bias, pad_mode
        self.conv = nn.Conv3d(
            chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, has_bias=True, pad_mode="valid", **kwargs
        )

    def construct(self, x):
        # x: (bs, Cin, T, H, W )

        # FIXME: check dynamic shape issue in graph mode
        # x.shape[2] -- T axis
        pad_mode = self.pad_mode if self.time_pad < x.shape[2] else "constant"

        # nn.Pad can be more efficient but it doesn't support 5-dim padding currently.
        x = ops.pad(x, self.time_causal_padding, mode=pad_mode)

        return self.conv(x)


def nonlinearity(x, upcast=False):
    # swish
    ori_dtype = x.dtype
    if upcast:
        return x * (ops.sigmoid(x.astype(ms.float32))).astype(ori_dtype)
    else:
        return x * (ops.sigmoid(x))


class GroupNormExtend(nn.GroupNorm):
    # GroupNorm supporting tensors with more than 4 dim
    def construct(self, x):
        x_shape = x.shape
        if x.ndim >= 3:
            x = x.view(x_shape[0], x_shape[1], x_shape[2], -1)
        y = super().construct(x)
        return y.view(x_shape)


def Normalize(in_channels, num_groups=32, extend=False):
    if extend:
        return GroupNormExtend(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True).to_float(
            ms.float32
        )
    else:
        return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True).to_float(ms.float32)


class Upsample(nn.Cell):
    def __init__(self, in_channels, with_conv, dtype=ms.float32):
        super().__init__()
        self.dtype = dtype
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True
            ).to_float(self.dtype)

    def construct(self, x):
        in_shape = x.shape[-2:]
        out_shape = tuple(2 * x for x in in_shape)
        x = ops.ResizeNearestNeighbor(out_shape)(x)

        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Cell):
    def __init__(self, in_channels, with_conv, dtype=ms.float32):
        super().__init__()
        self.dtype = dtype
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, pad_mode="valid", padding=0, has_bias=True
            ).to_float(self.dtype)

    def construct(self, x):
        if self.with_conv:
            pad = ((0, 0), (0, 0), (0, 1), (0, 1))
            x = nn.Pad(paddings=pad)(x)
            x = self.conv(x)
        else:
            x = ops.AvgPool(kernel_size=2, stride=2)(x)
        return x


# used in vae
class ResnetBlock(nn.Cell):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
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

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True
        ).to_float(dtype)
        if temb_channels > 0:
            self.temb_proj = nn.Dense(temb_channels, out_channels, bias_init="normal").to_float(dtype)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True
        ).to_float(dtype)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True
                ).to_float(dtype)
            else:
                self.nin_shortcut = nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, pad_mode="valid", has_bias=True
                ).to_float(dtype)

    def construct(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h, upcast=self.upcast_sigmoid)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb, upcast=self.upcast_sigmoid))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h, upcast=self.upcast_sigmoid)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Cell):
    def __init__(self, in_channels, dtype=ms.float32):
        super().__init__()
        self.in_channels = in_channels
        self.dtype = dtype
        self.bmm = ops.BatchMatMul()
        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, pad_mode="valid", has_bias=True).to_float(
            dtype
        )
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, pad_mode="valid", has_bias=True).to_float(
            dtype
        )
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, pad_mode="valid", has_bias=True).to_float(
            dtype
        )
        self.proj_out = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, pad_mode="valid", has_bias=True
        ).to_float(dtype)

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
        w_ = self.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]

        w_ = w_ * (int(c) ** (-0.5))
        w_ = ops.Softmax(axis=2)(w_)

        # attend to values
        v = ops.reshape(v, (b, c, h * w))
        w_ = ops.transpose(w_, (0, 2, 1))  # b,hw,hw (first hw of k, second of q)
        h_ = self.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = ops.reshape(h_, (b, c, h, w))

        h_ = self.proj_out(h_)

        return x + h_


class AttnBlock3D(nn.Cell):
    def __init__(self, in_channels, dtype=ms.float32):
        super().__init__()
        self.in_channels = in_channels
        self.dtype = dtype

        self.bmm = ops.BatchMatMul()
        self.norm = Normalize(in_channels)

        # TODO: 1x1 conv3d can be replaced with Linear
        self.q = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1).to_float(dtype)
        self.k = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1).to_float(dtype)
        self.v = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1).to_float(dtype)
        self.proj_out = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1).to_float(dtype)

    def construct(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, t, h, w = q.shape
        q = ops.reshape(q, (b, c, h * w))
        q = ops.transpose(q, (0, 2, 1))  # b,hw,c
        k = ops.reshape(k, (b, c, h * w))  # b,c,hw
        w_ = self.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]

        w_ = w_ * (int(c) ** (-0.5))
        w_ = ops.Softmax(axis=2)(w_)

        # attend to values
        v = ops.reshape(v, (b, c, h * w))
        w_ = ops.transpose(w_, (0, 2, 1))  # b,hw,hw (first hw of k, second of q)
        h_ = self.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = ops.reshape(h_, (b, c, h, w))

        h_ = self.proj_out(h_)

        return x + h_


def make_attn(in_channels, attn_type="vanilla", dtype=ms.float32):
    assert attn_type in ["vanilla", "vanilla3D"], f"attn_type {attn_type} not supported"
    _logger.debug(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels, dtype=dtype)
    elif attn_type == "vanilla3D":
        return AttnBlock3D(in_channels, dtype=dtype)


# used in vae
class Encoder(nn.Cell):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        double_z=True,
        use_linear_attn=False,
        attn_type="vanilla",
        dtype=ms.float32,
        upcast_sigmoid=False,
        **ignore_kwargs,
    ):
        super().__init__()
        # if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.dtype = dtype
        self.upcast_sigmoid = (upcast_sigmoid,)

        # downsampling
        self.conv_in = nn.Conv2d(
            in_channels, self.ch, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True
        ).to_float(self.dtype)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.CellList(auto_prefix=False)
        for i_level in range(self.num_resolutions):
            block = nn.CellList()
            attn = nn.CellList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        dtype=self.dtype,
                        upcast_sigmoid=upcast_sigmoid,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type, dtype=self.dtype))
            down = nn.Cell()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv, dtype=self.dtype)
            else:
                down.downsample = nn.Identity()
            curr_res = curr_res // 2
            down.update_parameters_name(prefix=self.param_prefix + f"down.{i_level}.")
            self.down.append(down)

        # middle
        self.mid = nn.Cell()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            dtype=self.dtype,
            upcast_sigmoid=upcast_sigmoid,
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type, dtype=self.dtype)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            dtype=self.dtype,
            upcast_sigmoid=upcast_sigmoid,
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            pad_mode="pad",
            padding=1,
            has_bias=True,
        ).to_float(self.dtype)

    def construct(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h, upcast=self.upcast_sigmoid)
        h = self.conv_out(h)
        return h


# used in vae
class Decoder(nn.Cell):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        give_pre_end=False,
        tanh_out=False,
        use_linear_attn=False,
        attn_type="vanilla",
        dtype=ms.float32,
        upcast_sigmoid=False,
        **ignorekwargs,
    ):
        super().__init__()
        # if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.dtype = dtype
        self.upcast_sigmoid = upcast_sigmoid

        # compute in_ch_mult, block_in and curr_res at lowest res
        # in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        _logger.debug("Working with z of shape {} = {} dimensions.".format(self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = nn.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True
        ).to_float(self.dtype)

        # middle
        self.mid = nn.Cell()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout, dtype=self.dtype
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type, dtype=self.dtype)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout, dtype=self.dtype
        )

        # upsampling
        self.up = nn.CellList(auto_prefix=False)
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.CellList()
            attn = nn.CellList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        dtype=self.dtype,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type, dtype=self.dtype))
            up = nn.Cell()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv, dtype=self.dtype)
            else:
                up.upsample = nn.Identity()
            curr_res = curr_res * 2
            up.update_parameters_name(prefix=self.param_prefix + f"up.{i_level}.")
            if len(self.up) != 0:
                self.up.insert(0, up)
            else:
                self.up.append(up)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(
            block_in, out_ch, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True
        ).to_float(self.dtype)

    def construct(self, z):
        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        i_level = self.num_resolutions
        while i_level > 0:
            i_level -= 1
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h
        h = self.norm_out(h)
        h = nonlinearity(h, upcast=self.upcast_sigmoid)
        h = self.conv_out(h)
        if self.tanh_out:
            h = ops.tanh(h)
        return h
