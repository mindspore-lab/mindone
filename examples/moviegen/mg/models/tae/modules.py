import logging

import numpy as np
from packaging import version

import mindspore as ms
from mindspore import mint, nn, ops

_logger = logging.getLogger(__name__)


def divisible_by(num, den):
    return (num % den) == 0


def is_odd(n):
    return not divisible_by(n, 2)


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


def nonlinearity(x):
    return x * (ops.sigmoid(x))


def symmetric_pad1d(x):
    # x: (B C T), work with kernel size = 1
    first_frame = x[:, :, :1]
    last_frame = x[:, :, -1:]
    # last_frame_pad = ops.cat([last_frame] * self.time_pad, axis=2)
    x = ops.concat((first_frame, x, last_frame), axis=2)

    return x


class GroupNorm5d(nn.GroupNorm):
    def construct(self, x):
        # x (b c t h w)
        x_shape = x.shape
        x_ndim = x.ndim
        if x_ndim == 5:
            # (b c f h w) -> (b c f h*w)
            x = ops.reshape(x, (x_shape[0], x_shape[1], x_shape[2], -1))

        out = super().construct(x)

        if x_ndim == 5:
            # (b c f h*w) -> (b c f h w)
            out = ops.reshape(out, (x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]))

        return out


def Normalize(in_channels, num_groups=32):
    if version.parse(ms.__version__) >= version.parse("2.3.1"):
        return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
    else:
        return GroupNorm5d(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


def rearrange_in_spatial(x):
    # (b t c h w) -> (b*t c h w)
    B, T, C, H, W = x.shape
    x = ops.reshape(x, (B * T, C, H, W))
    return x


def rearrange_out_spatial(x, T):
    # (b*t c h w) -> (b t c h w)
    BT, C, H, W = x.shape
    x = ops.reshape(x, (BT // T, T, C, H, W))
    return x


def rearrange_in_temporal(x):
    # (b t c h w) -> (b*h*w c t)
    B, C, T, H, W = x.shape
    # (b t c h w) -> (b h w c t)
    x = ops.transpose(x, (0, 3, 4, 2, 1))
    # (b h w c t) -> (b*h*w c t)
    x = ops.reshape(x, (B * H * W, C, T))
    return x


def rearrange_out_temporal(x, H, W):
    # (b*h*w c t) -> (b t c h w)
    BHW, C, T = x.shape
    # (b*h*w c t) -> (b h w c t)
    x = ops.reshape(x, (BHW // (H * W), H, W, C, T))
    # (b h w c t) -> (b t c h w)
    x = ops.transpose(x, (0, 4, 3, 1, 2))
    return x


class Conv2_5d(nn.Cell):
    r"""
    Conv2.5d, a 2D spatial convolution followed by 1D temporal convolution
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        pad_mode="valid",
        padding=0,
        dilation=1,
        has_bias=True,
        **kwargs,
    ):
        super().__init__()
        assert stride == 1
        assert dilation == 1
        # spatial conv
        self.conv_spat = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            pad_mode=pad_mode,
            padding=padding,
            has_bias=has_bias,
        )

        # temporal conv
        if kernel_size > 1:
            # symmetric padding + conv1d
            assert kernel_size == 3, "symmetric padding currently only support kernel size 3"
            self.conv_temp = nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                pad_mode="valid",
                has_bias=has_bias,
                bias_init="zeros",
            )
            self.pad = symmetric_pad1d
            self.use_pad = True
        else:
            self.use_pad = False
            self.conv_temp = nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                pad_mode="valid",
                has_bias=has_bias,
                bias_init="zeros",
            )

        self.init_temporal_weight("centric")

    def construct(self, x):
        """
        Parameters:
            x: (b c t h w)
        Returns:
            (b c t h w)
        """

        B, Ci, T, Hi, Wi = x.shape
        # (b c t h w) -> (b t c h w)
        x = ops.transpose(x, (0, 2, 1, 3, 4))

        # spatial conv2d
        # (b t c h w) -> (b*t c h w)
        x = ops.reshape(x, (B * T, Ci, Hi, Wi))

        x = self.conv_spat(x)

        # (b*t c h w) -> (b t c h w)
        _, Co, Ho, Wo = x.shape
        x = ops.reshape(x, (B, T, Co, Ho, Wo))

        # temporal conv1d
        # (b t c h w) -> (b*h*w c t)
        x = ops.transpose(x, (0, 3, 4, 2, 1))  # (b t c h w) -> (b h w c t)
        x = ops.reshape(x, (B * Ho * Wo, Co, T))

        if self.use_pad:
            # import pdb; pdb.set_trace()
            x = self.pad(x)

        # import pdb; pdb.set_trace()
        x = self.conv_temp(x)

        # (b*h*w c t) -> (b t c h w)
        _, _, To = x.shape
        # (b*h*w c t) -> (b h w c t)
        x = ops.reshape(x, (B, Ho, Wo, Co, To))
        # (b h w c t) -> (b c t h w)
        x = ops.transpose(x, (0, 3, 4, 1, 2))

        return x

    def init_temporal_weight(self, method="centric"):
        if method == "normal":
            return

        elif method == "centric":
            # temporal conv kernel: (cout, cin, 1, ks)
            # ks=1 or 3, cin == cout
            w = self.conv_temp.weight
            ch = int(w.shape[0])
            ks = int(w.shape[-1])
            value = np.zeros(tuple(w.shape))

            # only the middle element of the kernel is 1 so that the output is the same input in initialization
            for i in range(ch):
                value[i, i, 0, ks // 2] = 1
            w.set_data(ms.Tensor(value, dtype=ms.float32))

            # bias is initialized to zero in layer def
        else:
            raise NotImplementedError


class SpatialUpsample(nn.Cell):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True
            )

    def construct(self, x):
        """
        x: (b c t h w)
        return: (b c t h w)
        """
        B, Ci, T, Hi, Wi = x.shape
        # (b c t h w) -> (b t c h w)
        x = ops.transpose(x, (0, 2, 1, 3, 4))
        # (b t c h w) -> (b*t c h w)
        x = ops.reshape(x, (B * T, Ci, Hi, Wi))

        in_shape = x.shape[-2:]
        out_shape = tuple(2 * x for x in in_shape)
        x = ops.ResizeNearestNeighbor(out_shape)(x)

        if self.with_conv:
            x = self.conv(x)

        _, Co, Ho, Wo = x.shape
        x = ops.reshape(x, (B, T, Co, Ho, Wo))
        x = ops.transpose(x, (0, 2, 1, 3, 4))

        return x


class SpatialDownsample(nn.Cell):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, pad_mode="valid", padding=0, has_bias=True
            )

            # self.pad = nn.Pad(paddings=((0, 0), (0, 0), (0, 1), (0, 1)))

    def construct(self, x):
        # x (b c t h w)
        # TODO: reduce transpose and reshape op
        B, C, T, H, W = x.shape
        x = ops.transpose(x, (0, 2, 1, 3, 4))
        x = ops.reshape(x, (B * T, C, H, W))

        if self.with_conv:
            # x = self.pad(x)
            pad = (0, 1, 0, 1, 0, 0, 0, 0)
            x = mint.nn.functional.pad(x, pad)
            x = self.conv(x)
        else:
            x = ops.AvgPool(kernel_size=2, stride=2)(x)

        # (bt c h w) -> (b c t h w)
        _, Co, Ho, Wo = x.shape
        x = ops.reshape(x, (B, T, Co, Ho, Wo))
        x = ops.transpose(x, (0, 2, 1, 3, 4))

        return x


class TemporalDownsample(nn.Cell):
    def __init__(self, in_channels):
        super().__init__()
        self.ks = 3
        self.ch = in_channels
        self.conv = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=self.ks,
            stride=2,
            pad_mode="valid",
            padding=0,
            has_bias=True,
            bias_init="zeros",
        )
        # tail padding, pad with
        self.time_pad = self.ks - 1
        self.init_weight("centric")

    def init_weight(self, method="mean"):
        if method == "normal":
            # default conv init
            return

        # no way to reserve complete input since stride 2
        w = self.conv.weight
        value = np.zeros(tuple(w.shape))
        if method == "mean":
            # initially, it's a mean filter for temporal downsampling
            for i in range(self.ch):
                value[i, i, 0, :] = 1 / self.ks  # (cout, cin, 1, ks)
        elif method == "centric":
            # a centric filter for temporal downsampling
            for i in range(self.ch):
                value[i, i, 0, self.ks // 2] = 1  # (cout, cin, 1, ks)
        else:
            raise NotImplementedError

        w.set_data(ms.Tensor(value, dtype=ms.float32))

    def construct(self, x):
        # x (b c t h w)

        # -> (bhw c t)
        B, C, T, H, W = x.shape
        x = ops.transpose(x, (0, 3, 4, 1, 2))
        x = ops.reshape(x, (B * H * W, C, T))

        # symmetric padding
        x = symmetric_pad1d(x)

        x = self.conv(x)

        # (bhw c t) -> (b c t h w)
        _, Co, To = x.shape
        x = ops.reshape(x, (B, H, W, Co, To))
        x = ops.transpose(x, (0, 3, 4, 1, 2))

        return x


class TemporalUpsample(nn.Cell):
    def __init__(self, in_channels, manual_pad=True):
        super().__init__()
        self.manual_pad = manual_pad
        # to support danamic shape in graph mode
        if not self.manual_pad:
            self.conv = nn.Conv1d(
                in_channels, in_channels, kernel_size=3, stride=1, pad_mode="same", has_bias=True, bias_init="zeros"
            )
        else:
            self.conv = nn.Conv1d(
                in_channels, in_channels, kernel_size=3, stride=1, pad_mode="valid", has_bias=True, bias_init="zeros"
            )

        # TODO: init conv weight so that it pass in image mode
        self.ch = in_channels
        self.init_weight("centric")

    def init_weight(self, method="centric"):
        if method == "normal":
            return

        # init so that the output is the same as vae2d for image input
        w = self.conv.weight
        value = np.zeros(tuple(w.shape))
        if method == "centric":
            # consider image input, make sure it's the same
            for i in range(self.ch):
                value[i, i, 0, 1] = 1  # (cout, cin, 1, ks)
            w.set_data(ms.Tensor(value, dtype=ms.float32))
        else:
            raise NotImplementedError

    def construct(self, x):
        # x (b c t h w)
        B, C, T0, H, W = x.shape
        x = ops.reshape(x, (B, C, T0, H * W))

        # NOTE: bf16 only support 4D interpolate
        # x = ops.interpolate(x, scale_factor=(2.0, 1.0), mode="nearest")
        out_shape = (T0 * 2, H * W)
        x = ops.ResizeNearestNeighbor(out_shape)(x)

        # x (b c t hw) -> (bhw c t)
        T = T0 * 2
        x = ops.transpose(x, (0, 3, 1, 2))
        x = ops.reshape(x, (B * H * W, C, T))

        if self.manual_pad:
            # work with pad_mode = valid, kernel_size=1
            pad_t_l = ops.zeros((B * H * W, C, 1), x.dtype)
            pad_t_r = ops.zeros((B * H * W, C, 1), x.dtype)
            x = ops.cat([pad_t_l, x, pad_t_r], 2)

        x = self.conv(x)

        # x (bhw c t) -> (b c t h w)
        x = ops.reshape(x, (B, H, W, C, T))
        x = ops.transpose(x, (0, 3, 4, 1, 2))

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
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        assert not conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = Conv2_5d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            pad_mode="pad",
            padding=1,
            has_bias=True,
        )

        if temb_channels > 0:
            self.temb_proj = nn.Dense(temb_channels, out_channels, bias_init="normal")
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = Conv2_5d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            pad_mode="pad",
            padding=1,
            has_bias=True,
        )
        if self.in_channels != self.out_channels:
            # TODO:
            self.nin_shortcut = Conv2_5d(
                in_channels, out_channels, kernel_size=1, stride=1, pad_mode="valid", has_bias=True
            )

    def construct(self, x):
        # x: (b c t h w)
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)

        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class SpatialAttnBlock(nn.Cell):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.bmm = ops.BatchMatMul()
        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, pad_mode="valid", has_bias=True)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, pad_mode="valid", has_bias=True)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, pad_mode="valid", has_bias=True)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, pad_mode="valid", has_bias=True)

        self.hidden_dim = in_channels
        self.scale = ms.Tensor(self.hidden_dim ** (-0.5), dtype=ms.float32)

    def construct(self, x):
        # x (b c t h w)
        h_ = x
        h_ = self.norm(h_)

        # rearrange to spatial sequence (b c t h w) -> (bt c h w)
        T = x.shape[2]
        h_ = ops.transpose(h_, (0, 2, 1, 3, 4))
        h_ = ops.reshape(h_, (h_.shape[0] * h_.shape[1], h_.shape[2], h_.shape[3], h_.shape[4]))

        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = ops.reshape(q, (b, c, h * w))
        q = ops.transpose(q, (0, 2, 1))  # b,hw,c
        k = ops.reshape(k, (b, c, h * w))  # b,c,hw
        w_ = self.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]

        # TODO: use float32 for softmax
        w_ = w_.to(ms.float32) * self.scale
        w_ = ops.Softmax(axis=2)(w_).astype(v.dtype)

        # attend to values
        v = ops.reshape(v, (b, c, h * w))
        w_ = ops.transpose(w_, (0, 2, 1))  # b,hw,hw (first hw of k, second of q)
        h_ = self.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = ops.reshape(h_, (b, c, h, w))

        h_ = self.proj_out(h_)

        # rearrange back
        # -> (b t c h w)
        h_ = ops.reshape(h_, (b // T, T, c, h, w))
        h_ = ops.transpose(h_, (0, 2, 1, 3, 4))

        return x + h_


class SpatialAttnBlockV2(nn.Cell):
    # rewritten to reduce transpose and reshape ops
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.bmm = ops.BatchMatMul()
        self.norm = Normalize(in_channels)
        # TODO: load pretrained weight defined in Conv2d
        self.q = nn.Dense(in_channels, in_channels, has_bias=True)
        self.k = nn.Dense(in_channels, in_channels, has_bias=True)
        self.v = nn.Dense(in_channels, in_channels, has_bias=True)
        self.proj_out = nn.Dense(in_channels, in_channels, has_bias=True)

        self.scale = ms.Tensor(in_channels ** (-0.5), dtype=ms.float32)  # hidden_dim = in_channels

    def construct(self, x):
        # x (b c t h w)
        h_ = x
        h_ = self.norm(h_)

        # rearrange h_ to (b*t h*w c)
        B, C, T, H, W = h_.shape
        h_ = ops.transpose(h_, (0, 2, 3, 4, 1))
        h_ = ops.reshape(h_, (B * T, H * W, C))

        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        k = ops.transpose(k, (0, 2, 1))  # (bt c hw)
        m = self.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]

        m = m.to(ms.float32)
        m = m * self.scale
        attn = ops.softmax(m, axis=-1).astype(v.dtype)  # (bt nq nk)

        # attend to values (nk = nv)
        h_ = self.bmm(attn, v)  # (bt nq c) = (bt hw c)
        h_ = self.proj_out(h_)

        # rearrange back to input shape
        h_ = ops.reshape(h_, (B, T, H, W, C))
        h_ = ops.transpose(h_, (0, 4, 1, 2, 3))

        return x + h_


class TemporalAttnBlock(nn.Cell):
    def __init__(self, in_channels, has_bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.bmm = ops.BatchMatMul()
        # TODO: instead of GroupNorm, LayerNorm is better for tiling
        self.norm = Normalize(in_channels)
        # TODO: compare conv1d with Dense on performance
        self.to_q = nn.Dense(in_channels, in_channels, has_bias=has_bias)
        self.to_k = nn.Dense(in_channels, in_channels, has_bias=has_bias)
        self.to_v = nn.Dense(in_channels, in_channels, has_bias=has_bias)
        self.proj_out = nn.Dense(in_channels, in_channels, has_bias=has_bias)

        self.scale = ms.Tensor(in_channels ** (-0.5), dtype=ms.float32)  # hidden_dim = in_channels

    def construct(self, x):
        # x (b c t h w)
        h_ = x
        # TODO: use LayerNorm for (B N C) instead of GroupNorm for (B C N)
        h_ = self.norm(h_)

        # (b c t h w) -> (b*h*w t c) = (B S H)
        B, C, T, H, W = h_.shape
        h_ = ops.transpose(h_, (0, 3, 4, 2, 1))
        h_ = ops.reshape(h_, (B * H * W, T, C))

        # projection
        q = self.to_q(h_)  # (bhw t c)
        k = self.to_k(h_)  # (bhw t c) = (bhw nk c)
        v = self.to_v(h_)  # (bhw t c) = (bhw nv c)

        # compute attention
        # (B S H) -> (B H S)
        k = ops.transpose(k, (0, 2, 1))  # (bhw c t)
        m = self.bmm(q, k)  # bhw, t, t = (bhw nq nk)

        m = m.to(ms.float32)
        m = m * self.scale
        attn = ops.softmax(m, axis=-1).astype(v.dtype)

        # attend to values
        h_ = self.bmm(attn, v)  # (bhw nq c)
        h_ = self.proj_out(h_)

        # rearrange back to input shape
        h_ = ops.reshape(h_, (B, H, W, T, C))
        h_ = ops.transpose(h_, (0, 4, 3, 1, 2))

        return x + h_


def make_attn(in_channels, attn_type="vanilla"):
    # assert attn_type in ["vanilla", "vanilla3D"], f"attn_type {attn_type} not supported"
    # _logger.debug(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return nn.SequentialCell(
            SpatialAttnBlock(in_channels),
            TemporalAttnBlock(in_channels),
        )
    elif attn_type == "spat_only":
        # to ensure naming consistency
        return nn.SequentialCell(
            SpatialAttnBlock(in_channels),
        )
    else:
        raise NotImplementedError


# used in vae
class Encoder(nn.Cell):
    # @ms.lazy_inline()
    def __init__(
        self,
        ch=128,
        out_ch=3,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        attn_resolutions=[],
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=3,
        resolution=256,
        z_channels=4,  # TODO: use 16
        double_z=True,
        use_linear_attn=False,
        attn_type="vanilla",
        temporal_downsample_level=(0, 1, 2),  # same as spatial
        **kwargs,
    ):
        super().__init__()
        # if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.temporal_downsample_level = temporal_downsample_level

        # downsampling
        self.conv_in = Conv2_5d(
            in_channels,
            ch,
            kernel_size=3,
            stride=1,
            pad_mode="pad",
            padding=1,
            has_bias=True,
        )

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
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Cell()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample_spat = SpatialDownsample(block_in, resamp_with_conv)
            else:
                down.downsample_spat = nn.Identity()

            if i_level in self.temporal_downsample_level:
                down.downsample_temp = TemporalDownsample(block_in)
            else:
                down.downsample_temp = nn.Identity()

            curr_res = curr_res // 2
            down.update_parameters_name(prefix=self.param_prefix + f"down.{i_level}.")
            self.down.append(down)

        # middle
        self.mid = nn.Cell()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = Conv2_5d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            pad_mode="pad",
            padding=1,
            has_bias=True,
        )

    def construct(self, x):
        """
        Args:
            x: (b c t h w)
        Returns:
            (b c t h w)
        """
        # spatial and temporal conv
        hs = self.conv_in(x)

        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs = h
            # if i_level != self.num_resolutions - 1:
            hs = self.down[i_level].downsample_spat(hs)
            hs = self.down[i_level].downsample_temp(hs)

        # middle
        h = hs
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        return h


class Decoder(nn.Cell):
    # @ms.lazy_inline()
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
        temporal_upsample_level=(1, 2, 3),  # same as spatial
        **ignorekwargs,
    ):
        super().__init__()
        # if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.temporal_upsample_level = temporal_upsample_level

        # compute in_ch_mult, block_in and curr_res at lowest res
        # in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        _logger.debug("Working with z of shape {} = {} dimensions.".format(self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = Conv2_5d(z_channels, block_in, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True)

        # middle
        self.mid = nn.Cell()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)

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
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Cell()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample_spat = SpatialUpsample(block_in, resamp_with_conv)
            else:
                up.upsample_spat = nn.Identity()

            if i_level in self.temporal_upsample_level:
                up.upsample_temp = TemporalUpsample(block_in)
            else:
                up.upsample_temp = nn.Identity()

            curr_res = curr_res * 2
            up.update_parameters_name(prefix=self.param_prefix + f"up.{i_level}.")
            if len(self.up) != 0:
                self.up.insert(0, up)
            else:
                self.up.append(up)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = Conv2_5d(block_in, out_ch, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True)

    def construct(self, z):
        """
        Args:
            x: (b c t h w)
        Returns:
            (b c t h w)
        """

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        i_level = self.num_resolutions
        while i_level > 0:
            i_level -= 1
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)

            h = self.up[i_level].upsample_spat(h)
            h = self.up[i_level].upsample_temp(h)

        # end
        if self.give_pre_end:
            return h
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = ops.tanh(h)
        return h
