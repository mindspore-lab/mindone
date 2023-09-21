# pytorch_diffusion + derived encoder decoder
import logging
import math

import numpy as np
from audioldm.latent_diffusion.attention import LinearAttention
from audioldm.latent_diffusion.ldm_util import is_old_ms_version

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

_logger = logging.getLogger(__name__)


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = ops.exp(ops.arange(half_dim, dtype=ms.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = ops.cat([ops.sin(emb), ops.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = ops.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x * ops.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


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
        x = ops.interpolate(x, size=out_shape, mode="nearest")

        if self.with_conv:
            x = self.conv(x)
        return x


class UpsampleTimeStride4(nn.Cell):
    def __init__(self, in_channels, with_conv, dtype=ms.float32):
        super().__init__()
        self.dtype = dtype
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=1, padding=2).to_float(self.dtype)

    def forward(self, x):
        in_shape = x.shape[-2:]
        out_shape = tuple(4 * in_shape[0], 2 * in_shape[1])
        x = ops.interpolate(x, size=out_shape, mode="nearest")
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


class DownsampleTimeStride4(nn.Cell):
    def __init__(self, in_channels, with_conv, dtype=ms.float32):
        super().__init__()
        self.dtype = dtype
        self.with_conv = with_conv
        if self.with_conv:
            # Do time downsampling here
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=(4, 2), padding=1).to_float(
                self.dtype
            )

    def construct(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = ops.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = ops.avg_pool2d(x, kernel_size=(4, 2), stride=(4, 2))
        return x


class ResnetBlock(nn.Cell):
    def __init__(
        self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, temb_channels=512, dtype=ms.float32
    ):
        super().__init__()
        self.dtype = dtype
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True
        ).to_float(dtype)
        if temb_channels > 0:
            self.temb_proj = nn.Dense(temb_channels, out_channels, bias_init="normal").to_float(dtype)
        self.norm2 = Normalize(out_channels)
        if is_old_ms_version():
            self.dropout = nn.Dropout(1.0 - dropout)
        else:
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


class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""

    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)


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


def make_attn(in_channels, attn_type="vanilla", dtype=ms.float32):
    assert attn_type in ["vanilla", "linear", "none"], f"attn_type {attn_type} unknown"
    # print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels, dtype=dtype)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        return LinAttnBlock(in_channels)


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
        downsample_time_stride4_levels=[],
        **ignore_kwargs,
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.dtype = dtype
        self.downsample_time_stride4_levels = downsample_time_stride4_levels

        if len(self.downsample_time_stride4_levels) > 0:
            assert max(self.downsample_time_stride4_levels) < self.num_resolutions, (
                "The level to perform downsample 4 operation need to be smaller than the total resolution number %s"
                % str(self.num_resolutions)
            )

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
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type, dtype=self.dtype))
            down = nn.Cell()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                if i_level in self.downsample_time_stride4_levels:
                    down.downsample = DownsampleTimeStride4(block_in, resamp_with_conv, dtype=self.dtype)
                else:
                    down.downsample = Downsample(block_in, resamp_with_conv, dtype=self.dtype)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Cell()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout, dtype=self.dtype
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type, dtype=self.dtype)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout, dtype=self.dtype
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
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


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
        downsample_time_stride4_levels=[],
        attn_type="vanilla",
        dtype=ms.float32,
        **ignorekwargs,
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.dtype = dtype
        self.downsample_time_stride4_levels = downsample_time_stride4_levels

        if len(self.downsample_time_stride4_levels) > 0:
            assert max(self.downsample_time_stride4_levels) < self.num_resolutions, (
                "The level to perform downsample 4 operation need to be smaller than the total resolution number %s"
                % str(self.num_resolutions)
            )

        # compute in_ch_mult, block_in and curr_res at lowest res
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
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Cell()
            up.block = block
            up.attn = attn
            if i_level != 0:
                if i_level - 1 in self.downsample_time_stride4_levels:
                    up.upsample = UpsampleTimeStride4(block_in, resamp_with_conv)
                else:
                    up.upsample = Upsample(block_in, resamp_with_conv)
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
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

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
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = ops.tanh(h)
        return h
