# reference to https://github.com/Stability-AI/generative-models

from typing import Callable

from gm.modules.attention import FLASH_IS_AVAILABLE, FlashAttention, LinearAttention
from gm.modules.transformers import scaled_dot_product_attention

import mindspore as ms
import mindspore.numpy as mnp
from mindspore import Tensor, nn, ops


def nonlinearity(x):
    # swish
    return x * ops.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Cell):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True
            )

    def construct(self, x):
        # x = ops.interpolate(x, scale_factor=2.0, mode="nearest")
        # x = ops.interpolate(x, size=(x.shape[2] * 2, x.shape[3] * 2), mode="nearest")
        x = ops.ResizeNearestNeighbor(size=(x.shape[2] * 2, x.shape[3] * 2))(x)
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Cell):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in mindspore conv, must do it ourselves
            self.conv = nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0, pad_mode="pad", has_bias=True
            )

    def construct(self, x):
        if self.with_conv:
            # x = mnp.pad(x, ((0, 0), (0, 0), (0, 1), (0, 1)))
            b, c, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[-1]
            concat_h = ops.ones((b,c,h,1), x.dtype)
            concat_w = ops.ones((b,c,1,w+1), x.dtype)

            x = ops.Concat(3)([x, concat_h])
            x = ops.Concat(2)([x, concat_w])
            
            x = self.conv(x)
        else:
            x = ops.avg_pool2d(x, kernel_size=2, stride=2)
        return x


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
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True
        )
        if temb_channels > 0:
            self.temb_proj = nn.Dense(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True
                )
            else:
                self.nin_shortcut = nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0, pad_mode="pad", has_bias=True
                )

    def construct(self, x, temb=None):
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


class AttnBlock(nn.Cell):
    def __init__(self, in_channels, attn_dtype=None):
        super().__init__()
        self.in_channels = in_channels
        self.attn_dtype = attn_dtype

        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, pad_mode="valid", has_bias=True)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, pad_mode="valid", has_bias=True)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, pad_mode="valid", has_bias=True)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, pad_mode="valid", has_bias=True)

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape

        # q, k, v = map(
        #     lambda x: rearrange(x, "b c h w -> b 1 (h w) c").contiguous(), (q, k, v)
        # )
        b_q, c_q, _, _ = q.shape
        q = q.view(b_q, 1, c_q, -1).swapaxes(-1, -2)  # b c h w -> b 1 c (h w) -> b 1 (h w) c
        b_k, c_k, _, _ = k.shape
        k = k.view(b_k, 1, c_k, -1).swapaxes(-1, -2)  # b c h w -> b 1 c (h w) -> b 1 (h w) c
        b_v, c_v, _, _ = v.shape
        v = v.view(b_v, 1, c_v, -1).swapaxes(-1, -2)  # b c h w -> b 1 c (h w) -> b 1 (h w) c

        # compute attention
        h_ = scaled_dot_product_attention(q, k, v, dtype=self.attn_dtype)  # scale is dim ** -0.5 per default

        # out = rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)
        out = h_.swapaxes(-1, -2).view(b, c, h, w)  # b 1 (h w) c -> b 1 c (h w) -> b c h w

        return out

    def construct(self, x, **kwargs):
        h_ = x
        h_ = self.attention(h_)
        h_ = self.proj_out(h_)
        return x + h_


class MemoryEfficientAttnBlock(nn.Cell):
    def __init__(self, in_channels, attn_dtype=None):
        super().__init__()

        assert FLASH_IS_AVAILABLE

        self.in_channels = in_channels
        self.attn_dtype = attn_dtype

        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, pad_mode="valid", has_bias=True)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, pad_mode="valid", has_bias=True)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, pad_mode="valid", has_bias=True)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, pad_mode="valid", has_bias=True)

        self.flash_attention = FlashAttention(input_layout="BNSD")

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape

        # q, k, v = map(
        #     lambda x: rearrange(x, "b c h w -> b 1 (h w) c").contiguous(), (q, k, v)
        # )
        b_q, c_q, _, _ = q.shape
        q = q.view(b_q, 1, c_q, -1).swapaxes(-1, -2)  # b c h w -> b 1 c (h w) -> b 1 (h w) c
        b_k, c_k, _, _ = k.shape
        k = k.view(b_k, 1, c_k, -1).swapaxes(-1, -2)  # b c h w -> b 1 c (h w) -> b 1 (h w) c
        b_v, c_v, _, _ = v.shape
        v = v.view(b_v, 1, c_v, -1).swapaxes(-1, -2)  # b c h w -> b 1 c (h w) -> b 1 (h w) c

        q_n, k_n = q.shape[-2], k.shape[-2]
        head_dim = q.shape[-1]
        if q_n % 16 == 0 and k_n % 16 == 0 and head_dim <= 256:
            h_ = self.flash_attention(q, k, v, None, None, None, None, None)[3]
        else:
            h_ = scaled_dot_product_attention(q, k, v, dtype=self.attn_dtype)  # scale is dim_head ** -0.5 per default

        # out = rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)
        out = h_.swapaxes(-1, -2).view(b, c, h, w)  # b 1 (h w) c -> b 1 c (h w) -> b c h w

        return out

    def construct(self, x, **kwargs):
        h_ = x
        h_ = self.attention(h_)
        h_ = self.proj_out(h_)
        return x + h_


class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""

    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)


def make_attn(in_channels, attn_type="vanilla", attn_kwargs=None, attn_dtype=None):
    assert attn_type in ["vanilla", "flash-attention", "linear", "none"], f"attn_type {attn_type} unknown"

    if attn_type == "flash-attention" and not FLASH_IS_AVAILABLE:
        print(
            f"Attention mode '{attn_type}' is not available. Falling back to native attention. "
            f"This is not a problem in MindSpore >= 2.0.1 on Ascend devices; "
            f"FYI, you are running with MindSpore version {ms.__version__}"
        )
        attn_type = "vanilla"

    if attn_type == "vanilla":
        assert attn_kwargs is None
        return AttnBlock(in_channels, attn_dtype=attn_dtype)
    elif attn_type == "flash-attention":
        # print(f"building FlashAttention with {in_channels} in_channels...")
        # return MemoryEfficientAttnBlock(in_channels, attn_dtype=attn_dtype)
        raise NotImplementedError
    elif attn_type == "none":
        return nn.Identity()
    elif attn_type == "linear":
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
        encoder_attn_dtype=None,
        **ignore_kwargs,
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        if encoder_attn_dtype:
            encoder_attn_dtype = ms.float16 if encoder_attn_dtype in ("fp16", "float16") else None

        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = nn.Conv2d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True
        )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.CellList()
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
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type, attn_dtype=encoder_attn_dtype))

            class DownCell(nn.Cell):
                def __init__(self, block, attn, downsample=None):
                    super().__init__()
                    self.block = block
                    self.attn = attn
                    self.downsample = downsample

            downsample = None
            if i_level != self.num_resolutions - 1:
                downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(DownCell(block, attn, downsample))

        # middle
        self.mid = nn.Cell()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type, attn_dtype=encoder_attn_dtype)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            pad_mode="pad",
            has_bias=True,
        )

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
        attn_type="vanilla",
        decoder_attn_dtype=None,
        **ignorekwargs,
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        if decoder_attn_dtype:
            decoder_attn_dtype = ms.float16 if decoder_attn_dtype in ("fp16", "float16") else None

        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        # in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        # disable print
        # print("Working with z of shape {} = {} dimensions.".format(self.z_shape, np.prod(self.z_shape)))

        make_attn_cls = self._make_attn()
        make_resblock_cls = self._make_resblock()
        make_conv_cls = self._make_conv()
        # z to block_in
        self.conv_in = nn.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True
        )

        # middle
        self.mid = nn.Cell()
        self.mid.block_1 = make_resblock_cls(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = make_attn_cls(block_in, attn_type=attn_type, attn_dtype=decoder_attn_dtype)
        self.mid.block_2 = make_resblock_cls(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # upsampling
        ups = []
        for i_level in reversed(range(self.num_resolutions)):
            block, attn = [], []
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    make_resblock_cls(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn_cls(block_in, attn_type=attn_type, attn_dtype=decoder_attn_dtype))

            block = nn.CellList(block)
            attn = nn.CellList(attn)

            class UpCell(nn.Cell):
                def __init__(self, block, attn, upsample=None):
                    super().__init__()
                    self.block = block
                    self.attn = attn
                    self.upsample = upsample

            upsample = None
            if i_level != 0:
                upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            ups.insert(0, UpCell(block, attn, upsample))  # prepend to get consistent order
        self.up = nn.CellList(ups)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = make_conv_cls(
            block_in, out_ch, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True
        )

    def _make_attn(self) -> Callable:
        return make_attn

    def _make_resblock(self) -> Callable:
        return ResnetBlock

    def _make_conv(self) -> Callable:
        return nn.Conv2d

    def get_last_layer(self, **kwargs):
        return self.conv_out.weight

    def construct(self, z, **kwargs):
        # assert z.shape[1:] == self.z_shape[1:]

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb, **kwargs)
        h = self.mid.attn_1(h, **kwargs)
        h = self.mid.block_2(h, temb, **kwargs)

        # upsampling
        # for i_level in reversed(range(self.num_resolutions)):  # not support with mindspore jit compile
        for i_level in range(self.num_resolutions - 1, -1, -1):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb, **kwargs)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, **kwargs)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h, **kwargs)
        if self.tanh_out:
            h = ops.tanh(h)
        return h
