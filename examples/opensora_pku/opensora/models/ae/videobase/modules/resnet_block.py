import mindspore as ms
from mindspore import nn, ops

from .conv import CausalConv3d
from .ops import nonlinearity


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

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True
        ).to_float(dtype)
        if temb_channels > 0:
            self.temb_proj = nn.Dense(temb_channels, out_channels, bias_init="normal").to_float(dtype)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
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


class ResnetBlock3D(nn.Cell):
    def __init__(
        self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, dtype=ms.float32, upcast_sigmoid=False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut
        self.upcast_sigmoid = upcast_sigmoid

        # FIXME: GroupNorm precision mismatch with PT.
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = CausalConv3d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
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
        h = nonlinearity(h, self.upcast_sigmoid)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nonlinearity(h, self.upcast_sigmoid)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


# pku opensora v1.1
class ResnetBlock2D(nn.Cell):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
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

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True
        ).to_float(dtype)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
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
        # rearrange in
        F = x.shape[-3]
        x = self.rearrange_in(x)

        h = x
        h = self.norm1(h)
        h = nonlinearity(h, upcast=self.upcast_sigmoid)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h, upcast=self.upcast_sigmoid)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        x = x + h
        # rearrange out
        x = self.rearrange_out(x, F)
        return x
