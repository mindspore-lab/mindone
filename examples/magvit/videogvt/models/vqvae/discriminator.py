"""3D StyleGAN discriminator."""

import math

import ml_collections
import numpy as np

import mindspore as ms
from mindspore import nn, ops

from .model_utils import GroupNormExtend


def get_pad_layer(pad_type):
    if pad_type in ["refl", "reflect"]:
        PadLayer = nn.ReflectionPad3d
    elif pad_type in ["repl", "replicate"]:
        PadLayer = nn.ReplicationPad3d
    elif pad_type == "zero":
        PadLayer = nn.ConstantPad3d
    else:
        print("Pad type [%s] not recognized" % pad_type)
    return PadLayer


class BlurPool3d(nn.Cell):
    def __init__(
        self,
        channels,
        pad_type="reflect",
        filt_size=4,
        stride=2,
        pad_off=0,
        dtype=ms.float32,
    ):
        super(BlurPool3d, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [
            int(1.0 * (filt_size - 1) / 2),
            int(np.ceil(1.0 * (filt_size - 1) / 2)),
            int(1.0 * (filt_size - 1) / 2),
            int(np.ceil(1.0 * (filt_size - 1) / 2)),
            int(1.0 * (filt_size - 1) / 2),
            int(np.ceil(1.0 * (filt_size - 1) / 2)),
        ]
        self.pad_sizes = tuple([pad_size + pad_off for pad_size in self.pad_sizes])
        self.stride = stride
        self.off = int((self.stride - 1) / 2.0)
        self.channels = channels
        self.dtype = dtype

        if self.filt_size == 1:
            a = np.array(
                [
                    1.0,
                ]
            )
        elif self.filt_size == 2:
            a = np.array([1.0, 1.0])
        elif self.filt_size == 3:
            a = np.array([1.0, 2.0, 1.0])
        elif self.filt_size == 4:
            a = np.array([1.0, 3.0, 3.0, 1.0])
        elif self.filt_size == 5:
            a = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
        elif self.filt_size == 6:
            a = np.array([1.0, 5.0, 10.0, 10.0, 5.0, 1.0])
        elif self.filt_size == 7:
            a = np.array([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0])

        filt = ms.Tensor(
            np.repeat(np.expand_dims(a[:, None] * a[None, :], 0), self.filt_size, axis=0),
            self.dtype,
        )
        filt = filt / ops.sum(filt)
        filt = filt.unsqueeze(0).unsqueeze(0)
        filt = filt.repeat(self.channels, 0).repeat(self.channels, 1)
        self.filt = ms.Parameter(filt, requires_grad=False)
        self.pad = (
            get_pad_layer(pad_type)(self.pad_sizes)
            if pad_type != "zero"
            else get_pad_layer(pad_type)(self.pad_sizes, 0)
        )

    def construct(self, inp):
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, :: self.stride, :: self.stride, :: self.stride]
            else:
                return self.pad(inp)[:, :, :: self.stride, :: self.stride, :: self.stride]
        else:
            return ops.conv3d(self.pad(inp), self.filt, stride=self.stride, groups=1)


class ResBlockDown(nn.Cell):
    """3D StyleGAN ResBlock for D."""

    def __init__(
        self,
        in_channels,
        out_channels=None,
        dropout=0.1,
        dtype=ms.float32,
    ):
        super(ResBlockDown, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels

        self.conv1 = nn.Conv3d(self.in_channels, self.out_channels, (3, 3, 3)).to_float(dtype)
        self.norm1 = GroupNormExtend(
            num_groups=32,
            num_channels=self.out_channels,
            eps=1e-5,
            affine=True,
            dtype=dtype,
        )
        self.activation1 = nn.LeakyReLU()
        self.conv2 = nn.Conv3d(self.out_channels, self.out_channels, (3, 3, 3)).to_float(dtype)
        self.norm2 = GroupNormExtend(
            num_groups=32,
            num_channels=self.out_channels,
            eps=1e-5,
            affine=True,
            dtype=dtype,
        )
        self.activation2 = nn.LeakyReLU()
        # self.dropout = nn.Dropout(p=dropout)

        self.conv_shortcut = nn.Conv3d(self.in_channels, self.out_channels, (1, 1, 1), has_bias=False).to_float(dtype)

        self.blurpool1 = BlurPool3d(self.out_channels, pad_type="zero")
        self.blurpool2 = BlurPool3d(self.in_channels, pad_type="zero")

    def construct(self, x):
        h = x
        h = self.conv1(h)
        h = self.norm1(h)
        h = self.activation1(h)

        # h = ops.AvgPool3D(strides=(2, 2, 2))(h)
        h = self.blurpool1(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.activation2(h)

        # x = ops.AvgPool3D(strides=(2, 2, 2))(x)
        x = self.blurpool2(x)

        x = self.conv_shortcut(x)

        out = (x + h) / ops.sqrt(ms.Tensor(2, ms.float32))
        return out


class StyleGANDiscriminator(nn.Cell):
    """StyleGAN Discriminator."""

    def __init__(
        self,
        config: ml_collections.ConfigDict,
        height: int,
        width: int,
        depth: int,
        dtype: ms.dtype = ms.float32,
    ):
        super().__init__()
        self.config = config
        self.in_channles = 3
        self.filters = self.config.filters
        self.channel_multipliers = self.config.channel_multipliers

        self.conv_in = nn.Conv3d(self.in_channles, self.filters, kernel_size=(3, 3, 3)).to_float(dtype)
        # self.activation1 = nn.LeakyReLU()
        self.resnet_stack = nn.SequentialCell()

        num_blocks = len(self.channel_multipliers)
        sampling_rate = math.pow(2, num_blocks)
        for i in range(num_blocks):
            filters = self.filters * self.channel_multipliers[i]

            if i == 0:
                dim_in = self.filters
            else:
                dim_in = self.filters * self.channel_multipliers[i - 1]

            self.resnet_stack.append(ResBlockDown(dim_in, filters, dtype=dtype))

        dim_out = self.filters * self.channel_multipliers[-1]
        self.norm2 = GroupNormExtend(num_groups=32, num_channels=dim_out, eps=1e-5, affine=True, dtype=dtype)
        self.conv_out = nn.Conv3d(dim_out, dim_out, (3, 3, 3)).to_float(dtype)
        # self.activation2 = nn.LeakyReLU()

        dim_dense = int(
            dim_out * max(1, height // sampling_rate) * max(1, width // sampling_rate) * max(1, depth // sampling_rate)
        )

        self.linear1 = nn.Dense(dim_dense, 512, dtype=dtype)
        self.linear2 = nn.Dense(512, 1, dtype=dtype)

    def construct(self, x):
        # x = self.norm(x)
        x = self.conv_in(x)
        x = ops.elu(x)
        x = self.resnet_stack(x)
        x = self.conv_out(x)
        x = self.norm2(x)
        x = ops.elu(x)
        x = ops.reshape(x, (x.shape[0], -1))
        x = self.linear1(x)
        x = ops.elu(x)
        x = self.linear2(x)
        return x
