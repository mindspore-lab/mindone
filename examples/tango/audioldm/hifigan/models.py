import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.nn import Conv1d, Conv1dTranspose

LRELU_SLOPE = 0.1


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock(nn.Cell):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock, self).__init__()
        self.h = h
        self.convs1 = nn.CellList(
            [
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[0],
                    padding=get_padding(kernel_size, dilation[0]),
                    pad_mode="pad",
                    has_bias=True,
                ),
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[1],
                    padding=get_padding(kernel_size, dilation[1]),
                    pad_mode="pad",
                    has_bias=True,
                ),
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[2],
                    padding=get_padding(kernel_size, dilation[2]),
                    pad_mode="pad",
                    has_bias=True,
                ),
            ]
        )

        self.convs2 = nn.CellList(
            [
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1),
                    pad_mode="pad",
                    has_bias=True,
                ),
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1),
                    pad_mode="pad",
                    has_bias=True,
                ),
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1),
                    pad_mode="pad",
                    has_bias=True,
                ),
            ]
        )
        self.leaky_relu = nn.LeakyReLU(LRELU_SLOPE)

    def construct(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = self.leaky_relu(x)
            xt = c1(xt)
            xt = self.leaky_relu(xt)
            xt = c2(xt)
            x = xt + x
        return x


class Generator(nn.Cell):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3, pad_mode="pad", has_bias=True)
        resblock = ResBlock

        self.ups = nn.CellList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(
                Conv1dTranspose(
                    h.upsample_initial_channel // (2**i),
                    h.upsample_initial_channel // (2 ** (i + 1)),
                    k,
                    u,
                    padding=(k - u) // 2,
                    pad_mode="pad",
                    has_bias=True,
                )
            )

        self.resblocks = nn.CellList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, pad_mode="pad", has_bias=True)
        self.leaky_relu = nn.LeakyReLU(LRELU_SLOPE)
        self.leaky_relu_default = nn.LeakyReLU(0.01)

    def construct(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = self.leaky_relu(x)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = self.leaky_relu_default(x)
        x = self.conv_post(x)
        x = ops.tanh(x)

        return x
