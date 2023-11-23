from mindspore import nn


class Upsample(nn.Cell):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.Conv2dTranspose(
            input_dim, output_dim, kernel_size=kernel, stride=stride, has_bias=True
        )

    def construct(self, x):
        return self.upsample(x)


class ResidualConv(nn.Cell):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.SequentialCell(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, pad_mode='pad', padding=padding, has_bias=True
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, pad_mode='pad', padding=1, has_bias=True),
        )
        self.conv_skip = nn.SequentialCell(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, pad_mode='pad', padding=1, has_bias=True),
            nn.BatchNorm2d(output_dim),
        )

    def construct(self, x):
        return self.conv_block(x) + self.conv_skip(x)
