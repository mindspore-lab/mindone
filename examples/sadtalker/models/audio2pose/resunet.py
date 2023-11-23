from mindspore import nn, ops
from models.audio2pose.networks import ResidualConv, Upsample


class ResUNet(nn.Cell):
    def __init__(self, channel=1, filters=[32, 64, 128, 256]):
        super(ResUNet, self).__init__()

        self.input_layer = nn.SequentialCell(
            nn.Conv2d(channel, filters[0], kernel_size=3, pad_mode='pad', padding=1, has_bias=True),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, pad_mode='pad', padding=1, has_bias=True),
        )
        self.input_skip = nn.SequentialCell(
            nn.Conv2d(channel, filters[0], kernel_size=3, pad_mode='pad', padding=1, has_bias=True)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], stride=(2,1), padding=1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], stride=(2,1), padding=1)

        self.bridge = ResidualConv(filters[2], filters[3], stride=(2,1), padding=1)

        self.upsample_1 = Upsample(filters[3], filters[3], kernel=(2,1), stride=(2,1))
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], stride=1, padding=1)

        self.upsample_2 = Upsample(filters[2], filters[2], kernel=(2,1), stride=(2,1))
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], stride=1, padding=1)

        self.upsample_3 = Upsample(filters[1], filters[1], kernel=(2,1), stride=(2,1))
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], stride=1, padding=1)

        self.output_layer = nn.SequentialCell(
            nn.Conv2d(filters[0], 1, 1, 1, has_bias=True),
            nn.Sigmoid(),
        )

    def construct(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        # Bridge
        x4 = self.bridge(x3)

        # Decode
        x4 = self.upsample_1(x4)
        x5 = ops.cat([x4, x3], axis=1)

        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = ops.cat([x6, x2], axis=1)

        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = ops.cat([x8, x1], axis=1)

        x10 = self.up_residual_conv3(x9)

        output = self.output_layer(x10)

        return output
