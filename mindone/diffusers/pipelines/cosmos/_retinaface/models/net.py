# MindSpore adaptation from `retinaface.model.net` from retinaface-py pkg
# This file remains under the original license.


from mindspore import mint, nn


def conv_bn(inp, oup, stride=1, leaky=0):
    return nn.SequentialCell(
        mint.nn.Conv2d(inp, oup, 3, stride, 1, bias=False), mint.nn.BatchNorm2d(oup), nn.LeakyReLU(alpha=leaky)
    )


def conv_bn_no_relu(inp, oup, stride):
    return nn.SequentialCell(
        mint.nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        mint.nn.BatchNorm2d(oup),
    )


def conv_bn1X1(inp, oup, stride, leaky=0):
    return nn.SequentialCell(
        mint.nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False), mint.nn.BatchNorm2d(oup), nn.LeakyReLU(alpha=leaky)
    )


def conv_dw(inp, oup, stride, leaky=0.1):
    return nn.SequentialCell(
        mint.nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        mint.nn.BatchNorm2d(inp),
        nn.LeakyReLU(alpha=leaky),
        mint.nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        mint.nn.BatchNorm2d(oup),
        nn.LeakyReLU(alpha=leaky),
    )


class SSH(nn.Cell):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if out_channel <= 64:
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel // 2, stride=1)

        self.conv5X5_1 = conv_bn(in_channel, out_channel // 4, stride=1, leaky=leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)

        self.conv7X7_2 = conv_bn(out_channel // 4, out_channel // 4, stride=1, leaky=leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)

    def construct(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = mint.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = mint.functional.relu(out)
        return out


class FPN(nn.Cell):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        leaky = 0
        if out_channels <= 64:
            leaky = 0.1
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride=1, leaky=leaky)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride=1, leaky=leaky)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride=1, leaky=leaky)

        self.merge1 = conv_bn(out_channels, out_channels, leaky=leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky=leaky)

    def construct(self, input):
        # names = list(input.keys())
        input = list(input.values())

        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])

        up3 = mint.functional.interpolate(output3, size=[output2.shape[2], output2.shape[3]], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = mint.functional.interpolate(output2, size=[output1.shape[2], output1.shape[3]], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3]
        return out
