import mindspore
from mindspore import nn, ops, Tensor
from models.face3d.facexlib.resnet import ResNet, BasicBlock, Bottleneck
from typing import Any, Callable, List, Optional, Type, Union


def conv_bn(inp, oup, stride=1, leaky=0):
    return nn.SequentialCell(
        nn.Conv2d(inp, oup, 3, stride, pad_mode='pad', padding=1,
                  has_bias=False), nn.BatchNorm2d(oup),
        nn.LeakyReLU(alpha=leaky))


def conv_bn_no_relu(inp, oup, stride):
    return nn.SequentialCell(
        nn.Conv2d(inp, oup, 3, stride, pad_mode='pad',
                  padding=1, has_bias=False),
        nn.BatchNorm2d(oup),
    )


def conv_bn1X1(inp, oup, stride, leaky=0):
    return nn.SequentialCell(
        nn.Conv2d(inp, oup, 1, stride, padding=0,
                  has_bias=False), nn.BatchNorm2d(oup),
        nn.LeakyReLU(alpha=leaky))


def conv_dw(inp, oup, stride, leaky=0.1):
    return nn.SequentialCell(
        nn.Conv2d(inp, inp, 3, stride, pad_mode='pad',
                  padding=1, group=inp, has_bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(alpha=leaky),
        nn.Conv2d(inp, oup, 1, 1, padding=0, has_bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(alpha=leaky),
    )


def retinafacebody_r50():
    model = RetinaFaceBody(Bottleneck, [3, 4, 6, 3])
    return model


def retinafacebody_m25():
    raise NotImplementedError


class RetinaFaceBody(ResNet):
    """hardcode for RetinaFace.body with ResNet
    """

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        use_last_fc: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Cell]] = None
    ):

        super().__init__(block, layers, num_classes, zero_init_residual, use_last_fc,
                         groups, width_per_group, replace_stride_with_dilation, norm_layer)

        del self.avgpool
        if self.use_last_fc:
            del self.fc

    def _forward_impl(self, x: Tensor) -> tuple:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return (x2, x3, x4)

    def construct(self, x: Tensor) -> tuple:
        return self._forward_impl(x)


class SSH(nn.Cell):

    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if (out_channel <= 64):
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel // 2, stride=1)

        self.conv5X5_1 = conv_bn(
            in_channel, out_channel // 4, stride=1, leaky=leaky)
        self.conv5X5_2 = conv_bn_no_relu(
            out_channel // 4, out_channel // 4, stride=1)

        self.conv7X7_2 = conv_bn(
            out_channel // 4, out_channel // 4, stride=1, leaky=leaky)
        self.conv7x7_3 = conv_bn_no_relu(
            out_channel // 4, out_channel // 4, stride=1)

    def construct(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = ops.cat([conv3X3, conv5X5, conv7X7], axis=1)
        out = ops.relu(out)
        return out


class FPN(nn.Cell):

    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1
        self.output1 = conv_bn1X1(
            in_channels_list[0], out_channels, stride=1, leaky=leaky)
        self.output2 = conv_bn1X1(
            in_channels_list[1], out_channels, stride=1, leaky=leaky)
        self.output3 = conv_bn1X1(
            in_channels_list[2], out_channels, stride=1, leaky=leaky)

        self.merge1 = conv_bn(out_channels, out_channels, leaky=leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky=leaky)

    def construct(self, input):
        # names = list(input.keys())
        # input = list(input.values())

        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])

        up3 = ops.interpolate(
            output3, size=[output2.shape[2], output2.shape[3]], mode='nearest')
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = ops.interpolate(
            output2, size=[output1.shape[2], output1.shape[3]], mode='nearest')
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3]
        return out


class MobileNetV1(nn.Cell):

    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.SequentialCell(
            conv_bn(3, 8, 2, leaky=0.1),  # 3
            conv_dw(8, 16, 1),  # 7
            conv_dw(16, 32, 2),  # 11
            conv_dw(32, 32, 1),  # 19
            conv_dw(32, 64, 2),  # 27
            conv_dw(64, 64, 1),  # 43
        )
        self.stage2 = nn.SequentialCell(
            conv_dw(64, 128, 2),  # 43 + 16 = 59
            conv_dw(128, 128, 1),  # 59 + 32 = 91
            conv_dw(128, 128, 1),  # 91 + 32 = 123
            conv_dw(128, 128, 1),  # 123 + 32 = 155
            conv_dw(128, 128, 1),  # 155 + 32 = 187
            conv_dw(128, 128, 1),  # 187 + 32 = 219
        )
        self.stage3 = nn.SequentialCell(
            conv_dw(128, 256, 2),  # 219 +3 2 = 241
            conv_dw(256, 256, 1),  # 241 + 64 = 301
        )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Dense(256, 1000)

    def construct(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x


class ClassHead(nn.Cell):

    def __init__(self, inchannels=512, num_anchors=3):
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(
            inchannels, self.num_anchors * 2, kernel_size=(1, 1), stride=1, padding=0, has_bias=True)

    def construct(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1)

        return out.view(out.shape[0], -1, 2)


class BboxHead(nn.Cell):

    def __init__(self, inchannels=512, num_anchors=3):
        super(BboxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(
            inchannels, num_anchors * 4, kernel_size=(1, 1), stride=1, padding=0, has_bias=True)

    def construct(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1)

        return out.view(out.shape[0], -1, 4)


class LandmarkHead(nn.Cell):

    def __init__(self, inchannels=512, num_anchors=3):
        super(LandmarkHead, self).__init__()
        self.conv1x1 = nn.Conv2d(
            inchannels, num_anchors * 10, kernel_size=(1, 1), stride=1, padding=0, has_bias=True)

    def construct(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1)

        return out.view(out.shape[0], -1, 10)


def make_class_head(fpn_num=3, inchannels=64, anchor_num=2):
    classhead = nn.CellList()
    for i in range(fpn_num):
        classhead.append(ClassHead(inchannels, anchor_num))
    return classhead


def make_bbox_head(fpn_num=3, inchannels=64, anchor_num=2):
    bboxhead = nn.CellList()
    for i in range(fpn_num):
        bboxhead.append(BboxHead(inchannels, anchor_num))
    return bboxhead


def make_landmark_head(fpn_num=3, inchannels=64, anchor_num=2):
    landmarkhead = nn.CellList()
    for i in range(fpn_num):
        landmarkhead.append(LandmarkHead(inchannels, anchor_num))
    return landmarkhead
