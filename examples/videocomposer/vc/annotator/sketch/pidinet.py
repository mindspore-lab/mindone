r"""Modified from ``https://github.com/zhuoinoulu/pidinet''.
    Image augmentation: T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]).
"""
import math
from functools import partial

import numpy as np

from mindspore import Parameter, nn, ops
from mindspore.common.initializer import Constant, HeUniform, Uniform, initializer

from ...utils.pt2ms import load_pt_weights_in_model

__all__ = [
    "PiDiNet",
    "pidinet_bsd",
]


CONFIGS = {
    "carv4": {
        "layer0": "cd",
        "layer1": "ad",
        "layer2": "rd",
        "layer3": "cv",
        "layer4": "cd",
        "layer5": "ad",
        "layer6": "rd",
        "layer7": "cv",
        "layer8": "cd",
        "layer9": "ad",
        "layer10": "rd",
        "layer11": "cv",
        "layer12": "cd",
        "layer13": "ad",
        "layer14": "rd",
        "layer15": "cv",
    }
}


def config_model_converted(model):
    model_options = list(CONFIGS.keys())
    assert model in model_options, "unrecognized model, please choose from %s" % str(model_options)

    pdcs = []
    for i in range(16):
        layer_name = "layer%d" % i
        op = CONFIGS[model][layer_name]
        pdcs.append(op)
    return pdcs


def convert_pdc(op, weight: np.ndarray):
    if op == "cv":
        return weight
    elif op == "cd":
        shape = weight.shape
        weight_c = weight.sum(axis=(2, 3))
        weight = weight.reshape((shape[0], shape[1], -1))
        weight[:, :, 4] = weight[:, :, 4] - weight_c
        weight = weight.reshape(shape)
        return weight
    elif op == "ad":
        shape = weight.shape
        weight = weight.reshape((shape[0], shape[1], -1))
        weight_conv = (weight - weight[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]).reshape(shape)
        return weight_conv
    elif op == "rd":
        shape = weight.shape
        buffer = np.zeros((shape[0], shape[1], 5 * 5))
        weight = weight.reshape((shape[0], shape[1], -1))
        buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = weight[:, :, 1:]
        buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -weight[:, :, 1:]
        buffer = buffer.reshape((shape[0], shape[1], 5, 5))
        return buffer
    raise ValueError("wrong op {}".format(str(op)))


def convert_pidinet(state_dict, config):
    pdcs = config_model_converted(config)
    new_dict = {}
    for pname, p in state_dict.items():
        if "init_block.weight" in pname:
            new_dict[pname] = convert_pdc(pdcs[0], p)
        elif "block1_1.conv1.weight" in pname:
            new_dict[pname] = convert_pdc(pdcs[1], p)
        elif "block1_2.conv1.weight" in pname:
            new_dict[pname] = convert_pdc(pdcs[2], p)
        elif "block1_3.conv1.weight" in pname:
            new_dict[pname] = convert_pdc(pdcs[3], p)
        elif "block2_1.conv1.weight" in pname:
            new_dict[pname] = convert_pdc(pdcs[4], p)
        elif "block2_2.conv1.weight" in pname:
            new_dict[pname] = convert_pdc(pdcs[5], p)
        elif "block2_3.conv1.weight" in pname:
            new_dict[pname] = convert_pdc(pdcs[6], p)
        elif "block2_4.conv1.weight" in pname:
            new_dict[pname] = convert_pdc(pdcs[7], p)
        elif "block3_1.conv1.weight" in pname:
            new_dict[pname] = convert_pdc(pdcs[8], p)
        elif "block3_2.conv1.weight" in pname:
            new_dict[pname] = convert_pdc(pdcs[9], p)
        elif "block3_3.conv1.weight" in pname:
            new_dict[pname] = convert_pdc(pdcs[10], p)
        elif "block3_4.conv1.weight" in pname:
            new_dict[pname] = convert_pdc(pdcs[11], p)
        elif "block4_1.conv1.weight" in pname:
            new_dict[pname] = convert_pdc(pdcs[12], p)
        elif "block4_2.conv1.weight" in pname:
            new_dict[pname] = convert_pdc(pdcs[13], p)
        elif "block4_3.conv1.weight" in pname:
            new_dict[pname] = convert_pdc(pdcs[14], p)
        elif "block4_4.conv1.weight" in pname:
            new_dict[pname] = convert_pdc(pdcs[15], p)
        else:
            new_dict[pname] = p
    return new_dict


class Conv2d(nn.Cell):
    def __init__(
        self, pdc, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False
    ):
        super(Conv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = Parameter(initializer("normal", (out_channels, in_channels // groups, kernel_size, kernel_size)))
        if bias:
            self.bias = Parameter(initializer("zeros", (out_channels,)))
        else:
            self.bias = None
        self.reset_parameters()
        self.pdc = pdc

    def reset_parameters(self):
        self.weight.set_data(initializer(HeUniform(math.sqrt(5)), self.weight.shape, self.weight.dtype))
        if self.bias is not None:
            fan_in = self.in_channels // self.groups * self.kernel_size * self.kernel_size
            bound = 1 / math.sqrt(fan_in)
            self.bias.set_data(initializer(Uniform(scale=bound), self.bias.shape, self.bias.dtype))

    def construct(self, input):
        return self.pdc(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class CSAM(nn.Cell):
    r"""
    Compact Spatial Attention Module
    """

    def __init__(self, channels):
        super(CSAM, self).__init__()

        mid_channels = 4
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(channels, mid_channels, kernel_size=1, padding=0, has_bias=True, pad_mode="pad")
        self.conv2 = nn.Conv2d(mid_channels, 1, kernel_size=3, padding=1, has_bias=False, pad_mode="pad")
        self.sigmoid = nn.Sigmoid()
        self.conv1.bias.set_data(initializer(Constant(0), self.conv1.bias.shape, self.conv1.bias.dtype))

    def construct(self, x):
        y = self.relu1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sigmoid(y)

        return x * y


class CDCM(nn.Cell):
    r"""
    Compact Dilation Convolution based Module
    """

    def __init__(self, in_channels, out_channels):
        super(CDCM, self).__init__()

        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, has_bias=True, pad_mode="pad")
        self.conv2_1 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, dilation=5, padding=5, has_bias=False, pad_mode="pad"
        )
        self.conv2_2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, dilation=7, padding=7, has_bias=False, pad_mode="pad"
        )
        self.conv2_3 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, dilation=9, padding=9, has_bias=False, pad_mode="pad"
        )
        self.conv2_4 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, dilation=11, padding=11, has_bias=False, pad_mode="pad"
        )
        self.conv1.bias.set_data(initializer(Constant(0), self.conv1.bias.shape, self.conv1.bias.dtype))

    def construct(self, x):
        x = self.relu1(x)
        x = self.conv1(x)
        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x3 = self.conv2_3(x)
        x4 = self.conv2_4(x)
        return x1 + x2 + x3 + x4


class MapReduce(nn.Cell):
    r"""
    Reduce feature maps into a single edge map
    """

    def __init__(self, channels):
        super(MapReduce, self).__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1, padding=0, has_bias=True, pad_mode="pad")
        self.conv.bias.set_data(initializer(Constant(0), self.conv.bias.shape, self.conv.bias.dtype))

    def construct(self, x):
        return self.conv(x)


class PDCBlock(nn.Cell):
    def __init__(self, pdc, inplane, ouplane, stride=1):
        super(PDCBlock, self).__init__()

        self.stride = stride
        if self.stride > 1:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.shortcut = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0, has_bias=True, pad_mode="pad")
        self.conv1 = Conv2d(pdc, inplane, inplane, kernel_size=3, padding=1, groups=inplane, bias=False)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0, has_bias=False, pad_mode="pad")

    def construct(self, x):
        if self.stride > 1:
            x = self.pool(x)
        y = self.conv1(x)
        y = self.relu2(y)
        y = self.conv2(y)
        if self.stride > 1:
            x = self.shortcut(x)
        y = y + x
        return y


class PDCBlock_converted(nn.Cell):
    r"""
    CPDC, APDC can be converted to vanilla 3x3 convolution
    RPDC can be converted to vanilla 5x5 convolution
    """

    def __init__(self, pdc, inplane, ouplane, stride=1):
        super(PDCBlock_converted, self).__init__()
        self.stride = stride

        if self.stride > 1:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.shortcut = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0, has_bias=True, pad_mode="pad")
        if pdc == "rd":
            self.conv1 = nn.Conv2d(
                inplane, inplane, kernel_size=5, padding=2, group=inplane, has_bias=False, pad_mode="pad"
            )
        else:
            self.conv1 = nn.Conv2d(
                inplane, inplane, kernel_size=3, padding=1, group=inplane, has_bias=False, pad_mode="pad"
            )
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0, has_bias=False, pad_mode="pad")

    def construct(self, x):
        if self.stride > 1:
            x = self.pool(x)
        y = self.conv1(x)
        y = self.relu2(y)
        y = self.conv2(y)
        if self.stride > 1:
            x = self.shortcut(x)
        y = y + x
        return y


class PiDiNet(nn.Cell):
    def __init__(self, inplane, pdcs, dil=None, sa=False, convert=False):
        super(PiDiNet, self).__init__()
        self.sa = sa
        if dil is not None:
            assert isinstance(dil, int), "dil should be an int"
        self.dil = dil

        self.fuseplanes = []

        self.inplane = inplane
        if convert:
            if pdcs[0] == "rd":
                init_kernel_size = 5
                init_padding = 2
            else:
                init_kernel_size = 3
                init_padding = 1
            self.init_block = nn.Conv2d(
                3, self.inplane, kernel_size=init_kernel_size, padding=init_padding, has_bias=False, pad_mode="pad"
            )
            block_class = PDCBlock_converted
        else:
            self.init_block = Conv2d(pdcs[0], 3, self.inplane, kernel_size=3, padding=1)
            block_class = PDCBlock

        self.block1_1 = block_class(pdcs[1], self.inplane, self.inplane)
        self.block1_2 = block_class(pdcs[2], self.inplane, self.inplane)
        self.block1_3 = block_class(pdcs[3], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane)  # C

        inplane = self.inplane
        self.inplane = self.inplane * 2
        self.block2_1 = block_class(pdcs[4], inplane, self.inplane, stride=2)
        self.block2_2 = block_class(pdcs[5], self.inplane, self.inplane)
        self.block2_3 = block_class(pdcs[6], self.inplane, self.inplane)
        self.block2_4 = block_class(pdcs[7], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane)  # 2C

        inplane = self.inplane
        self.inplane = self.inplane * 2
        self.block3_1 = block_class(pdcs[8], inplane, self.inplane, stride=2)
        self.block3_2 = block_class(pdcs[9], self.inplane, self.inplane)
        self.block3_3 = block_class(pdcs[10], self.inplane, self.inplane)
        self.block3_4 = block_class(pdcs[11], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane)  # 4C

        self.block4_1 = block_class(pdcs[12], self.inplane, self.inplane, stride=2)
        self.block4_2 = block_class(pdcs[13], self.inplane, self.inplane)
        self.block4_3 = block_class(pdcs[14], self.inplane, self.inplane)
        self.block4_4 = block_class(pdcs[15], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane)  # 4C

        if self.sa and self.dil is not None:
            self.conv_reduces = nn.CellList([MapReduce(self.dil) for _ in range(4)])
            self.attentions = nn.CellList([CSAM(self.dil) for _ in range(4)])
            self.dilations = nn.CellList([CDCM(self.fuseplanes[i], self.dil) for i in range(4)])
        elif self.sa:
            self.conv_reduces = nn.CellList([MapReduce(self.fuseplanes[i]) for i in range(4)])
            self.attentions = nn.CellList([CSAM(self.fuseplanes[i]) for i in range(4)])
        elif self.dil is not None:
            self.conv_reduces = nn.CellList([MapReduce(self.dil) for _ in range(4)])
            self.dilations = nn.CellList([CDCM(self.fuseplanes[i], self.dil) for i in range(4)])
        else:
            self.conv_reduces = nn.CellList([MapReduce(self.fuseplanes[i]) for i in range(4)])

        self.classifier = nn.Conv2d(4, 1, kernel_size=1, has_bias=True, pad_mode="pad")
        self.classifier.weight.set_data(
            initializer(Constant(0.25), self.classifier.weight.shape, self.classifier.weight.dtype)
        )
        self.classifier.bias.set_data(
            initializer(Constant(0.0), self.classifier.bias.shape, self.classifier.bias.dtype)
        )

    def construct(self, x):
        H, W = x.shape[2:]

        x = self.init_block(x)

        x1 = self.block1_1(x)
        x1 = self.block1_2(x1)
        x1 = self.block1_3(x1)

        x2 = self.block2_1(x1)
        x2 = self.block2_2(x2)
        x2 = self.block2_3(x2)
        x2 = self.block2_4(x2)

        x3 = self.block3_1(x2)
        x3 = self.block3_2(x3)
        x3 = self.block3_3(x3)
        x3 = self.block3_4(x3)

        x4 = self.block4_1(x3)
        x4 = self.block4_2(x4)
        x4 = self.block4_3(x4)
        x4 = self.block4_4(x4)

        x_fuses = []
        if self.sa and self.dil is not None:
            for i, xi in enumerate([x1, x2, x3, x4]):
                x_fuses.append(self.attentions[i](self.dilations[i](xi)))
        elif self.sa:
            for i, xi in enumerate([x1, x2, x3, x4]):
                x_fuses.append(self.attentions[i](xi))
        elif self.dil is not None:
            for i, xi in enumerate([x1, x2, x3, x4]):
                x_fuses.append(self.dilations[i](xi))
        else:
            x_fuses = [x1, x2, x3, x4]

        e1 = self.conv_reduces[0](x_fuses[0])
        e1 = ops.interpolate(e1, (H, W), mode="bilinear", align_corners=False)

        e2 = self.conv_reduces[1](x_fuses[1])
        e2 = ops.interpolate(e2, (H, W), mode="bilinear", align_corners=False)

        e3 = self.conv_reduces[2](x_fuses[2])
        e3 = ops.interpolate(e3, (H, W), mode="bilinear", align_corners=False)

        e4 = self.conv_reduces[3](x_fuses[3])
        e4 = ops.interpolate(e4, (H, W), mode="bilinear", align_corners=False)

        outputs = [e1, e2, e3, e4]
        output = self.classifier(ops.concat(outputs, axis=1))

        outputs.append(output)
        outputs = [ops.sigmoid(r) for r in outputs]
        return outputs[-1]


def pidinet_bsd(pretrained=False, vanilla_cnn=True, ckpt_path=None):
    assert vanilla_cnn is True, "vanilla_cnn only support True for now!"
    pdcs = config_model_converted("carv4")
    model = PiDiNet(60, pdcs, dil=24, sa=True, convert=vanilla_cnn)

    def remove_prefix(sd):
        prefix = "module."
        return {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in sd.items()}

    if pretrained:
        if vanilla_cnn:
            load_pt_weights_in_model(model, ckpt_path, (partial(convert_pidinet, config="carv4"), remove_prefix))
        else:
            load_pt_weights_in_model(model, ckpt_path, (remove_prefix,))
    return model


if __name__ == "__main__":
    model = pidinet_bsd(pretrained=False)
