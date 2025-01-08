import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np

from .trident_conv import MultiScaleTridentConv

class ResidualBlock(nn.Cell):
    def __init__(
        self,
        in_planes,
        planes,
        norm_layer=nn.InstanceNorm2d,
        stride=1,
        dilation=1,
    ):

        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_planes,
            out_channels=planes,
            kernel_size=3,
            stride=stride,
            pad_mode='pad',
            padding=dilation,
            dilation=dilation,
            has_bias=False
        )
        self.conv2 = nn.Conv2d(
            in_channels=planes,
            out_channels=planes,
            kernel_size=3,
            stride=1,
            pad_mode='pad',
            padding=dilation,
            dilation=dilation,
            has_bias=False
        )
        self.relu = nn.ReLU()

        self.norm1 = norm_layer(planes, affine = False)
        self.norm2 = norm_layer(planes, affine = False)
        if not stride == 1 or in_planes != planes:
            self.norm3 = norm_layer(planes, affine = False)

        if stride == 1 and in_planes == planes:
            self.downsample = None
        else:
            self.downsample = nn.SequentialCell(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, has_bias=True), self.norm3)

    def construct(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class CNNEncoder(nn.Cell):
    def __init__(
        self,
        output_dim=128,
        norm_layer=nn.InstanceNorm2d,
        num_output_scales=1,
        **kwargs,
    ):
        super(CNNEncoder, self).__init__()
        self.num_branch = num_output_scales

        feature_dims = [64, 96, 128]

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=feature_dims[0],
            kernel_size=7,
            stride=2,
            pad_mode='pad',
            padding=3,
            has_bias=False
        )  # 1/2
        self.norm1 = norm_layer(feature_dims[0], affine = False)
        self.relu1 = nn.ReLU()

        self.in_planes = feature_dims[0]
        self.layer1 = self._make_layer(feature_dims[0], stride=1, norm_layer=norm_layer)  # 1/2
        self.layer2 = self._make_layer(feature_dims[1], stride=2, norm_layer=norm_layer)  # 1/4

        # highest resolution 1/4 or 1/8
        stride = 2 if num_output_scales == 1 else 1
        self.layer3 = self._make_layer(
            feature_dims[2],
            stride=stride,
            norm_layer=norm_layer,
        )  # 1/4 or 1/8

        self.conv2 = nn.Conv2d(
            in_channels=feature_dims[2],
            out_channels=output_dim,
            kernel_size=1,
            stride=1,
            pad_mode='valid',
            padding=0,
            has_bias=True
        )

        if self.num_branch > 1:
            if self.num_branch == 4:
                strides = (1, 2, 4, 8)
            elif self.num_branch == 3:
                strides = (1, 2, 4)
            elif self.num_branch == 2:
                strides = (1, 2)
            else:
                raise ValueError

            self.trident_conv = MultiScaleTridentConv(
                output_dim,
                output_dim,
                kernel_size=3,
                strides=strides,
                paddings=1,
                num_branch=self.num_branch,
            )

        self._initialize_weights()

    def _initialize_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                # HeUniform initialization by default
                pass
            elif isinstance(cell, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if hasattr(cell, 'gamma') and cell.gamma is not None:
                    cell.gamma.set_data(ms.Tensor(np.ones(cell.gamma.shape), ms.float32))
                if hasattr(cell, 'beta') and cell.beta is not None:
                    cell.beta.set_data(ms.Tensor(np.zeros(cell.beta.shape), ms.float32))

    def _make_layer(self, dim, stride=1, dilation=1, norm_layer=nn.InstanceNorm2d):
        layer1 = ResidualBlock(self.in_planes, dim, norm_layer=norm_layer, stride=stride, dilation=dilation)
        layer2 = ResidualBlock(dim, dim, norm_layer=norm_layer, stride=1, dilation=dilation)

        layers = [layer1, layer2]

        self.in_planes = dim
        return nn.SequentialCell(layers)

    def construct(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)  # 1/2
        x = self.layer2(x)  # 1/4
        x = self.layer3(x)  # 1/8 or 1/4

        x = self.conv2(x)

        if self.num_branch > 1:
            out = self.trident_conv([x] * self.num_branch)  # high to low res
        else:
            out = [x]

        return out
