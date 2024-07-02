# reference to https://github.com/mlfoundations/open_clip

from collections import OrderedDict

import numpy as np
from sgm.modules.transformers import multi_head_attention_forward

import mindspore as ms
from mindspore import Parameter, Tensor, nn, ops
from mindspore.common import initializer as init


class Bottleneck(nn.Cell):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, has_bias=False, pad_mode="valid")
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, has_bias=False, pad_mode="valid")
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = nn.ReLU()

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, has_bias=False, pad_mode="valid")
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.act3 = nn.ReLU()

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.SequentialCell(
                OrderedDict(
                    [
                        ("-1", nn.AvgPool2d(stride)),
                        (
                            "0",
                            nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, has_bias=False, pad_mode="valid"),
                        ),
                        ("1", nn.BatchNorm2d(planes * self.expansion)),
                    ]
                )
            )

    def construct(self, x):
        identity = x

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act3(out)
        return out


class AttentionPool2d(nn.Cell):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = Parameter(
            Tensor(np.random.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5, ms.float32)
        )
        self.k_proj = nn.Dense(embed_dim, embed_dim)
        self.q_proj = nn.Dense(embed_dim, embed_dim)
        self.v_proj = nn.Dense(embed_dim, embed_dim)
        self.c_proj = nn.Dense(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def construct(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).transpose(2, 0, 1)  # NCHW -> (HW)NC
        x = ops.concat((x.mean(axis=0, keepdim=True), x), axis=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].astype(x.dtype)  # (HW+1)NC
        x, _ = multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_bias=ops.concat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
        )

        return x[0]


class ModifiedResNet(nn.Cell):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, image_size=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.image_size = image_size

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, has_bias=False, pad_mode="valid")
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, has_bias=False, pad_mode="valid")
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, has_bias=False, pad_mode="valid")
        self.bn3 = nn.BatchNorm2d(width)
        self.act3 = nn.ReLU()
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(image_size // 32, embed_dim, heads, output_dim)

        self.init_parameters()

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.SequentialCell(layers)

    def init_parameters(self):
        if self.attnpool is not None:
            std = self.attnpool.c_proj.in_features**-0.5
            for weight in (
                self.attnpool.q_proj.weight,
                self.attnpool.k_proj.weight,
                self.attnpool.v_proj.weight,
                self.attnpool.c_proj.weight,
            ):
                weight.set_data(init.initializer(init.Normal(sigma=std), weight.shape, weight.dtype))

        for resnet_block in (self.layer1, self.layer2, self.layer3, self.layer4):
            for name, weight in resnet_block.parameters_and_names():
                if name.endswith("bn3.gamma"):
                    weight.set_data(init.initializer(init.Zero(), weight.shape, weight.dtype))

    def stem(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        return x

    def construct(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x
