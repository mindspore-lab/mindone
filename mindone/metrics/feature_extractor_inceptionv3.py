# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import sys
from contextlib import redirect_stdout
from typing import Tuple, Union

import mindspore
import mindspore.common.initializer as init
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

from mindone.metrics.utils.download_weights import load_state_dict_from_url

# InceptionV3 weights converted from the official TensorFlow weights using utils/util_convert_inception_weights.py
#   Original weights distributed under Apache License 2.0: https://github.com/tensorflow/models/blob/master/LICENSE
URL_INCEPTION_V3_WEIGHTS = \
    "https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/weights-inception-2015-12-05-6726825d.pth"


class BasicConv2d(nn.Cell):
    """A block for conv bn and relu"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple] = 1,
            stride: int = 1,
            padding: int = 0,
            pad_mode: str = "same",
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, pad_mode=pad_mode)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.9997)
        self.relu = nn.ReLU()

    def construct(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class InceptionA(nn.Cell):
    def __init__(
            self,
            in_channels: int,
            pool_features: int,
    ) -> None:
        super().__init__()
        self.branch0 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch1 = nn.SequentialCell([
            BasicConv2d(in_channels, 48, kernel_size=1),
            BasicConv2d(48, 64, kernel_size=5)
        ])
        self.branch2 = nn.SequentialCell([
            BasicConv2d(in_channels, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3),
            BasicConv2d(96, 96, kernel_size=3)
        ])
        self.branch_pool = nn.SequentialCell([
            nn.AvgPool2d(kernel_size=3, pad_mode="same"),
            BasicConv2d(in_channels, pool_features, kernel_size=1)
        ])

    def construct(self, x: Tensor) -> Tensor:
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        branch_pool = self.branch_pool(x)
        out = ops.concat((x0, x1, x2, branch_pool), axis=1)
        return out


class InceptionB(nn.Cell):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.branch0 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2, pad_mode='valid')
        self.branch1 = nn.SequentialCell([
            BasicConv2d(in_channels, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3),
            BasicConv2d(96, 96, kernel_size=3, stride=2, pad_mode="valid")
        ])
        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def construct(self, x: Tensor) -> Tensor:
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        branch_pool = self.branch_pool(x)
        out = ops.concat((x0, x1, branch_pool), axis=1)
        return out


class InceptionC(nn.Cell):
    def __init__(
            self,
            in_channels: int,
            channels_7x7: int,
    ) -> None:
        super().__init__()
        self.branch0 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch1 = nn.SequentialCell([
            BasicConv2d(in_channels, channels_7x7, kernel_size=1),
            BasicConv2d(channels_7x7, channels_7x7, kernel_size=(1, 7)),
            BasicConv2d(channels_7x7, 192, kernel_size=(7, 1))
        ])
        self.branch2 = nn.SequentialCell([
            BasicConv2d(in_channels, channels_7x7, kernel_size=1),
            BasicConv2d(channels_7x7, channels_7x7, kernel_size=(7, 1)),
            BasicConv2d(channels_7x7, channels_7x7, kernel_size=(1, 7)),
            BasicConv2d(channels_7x7, channels_7x7, kernel_size=(7, 1)),
            BasicConv2d(channels_7x7, 192, kernel_size=(1, 7))
        ])
        self.branch_pool = nn.SequentialCell([
            nn.AvgPool2d(kernel_size=3, pad_mode="same"),
            BasicConv2d(in_channels, 192, kernel_size=1)
        ])

    def construct(self, x: Tensor) -> Tensor:
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        branch_pool = self.branch_pool(x)
        out = ops.concat((x0, x1, x2, branch_pool), axis=1)
        return out


class InceptionD(nn.Cell):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.branch0 = nn.SequentialCell([
            BasicConv2d(in_channels, 192, kernel_size=1),
            BasicConv2d(192, 320, kernel_size=3, stride=2, pad_mode="valid")
        ])
        self.branch1 = nn.SequentialCell([
            BasicConv2d(in_channels, 192, kernel_size=1),
            BasicConv2d(192, 192, kernel_size=(1, 7)),  # check
            BasicConv2d(192, 192, kernel_size=(7, 1)),
            BasicConv2d(192, 192, kernel_size=3, stride=2, pad_mode="valid")
        ])
        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def construct(self, x: Tensor) -> Tensor:
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        branch_pool = self.branch_pool(x)
        out = ops.concat((x0, x1, branch_pool), axis=1)
        return out


class InceptionE1(nn.Cell):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.branch0 = BasicConv2d(in_channels, 320, kernel_size=1)
        self.branch1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch1a = BasicConv2d(384, 384, kernel_size=(1, 3))
        self.branch1b = BasicConv2d(384, 384, kernel_size=(3, 1))
        self.branch2 = nn.SequentialCell([
            BasicConv2d(in_channels, 448, kernel_size=1),
            BasicConv2d(448, 384, kernel_size=3)
        ])
        self.branch2a = BasicConv2d(384, 384, kernel_size=(1, 3))
        self.branch2b = BasicConv2d(384, 384, kernel_size=(3, 1))
        self.branch_pool = nn.SequentialCell([
            nn.AvgPool2d(kernel_size=3, pad_mode="same"),
            BasicConv2d(in_channels, 192, kernel_size=1)
        ])

    def construct(self, x: Tensor) -> Tensor:
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x1 = ops.concat((self.branch1a(x1), self.branch1b(x1)), axis=1)
        x2 = self.branch2(x)
        x2 = ops.concat((self.branch2a(x2), self.branch2b(x2)), axis=1)
        branch_pool = self.branch_pool(x)
        out = ops.concat((x0, x1, x2, branch_pool), axis=1)
        return out


class InceptionE2(nn.Cell):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.branch0 = BasicConv2d(in_channels, 320, kernel_size=1)
        self.branch1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch1a = BasicConv2d(384, 384, kernel_size=(1, 3))
        self.branch1b = BasicConv2d(384, 384, kernel_size=(3, 1))
        self.branch2 = nn.SequentialCell([
            BasicConv2d(in_channels, 448, kernel_size=1),
            BasicConv2d(448, 384, kernel_size=3)
        ])
        self.branch2a = BasicConv2d(384, 384, kernel_size=(1, 3))
        self.branch2b = BasicConv2d(384, 384, kernel_size=(3, 1))
        self.branch_pool = nn.SequentialCell([
            # Patch: TensorFlow Inception model uses max pooling instead of average
            # pooling. This is likely an error in this specific Inception
            # implementation, as other Inception models use average pooling here
            # (which matches the description in the paper).
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, pad_mode='pad'),
            BasicConv2d(in_channels, 192, kernel_size=1)
        ])

    def construct(self, x: Tensor) -> Tensor:
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x1 = ops.concat((self.branch1a(x1), self.branch1b(x1)), axis=1)
        x2 = self.branch2(x)
        x2 = ops.concat((self.branch2a(x2), self.branch2b(x2)), axis=1)
        branch_pool = self.branch_pool(x)
        out = ops.concat((x0, x1, x2, branch_pool), axis=1)
        return out


class FeatureExtractorInceptionV3(nn.Cell):
    """
    InceptionV3 Feature Extractor Module for 2D RGB 24bit images that never leaves evaluation mode.
        Args:

        name (str): Unique name of the feature extractor, must be the same as used in
            :func:`register_feature_extractor`.

        features_list (list): A list of the requested feature names, which will be produced for each input. This
            feature extractor provides the following features:

            - '64'
            - '192'
            - '768'
            - '2048'
            - 'logits_unbiased'

        feature_extractor_weights_path (str): Path to the pretrained InceptionV3 model weights in Mindspore format.
            Refer to `util_convert_inception_weights` for making your own. Downloads from internet if `None`.

    """

    INPUT_IMAGE_SIZE = 299

    def __init__(
            self,
            name,
            request_feature,
            feature_extractor_weights_path=None,
            custom_dtype=None,
            num_classes: int = 1008,
            aux_logits: bool = False,
            in_channels: int = 3,
            drop_rate: float = 0.2
    ):
        super(FeatureExtractorInceptionV3, self).__init__()
        self.validate_name(name)
        self.name = name
        self.validate_features_list(request_feature)
        self.request_feature = request_feature
        self.custom_dtype = custom_dtype

        self.aux_logits = aux_logits
        self.conv1a = BasicConv2d(in_channels, 32, kernel_size=3, stride=2, pad_mode="valid")
        self.conv2a = BasicConv2d(32, 32, kernel_size=3, stride=1, pad_mode="valid")
        self.conv2b = BasicConv2d(32, 64, kernel_size=3, stride=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3b = BasicConv2d(64, 80, kernel_size=1)
        self.conv4a = BasicConv2d(80, 192, kernel_size=3, pad_mode="valid")
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.inception5b = InceptionA(192, pool_features=32)
        self.inception5c = InceptionA(256, pool_features=64)
        self.inception5d = InceptionA(288, pool_features=64)
        self.inception6a = InceptionB(288)
        self.inception6b = InceptionC(768, channels_7x7=128)
        self.inception6c = InceptionC(768, channels_7x7=160)
        self.inception6d = InceptionC(768, channels_7x7=160)
        self.inception6e = InceptionC(768, channels_7x7=192)
        self.inception7a = InceptionD(768)
        self.inception7b = InceptionE1(1280)
        self.inception7c = InceptionE2(2048)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=drop_rate)
        self.classifier = nn.Dense(2048, num_classes)
        self._initialize_weights()

        self.resize = ops.ResizeBilinearV2()
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Dense(2048, 1008)

        if feature_extractor_weights_path is None:
            with redirect_stdout(sys.stderr):
                param_dict = load_state_dict_from_url(URL_INCEPTION_V3_WEIGHTS)
        else:
            param_dict = mindspore.load_checkpoint(feature_extractor_weights_path)

        mindspore.load_param_into_net(self, param_dict)
        for p in self.get_parameters():
            p.requires_grad = False

    def _initialize_weights(self) -> None:
        """Initialize weights for cells."""
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(
                    init.initializer(init.XavierUniform(), cell.weight.shape, cell.weight.dtype))

    def construct(self, x: Tensor) -> Tensor:
        if x.dtype != mindspore.uint8:
            raise ValueError("Expecting image as mindspore.Tensor with dtype=mindspore.uint8")
        if self.custom_dtype is not None:
            x = x.to(self.custom_dtype)
        else:
            x = x.float()

        x = self.resize(x, (self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE))
        # N x 3 x 299 x 299

        # x = (x - 128) * torch.tensor(0.0078125, dtype=torch.float32, device=x.device)  # really happening in graph
        x = (x - 128) / 128  # but this gives bit-exact output _of this step_ too
        # N x 3 x 299 x 299

        x = self.conv1a(x)
        # N x 32 x 149 x 149
        x = self.conv2a(x)
        # N x 32 x 147 x 147
        x = self.conv2b(x)
        # N x 64 x 147 x 147
        x = self.maxpool1(x)
        # N x 64 x 73 x 73

        if '64' == self.request_feature:
            return self.adaptive_avg_pool(x).squeeze(-1).squeeze(-1)

        x = self.conv3b(x)
        # N x 80 x 73 x 73
        x = self.conv4a(x)
        # N x 192 x 71 x 71
        x = self.maxpool2(x)
        # N x 192 x 35 x 35

        if '192' == self.request_feature:
            return self.adaptive_avg_pool(x).squeeze(-1).squeeze(-1)

        x = self.inception5b(x)
        # N x 256 x 35 x 35
        x = self.inception5c(x)
        # N x 288 x 35 x 35
        x = self.inception5d(x)
        # N x 288 x 35 x 35
        x = self.inception6a(x)
        # N x 768 x 17 x 17
        x = self.inception6b(x)
        # N x 768 x 17 x 17
        x = self.inception6c(x)
        # N x 768 x 17 x 17
        x = self.inception6d(x)
        # N x 768 x 17 x 17
        x = self.inception6e(x)
        # N x 768 x 17 x 17

        if '768' == self.request_feature:
            return self.adaptive_avg_pool(x).squeeze(-1).squeeze(-1)

        x = self.inception7a(x)
        # N x 1280 x 8 x 8
        x = self.inception7b(x)
        # N x 2048 x 8 x 8
        x = self.inception7c(x)
        # N x 2048 x 8 x 8
        x = self.pool(x)
        # N x 2048 x 1 x 1

        x = self.flatten(x)
        # N x 2048

        if '2048' == self.request_feature:
            return x

        if 'logits_unbiased' == self.request_feature:
            # N x 1008 (num_classes)
            return ops.mm(x, self.classifier.weight.T)

    def get_provided_features_list(self):
        return '64', '192', '768', '2048', 'logits_unbiased', 'logits'

    def validate_name(self, name):
        if not isinstance(name, str):
            raise TypeError("FeatureExtractorBase name must be a string")

    def validate_features_list(self, request_feature):
        if request_feature not in self.get_provided_features_list():
            raise ValueError(
                f"Requested feature {request_feature} is not on the list "
                f"provided by the selected feature extractor {self.get_provided_features_list()}")

    def add_flags_recursive(self, **flags):
        """implement add_flags_recursive to make sure train mode is forbidden"""
        if 'training' in flags:
            flags['training'] = False

        return super().add_flags_recursive(**flags)
