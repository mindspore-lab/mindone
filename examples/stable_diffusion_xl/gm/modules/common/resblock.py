# Copyright © 2023 Huawei Technologies Co, Ltd. All Rights Reserved.
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

from gm.util.init_weights import default_init_weights

import mindspore.nn as nn


class ResidualBlockNoBN(nn.Cell):
    """Residual block without Batch Norm.
    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|
    Args:
        mid_channels [int]: Channel number of intermediate features.
            Default: 64.
        res_scale [float]: Used to scale the residual before addition.
            Default: 1.0.
        has_bias [bool]: Add bias after convolution layers.
            Default: True.
    """

    def __init__(self, mid_channels=64, res_scale=1.0, has_bias=True):
        super().__init__()

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride=1, pad_mode="pad", padding=1, has_bias=has_bias)
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, stride=1, pad_mode="pad", padding=1, has_bias=has_bias)
        self.relu = nn.ReLU()

        # if res_scale < 1.0, use the default initialization, as in EDSR.
        # if res_scale = 1.0, use scaled kaiming_init, as in MSRResNet.
        self.res_scale = res_scale
        if res_scale == 1.0:
            self.init_weights()

    def init_weights(self):
        """Initialize weights for ResidualBlockNoBN.
        Initialization methods like `kaiming_init` are for VGG-style
        modules. For modules with residual paths, using smaller std is
        better for stability and performance. We empirically use 0.1.
        See more details in "ESRGAN: Enhanced Super-Resolution Generative
        Adversarial Networks"
        """

        for net in [self.conv1, self.conv2]:
            default_init_weights(net, 0.1)

    def construct(self, x):
        """Construct function.
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
        Returns:
            Tensor: Results.
        """

        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv2(y)
        out = x + y * self.res_scale
        return out
