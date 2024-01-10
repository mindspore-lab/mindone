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

from gm.util.init_weights import make_layer

import mindspore.nn as nn

from .resblock import ResidualBlockNoBN


class ResidualBlocksWithInputConv(nn.Cell):
    def __init__(self, in_channels, out_channels=64, num_blocks=30, has_bias=True, act=None):
        super().__init__()

        main = []

        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, padding=1, pad_mode="pad", has_bias=has_bias))
        if act is None:
            main.append(nn.LeakyReLU(alpha=0.1))
        else:
            main.append(act)

        # residual blocks
        main.append(make_layer(ResidualBlockNoBN, num_blocks, mid_channels=out_channels, has_bias=has_bias))

        self.main = nn.SequentialCell(*main)

    def construct(self, x):
        return self.main(x)
