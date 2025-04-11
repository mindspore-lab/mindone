# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from https://github.com/descriptinc/descript-audio-codec under the Apache License 2.0


from mindspore import Parameter, mint, nn
from mindspore.common.initializer import Constant, TruncatedNormal, initializer

from mindone.utils import WeightNorm


def WNConv1d(*args, **kwargs):
    return WeightNorm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return WeightNorm(nn.Conv1dTranspose(*args, **kwargs))


def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * mint.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Cell):
    def __init__(self, channels):
        super().__init__()
        self.alpha = Parameter(mint.ones((1, channels, 1)))

    def construct(self, x):
        return snake(x, self.alpha)


class ResidualUnit(nn.Cell):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.SequentialCell(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad, pad_mode="pad", has_bias=True),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1, has_bias=True),
        )

    def construct(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        m.weight.set_data(initializer(TruncatedNormal(sigma=0.02), shape=m.weight.shape, dtype=m.weight.dtype))
        m.bias.set_data(initializer(Constant(0), shape=m.bias.shape, dtype=m.bias.dtype))
