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

"""spectral conv"""

import numpy as np
from mindspore import Parameter, nn, ops
from mindspore.common.initializer import Normal, initializer

from utils.common import is_ascend


class SpectralNormAscendOpt(nn.Cell):
    """
    Compute the spectral norm of a weights matrix,
    given a pair of so-approximated singular vectors.
    """

    def construct(self, w, u, v):
        """
        Computes spectral norm as u.T * w * v.
        """
        return ops.MatMul(True, False)(u, ops.MatMul(False, False)(w, v))

    def bprop(self, w, u, v, out, dout):
        """
        Computes gradient as u * v.T.
        """
        res = dout * u * v.reshape(1, -1)
        return res, None, None


class SpectralNorm(nn.Cell):
    """
    Compute the spectral norm of a weights matrix,
    given a pair of so-approximated singular vectors.
    """

    def construct(self, w, u, v):
        """
        Computes spectral norm as u.T * w * v.
        """
        return ops.MatMul(True, False)(u, ops.MatMul(False, False)(w, v))


class Conv2dNormalized(nn.Cell):
    """Conv2d layer with spectral normalization"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        has_bias=False,
        padding=0,
        pad_mode="same",
        dilation=1,
    ):
        super().__init__()
        self.conv2d = ops.Conv2D(
            out_channel=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            pad=padding,
            pad_mode=pad_mode,
            dilation=dilation,
        )
        self.bias_add = ops.BiasAdd(data_format="NCHW")
        self.has_bias = has_bias

        if self.has_bias:
            self.bias = Parameter(initializer(
                "zeros", (out_channels,)), name="bias")

        self.weight_orig = Parameter(
            initializer(
                Normal(sigma=0.02),
                (out_channels, in_channels, kernel_size, kernel_size)
            ),
            name="weight_orig"
        )

        self.weight_u = Parameter(
            self.initialize_param(
                out_channels, 1
            ),
            requires_grad=False,
            name="weight_u"
        )

        self.weight_v = Parameter(
            self.initialize_param(
                in_channels * kernel_size * kernel_size, 1
            ),
            requires_grad=False,
            name="weight_v"
        )

        if is_ascend():
            self.spectral_norm = SpectralNormAscendOpt()
        else:
            self.spectral_norm = SpectralNorm()

    @staticmethod
    def initialize_param(*param_shape):
        """initialize params"""
        param = np.random.randn(*param_shape).astype("float32")
        return param / np.linalg.norm(param)

    def normalize_weights(self, weight_orig, u, v):
        """Weights normalization"""
        eps = 1e-12
        size = weight_orig.shape
        weight_mat = weight_orig.ravel().view(size[0], -1)

        if self.training:
            v = ops.matmul(weight_mat.T, u)
            v_norm = nn.Norm()(v).clip(eps, None)
            v = v / v_norm

            u = ops.matmul(weight_mat, v)
            u_norm = nn.Norm()(u).clip(eps, None)
            u = u / u_norm

            u = ops.depend(u, ops.assign(self.weight_u, u))
            v = ops.depend(v, ops.assign(self.weight_v, v))

        u = ops.stop_gradient(u)
        v = ops.stop_gradient(v)

        weight_norm = self.spectral_norm(weight_mat, u, v)
        weight_sn = weight_mat / weight_norm.clip(eps, None)
        weight_sn = weight_sn.view(*size)

        return weight_sn

    def construct(self, x):
        """Feed forward"""
        weight = self.normalize_weights(
            self.weight_orig, self.weight_u, self.weight_v)
        output = self.conv2d(x, weight)
        if self.has_bias:
            output = self.bias_add(output, self.bias)
        return output
