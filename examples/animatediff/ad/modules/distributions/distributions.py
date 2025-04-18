# Copyright 2023 Huawei Technologies Co., Ltd
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
from mindspore import mint, ops


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.mean, self.logvar = mint.split(parameters, parameters.shape[1] // 2, dim=1)
        self.logvar = mint.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.stdnormal = ops.StandardNormal()
        self.std = mint.exp(0.5 * self.logvar)

    def sample(self):
        x = self.mean + self.std * self.stdnormal(self.mean.shape)
        return x
