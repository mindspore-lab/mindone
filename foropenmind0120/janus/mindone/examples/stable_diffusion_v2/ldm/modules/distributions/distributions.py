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
import mindspore.ops as ops


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.mean, self.logvar = ops.Split(axis=1, output_num=2)(parameters)
        self.logvar = ops.clip_by_value(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = ops.exp(0.5 * self.logvar)
        self.stdnormal = ops.StandardNormal()

    def sample(self):
        x = self.mean + self.std * self.stdnormal(self.mean.shape)
        return x
