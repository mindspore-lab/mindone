# Copyright 2022 Huawei Technologies Co., Ltd
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
"""
 build optimizer for ms
"""
from mindspore.nn.optim.adam import Adam, AdamWeightDecay


def build_optimizer(model, opts, lr):
    """

    :param model:
    :param opts:
    :param lr:
    :return: optimizer
    """

    def decay_filter(x):
        return "layernorm" not in x.name.lower() and "bias" not in x.name.lower()

    param_optimizer = model.trainable_params()
    decay_params = list(filter(decay_filter, param_optimizer))
    other_params = list(filter(lambda x: not decay_filter(x), param_optimizer))
    group_params = []
    if len(decay_params) > 0:
        group_params.append({"params": decay_params, "weight_decay": opts.weight_decay})  # 1e-6})
    if len(other_params) > 0:
        group_params.append({"params": other_params, "weight_decay": 0.0})
    group_params.append({"order_params": param_optimizer})

    if opts.optim == "adam":
        OptimCls = Adam
    elif opts.optim == "adamw":
        OptimCls = AdamWeightDecay
    else:
        raise ValueError("invalid optimizer")

    optimizer = OptimCls(group_params, learning_rate=lr, beta1=opts.betas[0], beta2=opts.betas[1])
    return optimizer
