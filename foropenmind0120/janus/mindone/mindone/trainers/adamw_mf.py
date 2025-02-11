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
"""AdamW API"""
import numpy as np

from mindspore import Parameter, ParameterTuple, Tensor
from mindspore import _checkparam as validator
from mindspore._checkparam import GT, INC_NEITHER
from mindspore.common import dtype as mstype
from mindspore.common.initializer import initializer
from mindspore.nn.optim.optimizer import Optimizer
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P

op_mul = P.Mul()
op_pow = P.Pow()
op_sqrt = P.Sqrt()
op_maximum = P.Maximum()
addcmul = P.Addcmul()

__all__ = ["AdamW"]

_adamw_opt = C.MultitypeFuncGraph("adamw_opt")


@_adamw_opt.register(
    "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Bool"
)
def _update_run_op(beta1, beta2, eps, step, lr, weight_decay, parameters, grads, exp_avg, exp_avg_sq, optim_filter):
    """Apply AdamW optimizer to the weight parameter."""
    op_cast = P.Cast()
    if optim_filter:
        param_fp32 = op_cast(parameters, mstype.float32)
        next_param = op_mul(param_fp32, 1 - lr * weight_decay)
        gradient_fp32 = op_cast(grads, mstype.float32)

        next_param = F.depend(
            next_param,
            F.assign(
                exp_avg,
                op_mul(exp_avg, beta1)
                + op_mul(gradient_fp32, op_cast(F.tuple_to_array((1.0,)), mstype.float32) - beta1),
            ),
        )
        next_param = F.depend(
            next_param,
            F.assign(
                exp_avg_sq,
                addcmul(
                    op_mul(exp_avg_sq, beta2),
                    gradient_fp32,
                    gradient_fp32,
                    op_cast(F.tuple_to_array((1.0,)), mstype.float32) - beta2,
                ),
            ),
        )

        bias_correction1 = 1 - op_pow(op_cast(beta1, mstype.float32), step)
        bias_correction2 = 1 - op_pow(op_cast(beta2, mstype.float32), step)
        step_size = lr / bias_correction1

        denom = op_sqrt(exp_avg_sq / bias_correction2) + eps

        return_param = next_param - op_mul(exp_avg / denom, step_size)
        F.assign(parameters, op_cast(return_param, F.dtype(parameters)))
        return op_cast(return_param, F.dtype(parameters))
    return op_cast(grads, F.dtype(parameters))


def _check_param_value(betas, eps, weight_decay, prim_name):
    """Check the type of inputs."""
    validator.check_value_type("betas", betas, [list, tuple], prim_name)
    validator.check("betas size", len(betas), "", [2], validator.IN, prim_name)
    validator.check_value_type("betas[0]", betas[0], [float], prim_name)
    validator.check_value_type("betas[1]", betas[1], [float], prim_name)
    validator.check_float_range(betas[0], 0.0, 1.0, INC_NEITHER, "beta1", prim_name)
    validator.check_float_range(betas[1], 0.0, 1.0, INC_NEITHER, "beta2", prim_name)
    validator.check_value_type("eps", eps, [float], prim_name)
    validator.check_float(eps, 0.0, GT, "eps", prim_name)
    validator.check_value_type("weight_decay", weight_decay, [float], prim_name)


class AdamW(Optimizer):
    """
    This is the implementation of AdamW.

    Args:
        params (Union[list[Parameter], list[dict]]): Must be list of `Parameter` or list of `dict`. When the
            `params` is a list of `dict`, the string "params", "lr", "weight_decay", and "order_params"
            are the keys can be parsed.

            - params: Required. Parameters in current group. The value must be a list of `Parameter`.

            - lr: Optional. If "lr" in the keys, the value of corresponding learning rate will be used.
              If not, the `learning_rate` in optimizer will be used. Fixed and dynamic learning rate are supported.

            - weight_decay: Optional. If "weight_decay" in the keys, the value of corresponding weight decay
              will be used. If not, the `weight_decay` in the optimizer will be used. It should be noted that weight
              decay can be a constant value or a Cell. It is a Cell only when dynamic weight decay is applied. Dynamic
              weight decay is similar to dynamic learning rate, users need to customize a weight decay schedule only
              with global step as input, and during training, the optimizer calls the instance of WeightDecaySchedule
              to get the weight decay value of current step.

            - order_params: Optional. When parameters is grouped, this usually is used to maintain the order of
              parameters that appeared in the network to improve performance. The value should be parameters whose
              order will be followed in optimizer.
              If `order_params` in the keys, other keys will be ignored and the element of 'order_params' must be in
              one group of `params`.

        learning_rate (Union[float, int, Tensor, Iterable, LearningRateSchedule]): Default: 1e-3.

            - float: The fixed learning rate value. Must be equal to or greater than 0.

            - int: The fixed learning rate value. Must be equal to or greater than 0. It will be converted to float.

            - Tensor: Its value should be a scalar or a 1-D vector. For scalar, fixed learning rate will be applied.
              For vector, learning rate is dynamic, then the i-th step will take the i-th value as the learning rate.

            - Iterable: Learning rate is dynamic. The i-th step will take the i-th value as the learning rate.

            - LearningRateSchedule: Learning rate is dynamic. During training, the optimizer calls the instance of
              LearningRateSchedule with step as the input to get the learning rate of current step.

        betas (Union[list(float), tuple(float)]): The exponential decay rate for the 1st and 2nd moment estimations.
            Default: (0.9, 0.999). Each element should be in range (0.0, 1.0).

        eps (float): Term added to the denominator to improve numerical stability. Default: 1e-6.
            Should be greater than 0.

        weight_decay (Union[float, int, Cell]): Weight decay (L2 penalty). Default: 0.0.

            - float: The fixed weight decay value. Must be equal to or greater than 0.

            - int: The fixed weight decay value. Must be equal to or greater than 0. It will be converted to float.

            - Cell: Weight decay is dynamic. During training, the optimizer calls the instance of
              the Cell with step as the input to get the weight decay value of current step.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`, the shape is the same as `params`.

    Outputs:
        tuple[bool], all elements are True.

    Raises:
        TypeError: If `learning_rate` is not one of int, float, Tensor, Iterable, LearningRateSchedule.
        TypeError: If element of `parameters` is neither Parameter nor dict.
        TypeError: If `betas[0]`, `betas[1]` or `eps` is not a float.
        TypeError: If `weight_decay` is neither float nor int.
        ValueError: If `eps` is less than or equal to 0.
        ValueError: If `betas[0]`, `betas[1]` is not in range (0.0, 1.0).
        ValueError: If `weight_decay` is less than 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> import mindspore.nn as nn
        >>> from mindformers import AutoModel
        >>> from mindformers.core.optim import AdamW
        >>>
        >>> net = AutoModel.from_pretrained("llama2_7b", num_layers=2)
        >>> #1) All parameters use the same learning rate and weight decay
        >>> optim = AdamW(params=net.trainable_params())
        >>>
        >>> #2) Use parameter groups and set different values
        >>> layernorm_params = list(filter(lambda x: 'norm' in x.name, net.trainable_params()))
        >>> no_layernorm_params = list(filter(lambda x: 'norm' not in x.name, net.trainable_params()))
        >>> group_params = [{'params': layernorm_params, 'weight_decay': 0.01},
        ...                 {'params': no_layernorm_params, 'lr': 0.01},
        ...                 {'order_params': net.trainable_params()}]
        >>> optim = AdamW(group_params, learning_rate=0.1, weight_decay=0.0)
        >>> # The layernorm_params's parameters will use default learning rate of 0.1 and weight decay of 0.01.
        >>> # The no_layernorm_params's parameters will use learning rate of 0.01 and default weight decay of 0.0.
        >>> # The final parameters order in which the optimizer will be followed is the value of 'order_params'.
        >>>
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> model = ms.Model(net, loss_fn=loss, optimizer=optim)
    """

    def __init__(self, params, learning_rate=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        _check_param_value(betas, eps, weight_decay, self.cls_name)

        super(AdamW, self).__init__(learning_rate, params, weight_decay=weight_decay)

        self.beta1 = Tensor(np.array([betas[0]]).astype(np.float32))
        self.beta2 = Tensor(np.array([betas[1]]).astype(np.float32))
        self.eps = Tensor(np.array([eps]).astype(np.float32))
        self.exp_avg = self.clone_state(prefix="adam_m", init="zeros")
        self.exp_avg_sq = self.clone_state(prefix="adam_v", init="zeros")

    def clone_state(self, prefix, init):
        r"""clone state
        Args:
            prefix (str): The prefix name of the parameters
            init (str): The initialization method
        """
        parameter_tuple = self.parameters
        new = []
        for old_param in parameter_tuple:
            new_state = Parameter(initializer(init, shape=old_param.shape, dtype=mstype.float32))
            new_state.param_info = old_param.param_info.clone()
            if hasattr(old_param.param_info, "cloned_obj"):
                old_param.param_info.cloned_obj.append(new_state)
            else:
                old_param.param_info.cloned_obj = [new_state]
            new_state.is_init = False
            new_state.set_data(initializer(init, shape=old_param.shape, dtype=mstype.float32))
            new_state.name = prefix + "." + new_state.name
            new.append(new_state)
        return ParameterTuple(new)

    # pylint: disable=W0221
    def construct(self, gradients):
        """forward process"""
        gradients = self.flatten_gradients(gradients)
        weight_decay = self.get_weight_decay()
        lr = self.get_lr()
        self.assignadd(self.global_step, self.global_step_increase_tensor)

        if self.is_group:
            if self.is_group_lr:
                optim_result = self.hyper_map(
                    F.partial(_adamw_opt, self.beta1, self.beta2, self.eps, self.global_step),
                    lr,
                    weight_decay,
                    self._parameters,
                    gradients,
                    self.exp_avg,
                    self.exp_avg_sq,
                    self.optim_filter,
                )
            else:
                optim_result = self.hyper_map(
                    F.partial(_adamw_opt, self.beta1, self.beta2, self.eps, self.global_step, lr),
                    weight_decay,
                    self._parameters,
                    gradients,
                    self.exp_avg,
                    self.exp_avg_sq,
                    self.optim_filter,
                )
        else:
            optim_result = self.hyper_map(
                F.partial(_adamw_opt, self.beta1, self.beta2, self.eps, self.global_step, lr, weight_decay),
                self._parameters,
                gradients,
                self.exp_avg,
                self.exp_avg_sq,
                self.optim_filter,
            )
        if self.use_parallel:
            self.broadcast_params(optim_result)

        return optim_result
