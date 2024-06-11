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
"AdamWeightDecay Optimizer"
import numpy as np

import mindspore as ms
from mindspore.common import dtype as mstype
from mindspore import ParameterTuple, Tensor, ops
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.nn.optim.optimizer import Optimizer
from mindspore.common.initializer import initializer
from mindspore.communication.management import GlobalComm, get_group_size, get_rank
from mindspore.communication._comm_helper import _get_group_ranks
import mindspore._checkparam as validator

__all__ = ["AdamWeightDecayZeRO2"]

_adamw_opt = ops.MultitypeFuncGraph("adamw_opt")
_split_params = ops.MultitypeFuncGraph("split_params")
_update_params = ops.MultitypeFuncGraph("update_params")


@_adamw_opt.register("Function", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor",
                     "Tensor", "Tensor", "Tensor", "Tensor", "Bool", "Bool")
def _update_by_opt(opt, beta1, beta2, eps, lr, weight_decay, param, m, v, gradient, decay_flag, cpu_offload):
    """
    Apply AdamWeigthDecay operator to update parameters.
    """
    if decay_flag:
        output, _, _ = opt(param, m, v, lr, beta1, beta2, eps, weight_decay, gradient)
    else:
        output, _, _ = opt(param, m, v, lr, beta1, beta2, eps, 0.0, gradient)
    if cpu_offload:
        return param
    return output


@_split_params.register("Number", "Number", "Tensor", "Bool")
def _split_params_to_fp32(shard_id, shard_size, param, need_split):
    """
    Split parameters.
    """
    split = P.Split(0, shard_size)
    cast = P.Cast()
    if need_split:
        splited_param = split(param)[shard_id]
    else:
        splited_param = param
    if splited_param.dtype != mstype.float32:
        splited_param = cast(splited_param, mstype.float32)
    return splited_param


@_update_params.register("Tensor", "Tensor", "Function")
def _update_params_opt_parallel(param, update, all_gather):
    """
    Allgather updated parameters and load.
    """
    cast = P.Cast()
    if all_gather:
        update = all_gather(update)
    if update.dtype != param.dtype:
        update = cast(update, param.dtype)
    param.assign_value(update)


def _check_param_value(beta1, beta2, eps, use_parallel, opt_parallel_group, cpu_offload, prim_name):
    """Check the type of inputs."""
    validator.check_value_type("beta1", beta1, [float], prim_name)
    validator.check_value_type("beta2", beta2, [float], prim_name)
    validator.check_value_type("eps", eps, [float], prim_name)
    validator.check_value_type("use_parallel", use_parallel, [bool], prim_name)
    validator.check_value_type("opt_parallel_group", opt_parallel_group, [str, type(None)], prim_name)
    validator.check_value_type("cpu_offload", cpu_offload, [bool], prim_name)
    validator.check_float_range(beta1, 0.0, 1.0, validator.INC_NEITHER, "beta1", prim_name)
    validator.check_float_range(beta2, 0.0, 1.0, validator.INC_NEITHER, "beta2", prim_name)
    validator.check_positive_float(eps, "eps", prim_name)


class AdamWeightDecayZeRO2(Optimizer):
    """
    This class is an implementation of AdamWeightDecay optimizer, which support ZeRO2 optimizer parallel.

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

        beta1 (float): The exponential decay rate for the 1st moment estimations. Default: 0.9.
                Should be in range (0.0, 1.0).
        beta2 (float): The exponential decay rate for the 2nd moment estimations. Default: 0.999.
                Should be in range (0.0, 1.0).
        eps (float): Term added to the denominator to improve numerical stability. Default: 1e-6.
                Should be greater than 0.

        weight_decay (Union[float, int, Cell]): Weight decay (L2 penalty). Default: 0.0.

            - float: The fixed weight decay value. Must be equal to or greater than 0.

            - int: The fixed weight decay value. Must be equal to or greater than 0. It will be converted to float.

            - Cell: Weight decay is dynamic. During training, the optimizer calls the instance of
              the Cell with step as the input to get the weight decay value of current step.

        use_parallel (bool): Enable optimizer parallel. Default: False.

        shard_size (int): Number of shards when using optimizer parallel, which should less or equal to dimension of
            data parallel. When set to -1, shard_size equal to data parallel dimension. Default: -1.

        comm_group (str): Name of communication group used by optimizer parallel. Default: None.

        cpu_offload (bool): The process of optimizer will be offload to host. The gradients, parameters and optimizer
            status will be offload to host. Default: Flase.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`, the shape is the same as `params`.

    Outputs:
        tuple[bool], all elements are True.

    Examples:
        >>> import mindspore as ms
        >>> from mindspore.nn.wrap.cell_wrapper import WithLossCell
        >>> from mindformers import AdamWeightDecayZeRO2
        >>> loss = SoftmaxCrossEntropyWithLogits()
        >>> opt = AdamWeightDecayZeRO2(params=net.get_parameters, use_parallel=True, comm_group=comm_group)
    """
    _support_parallel_optimizer = True

    def __init__(self, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-6, weight_decay=0.0,
                 use_parallel=False, opt_parallel_group=None, cpu_offload=False):
        super(AdamWeightDecayZeRO2, self).__init__(learning_rate, params, weight_decay)
        _check_param_value(beta1, beta2, eps, use_parallel, opt_parallel_group, cpu_offload, self.cls_name)
        self._is_stand_alone_mode = (ms.get_auto_parallel_context("parallel_mode") == ms.ParallelMode.STAND_ALONE)
        self.use_parallel = use_parallel
        if opt_parallel_group:
            self.opt_parallel_group = opt_parallel_group
        elif self.use_parallel:
            self.opt_parallel_group = GlobalComm.WORLD_COMM_GROUP
        self.cpu_offload = cpu_offload

        # init communication group info
        self._init_optimizer_shard_info()

        self._parameter_splited = [False] * len(self._parameters)
        self.all_gather_ops = self._init_all_gather_ops(self._parameters)
        if self.use_parallel or not self._is_stand_alone_mode:
            self._regist_hook_for_params()

        self.beta1 = Tensor(np.array([beta1]).astype(np.float32))
        self.beta2 = Tensor(np.array([beta2]).astype(np.float32))
        self.eps = Tensor(np.array([eps]).astype(np.float32))

        self.moments1 = self._init_momentum(self._parameters, prefix="adam_m", init="zeros")
        self.moments2 = self._init_momentum(self._parameters, prefix="adam_v", init="zeros")

        self._opt_params_need_offload = {"beta1": self.beta1, "beta2": self.beta2, "eps": self.eps,
                                         "moments1": self.moments1, "moments2": self.moments2}

        self.opt = P.AdamWeightDecay()
        if self.cpu_offload:
            self.opt.set_device("CPU")

    def _init_optimizer_shard_info(self):
        """Init optimizer parallel information."""
        # pylint: disable=W0212
        if not self.use_parallel:
            self.shard_id = 0
            self.shard_size = 1
        else:
            self.shard_size = get_group_size(self.opt_parallel_group)
            group_list = _get_group_ranks(self.opt_parallel_group)
            group_list.sort()
            self.rank_id = get_rank()
            self.shard_id = group_list.index(self.rank_id)

    def _init_all_gather_ops(self, params):
        """Init allgather operations for each parameter."""
        op_list = []
        for i, param in enumerate(params):
            if self.use_parallel and param.shape[0] % self.shard_size == 0:
                op_list.append(P.AllGather(self.opt_parallel_group))
                self._parameter_splited[i] = True
            else:
                op_list.append(None)
        return tuple(op_list)

    def _regist_hook_for_params(self):
        """Register hook for model parameters for optimizer parallel."""
        def reduce_scatter_hook(grad):
            allreduce = P.AllReduce()
            split = P.Split(0, self.shard_size)
            return split(allreduce(grad))[self.shard_id]
        def reduce_hook(grad):
            allreduce = P.AllReduce()
            return allreduce(grad)
        for i, param in enumerate(self._parameters):
            if self._parameter_splited[i]:
                param.register_hook(reduce_scatter_hook)
            else:
                param.register_hook(reduce_hook)

    def _init_momentum(self, params, prefix, init="zeros"):
        """Init momentum or variance for adamw optimizer."""
        moments_list = []
        for i, param in enumerate(params):
            param_shape = param.shape
            if self._parameter_splited[i]:
                param_shape = list(param_shape)
                param_shape[0] = param_shape[0] // self.shard_size
                param_shape = tuple(param_shape)
            moment = ms.Parameter(initializer(init, shape=param_shape, dtype=mstype.float32),
                                  name=prefix+"."+param.name)
            moments_list.append(moment)

        return ParameterTuple(moments_list)

    def _offload_optimizer_params(self):
        """Offload optimizer parameters to host."""
        for _, value in self._opt_params_need_offload.items():
            # pylint: disable=W0212
            if isinstance(value, ParameterTuple):
                for param in value:
                    param._offload()
            else:
                value._offload()

    def construct(self, grads):
        """construct method"""
        params = self.hyper_map(F.partial(_split_params, self.shard_id, self.shard_size),
                                self._parameters, self._parameter_splited)
        grads = self.flatten_gradients(grads)
        weight_decay = self.get_weight_decay()
        lr = self.get_lr()

        # pylint: disable=W0212
        if self.cpu_offload:
            self._offload_optimizer_params()
            for grad in grads:
                grad._offload()
            for param in params:
                param._offload()

        self.assignadd(self.global_step, self.global_step_increase_tensor)

        if self.is_group:
            if self.is_group_lr:
                optim_result = self.hyper_map(
                    F.partial(_adamw_opt, self.opt, self.beta1, self.beta2, self.eps),
                    lr,
                    weight_decay,
                    params,
                    self.moments1,
                    self.moments2,
                    grads,
                    self.decay_flags,
                    (self.cpu_offload,)*self.param_length
                )
            else:
                optim_result = self.hyper_map(
                    F.partial(_adamw_opt, self.opt, self.beta1, self.beta2, self.eps, lr),
                    weight_decay,
                    params,
                    self.moments1,
                    self.moments2,
                    grads,
                    self.decay_flags,
                    (self.cpu_offload,)*self.param_length
                )
        else:
            optim_result = self.hyper_map(
                F.partial(_adamw_opt, self.opt, self.beta1, self.beta2, self.eps, lr, weight_decay),
                params,
                self.moments1,
                self.moments2,
                grads,
                self.decay_flags,
                (self.cpu_offload,)*self.param_length
            )

        self.hyper_map(_update_params, self._parameters, optim_result, self.all_gather_ops)
