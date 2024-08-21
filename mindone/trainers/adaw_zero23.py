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
"AdamWeightDecay ZeRO Optimizer"
import numpy as np
import mindspore as ms
from mindspore import ParameterTuple, Tensor, ops, nn, _no_grad
import mindspore._checkparam as validator
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.nn.optim.optimizer import Optimizer
from mindspore.common import dtype as mstype
from mindspore.common.initializer import initializer

from .create_comm import get_dp_rank, \
    get_dp_world_size, get_dp_group


__all__ = ["AdamW"]

_adamw_opt = ops.MultitypeFuncGraph("adamw_opt")
_split_params = ops.MultitypeFuncGraph("split_params")
_update_params = ops.MultitypeFuncGraph("update_params")


@_adamw_opt.register("Function", "Function", "Function", "Function", "Function", "Tensor", "Tensor",
                     "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor")
def _update_by_opt(op_mul, op_pow, op_sqrt, addcmul, op_cast, beta1, beta2, eps, step, lr,
                   weight_decay, parameters, grads, exp_avg, exp_avg_sq):
    """
    Apply AdamWeigthDecay operator to update parameters.
    """
    param_fp32 = op_cast(parameters, mstype.float32)
    next_param = op_mul(param_fp32, 1 - lr * weight_decay)
    gradient_fp32 = op_cast(grads, mstype.float32)
    one = op_cast(F.tuple_to_array((1.0,)), mstype.float32)
    exp_avg_update = op_mul(exp_avg, beta1) + op_mul(gradient_fp32,
                                                     one - beta1)
    F.assign(exp_avg, exp_avg_update)
    exp_avg_sq_update = addcmul(op_mul(exp_avg_sq, beta2),
                                gradient_fp32,
                                gradient_fp32,
                                one - beta2)
    F.assign(exp_avg_sq, exp_avg_sq_update)
    bias_correction1 = 1 - op_pow(op_cast(beta1, mstype.float32), step)
    bias_correction2 = 1 - op_pow(op_cast(beta2, mstype.float32), step)
    step_size = lr / bias_correction1

    denom = op_sqrt(exp_avg_sq / bias_correction2) + eps
    return_param = next_param - op_mul(exp_avg / denom, step_size)
    return_param = op_cast(return_param, F.dtype(parameters))
    return return_param


@_split_params.register("Number", "Function", "Tensor", "Bool")
def _split_params_to_fp32(shard_id, split, param, need_split):
    """
    Split parameters.
    """
    if need_split:
        splited_param = split(param)[shard_id]
    else:
        splited_param = param
    if splited_param.dtype != mstype.float32:
        splited_param = ops.cast(splited_param, mstype.float32)
    return splited_param


@_update_params.register("Tensor", "Tensor", "Function")
def _update_params_opt_parallel(param, update, all_gather):
    """
    Allgather updated parameters and load.
    """
    if all_gather:
        update = all_gather(update)
    if update.dtype != param.dtype:
        update = ops.cast(update, param.dtype)
    param.assign_value(update)


def _inner_grad_reduce_scatter(reduce_scatter, grads):
    grads = reduce_scatter(grads)
    return grads


def _check_param_value(beta1, beta2, eps, opt_parallel_group, cpu_offload, prim_name):
    """Check the type of inputs."""
    validator.check_value_type("beta1", beta1, [float], prim_name)
    validator.check_value_type("beta2", beta2, [float], prim_name)
    validator.check_value_type("eps", eps, [float], prim_name)
    validator.check_value_type("opt_parallel_group", opt_parallel_group, [str, type(None)], prim_name)
    validator.check_value_type("cpu_offload", cpu_offload, [bool], prim_name)
    validator.check_float_range(beta1, 0.0, 1.0, validator.INC_NEITHER, "beta1", prim_name)
    validator.check_float_range(beta2, 0.0, 1.0, validator.INC_NEITHER, "beta2", prim_name)
    validator.check_positive_float(eps, "eps", prim_name)


class AdamW(Optimizer):
    """
    This class is an implementation of AdamWeightDecay optimizer, which support ZeRO1, ZeRO2 and ZeRO3
    optimizer parallel.

    Args:
        network (mindspore.nn.Cell): Training model.
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

        zero_level (str): Enable optimizer parallel. Default: z1.

            - z1: ZeRO1, Splitting optimizer states.

            - z2: ZeRO2, Splitting optimizer states and gradient.

            - z3: ZeRO3, Splitting optimizer states, gradient and model parameter.

        param_resident (bool): After the forward propagation, the parameters are resident and not split.
            Default: Flase.

        allreduce_after_grad_accumulation (bool): Use allreduce in optimizer after gradient accumulation.
            Default: Flase.

        grad_allreduce_op (str): Gradient allreduce operator. like `sum`, `mean`. Default: sum.

        opt_parallel_group (str): Name of communication group used by optimizer parallel. Default: None.

        cpu_offload (bool): The process of optimizer will be offload to host. The gradients, parameters and optimizer
            status will be offload to host. Default: Flase.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`, the shape is the same as `params`.

    Raises:
        NotImplementedError: If `grad_allreduce_op` is not mean or sum.

    Examples:
        >>> import mindspore as ms
        >>> from mindspore.communication.management import init, GlobalComm
        >>> from mindspore.nn.wrap.cell_wrapper import WithLossCell
        >>> from mindformers import AdamW
        >>> from mindformers.experimental.distri_cores.dynamic_cluster import initialize_model_parallel
        >>> loss = SoftmaxCrossEntropyWithLogits()
        >>> ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE)
        >>> ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL)
        >>> init()
        >>> initialize_model_parallel()
        >>> optimizer = AdamW(network=network, params=network.trainable_params(), learning_rate=1e-3,
                              zero_level="z3", cpu_offload=False)
    """
    _support_parallel_optimizer = True

    def __init__(self, network, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-6, weight_decay=0.0,
                 zero_level="z1", param_resident=False, allreduce_after_grad_accumulation=False,
                 grad_allreduce_op="sum", opt_parallel_group=None, cpu_offload=False, with_context_parallel=False):
        super(AdamW, self).__init__(learning_rate, params, weight_decay)
        _check_param_value(beta1, beta2, eps, opt_parallel_group, cpu_offload, self.cls_name)
        if grad_allreduce_op not in ["mean", "sum"]:
            raise NotImplementedError(f"{grad_allreduce_op} is not supported in AdamWeightDecay yet.")
        self.network = network
        self.zero_level = zero_level
        self.cpu_offload = cpu_offload
        self.param_resident = param_resident
        self.allreduce_after_grad_accumulation = allreduce_after_grad_accumulation
        self.with_context_parallel = with_context_parallel
        self.grad_allreduce_sum = grad_allreduce_op == "sum"
        self._parameter_splited = [False] * len(self._parameters)
        self._status_splited = [False] * len(self._parameters)
        self.all_gather_ops = [None] * len(self._parameters)
        # init communication group info
        self._init_optimizer_shard_info()

        self.reduce_scatter = ops.ReduceScatter(group=get_dp_group(with_context_parallel=self.with_context_parallel))
        self.allreduce = P.AllReduce(group=get_dp_group(with_context_parallel=self.with_context_parallel))
        self.allgather = P.AllGather(group=get_dp_group(with_context_parallel=self.with_context_parallel))
        self.split = P.Split(0, self.shard_size)

        if self.zero_level == "z3":
            self._regist_hook_for_cells()
        self._parameter_status_split()
        if not self.allreduce_after_grad_accumulation:
            self._regist_hook_for_params()
        self._init_all_gather_ops()
        self.all_gather_ops = tuple(self.all_gather_ops)
        self.beta1 = Tensor(np.array([beta1]).astype(np.float32))
        self.beta2 = Tensor(np.array([beta2]).astype(np.float32))
        self.eps = Tensor(np.array([eps]).astype(np.float32))

        self.moments1 = self._init_momentum(self._parameters, prefix="adam_m", init="zeros")
        self.moments2 = self._init_momentum(self._parameters, prefix="adam_v", init="zeros")

        self._opt_params_need_offload = {"beta1": self.beta1, "beta2": self.beta2, "eps": self.eps,
                                         "moments1": self.moments1, "moments2": self.moments2}
        self.op_mul = P.Mul()
        self.op_pow = P.Pow()
        self.op_sqrt = P.Sqrt()
        self.op_maximum = P.Maximum()
        self.addcmul = P.Addcmul()
        self.op_cast = P.Cast()
        if self.cpu_offload:
            self.op_mul.set_device("CPU")
            self.op_pow.set_device("CPU")
            self.op_sqrt.set_device("CPU")
            self.op_maximum.set_device("CPU")
            self.addcmul.set_device("CPU")
            self.op_cast.set_device("CPU")

    def _init_optimizer_shard_info(self):
        """Init optimizer parallel information."""
        self.shard_size = get_dp_world_size(with_context_parallel=self.with_context_parallel)
        self.shard_id = get_dp_rank(with_context_parallel=self.with_context_parallel)

    def _init_all_gather_ops(self):
        for i, param_splited in enumerate(self._status_splited):
            if param_splited:
                self.all_gather_ops[i] = self.allgather

    def _regist_hook_for_cells(self):
        """Register hook for model parameters for optimizer parallel."""

        # pylint: disable=W0622
        @_no_grad()
        def _pre_forward_cell_hook(cell, input):
            for cell_param in cell.get_parameters():
                ag_param = self.allgather(cell_param)
                cell_param.assign_value(ag_param)
            return input

        # pylint: disable=W0622, W0613
        @_no_grad()
        def _post_forward_cell_hook(cell, input, output):
            for cell_param in cell.get_parameters():
                split_param = self.split(cell_param)[self.shard_id].contiguous()
                cell_param.assign_value(split_param)
            return output

        # pylint: disable=W0622, W0613
        def _pre_backward_cell_hook(cell, input, output):
            if not hasattr(cell, "pre_back_cell"):
                @_no_grad()
                def _run_before_backward_function(sub_cell):
                    for cell_param in sub_cell.get_parameters():
                        ag_param = self.allgather(cell_param)
                        cell_param.assign_value(ag_param)

                class PreBackwardCell(nn.Cell):
                    "Insert a cell before backward propagation"

                    def __init__(self, func):
                        super().__init__()
                        self._auto_prefix = False
                        self.pre_backward_function = func
                        self.used_bprop_inputs = []

                    def construct(self, input, input_bias):
                        return ops.stop_gradient(input), ops.stop_gradient(input_bias)

                    def bprop(self, *args):
                        self.pre_backward_function(cell)
                        return args[-1]

                cell.pre_back_cell = PreBackwardCell(_run_before_backward_function)
            output, output_bias = output
            output = cell.pre_back_cell(output, output_bias)
            return output

        def _post_backward_cell_hook(cell, input):
            if not hasattr(cell, "post_back_cell"):
                @_no_grad()
                def _run_after_backward_function(sub_cell):
                    for cell_param in sub_cell.get_parameters():
                        split_param = self.split(cell_param)[self.shard_id].contiguous()
                        cell_param.assign_value(split_param)

                class PostBackwardCell(nn.Cell):
                    "Insert a cell after backward propagation"

                    def __init__(self, func):
                        super().__init__()
                        self._auto_prefix = False
                        self.post_backward_function = func
                        self.used_bprop_inputs = []

                    def construct(self, input):
                        return ops.stop_gradient(input)

                    def bprop(self, *args):
                        self.post_backward_function(cell)
                        return args[-1]

                cell.post_back_cell = PostBackwardCell(_run_after_backward_function)
            input = cell.post_back_cell(input)
            return input

        self.z3_optim_cells = []

        def recursion_cells(cell):
            sub_cells_list = cell.cells()
            for sub_cell in sub_cells_list:
                if sub_cell.__class__.__name__ in ["ColumnParallelLinear", "RowParallelLinear"] and sub_cell.use_zero3:
                    self.z3_optim_cells.append(sub_cell)
                else:
                    recursion_cells(sub_cell)

        recursion_cells(self.network)

        self.zero3_parameters = []
        for sub_cell in self.z3_optim_cells:
            sub_cell.register_forward_pre_hook(_pre_forward_cell_hook)
            if not self.param_resident:
                sub_cell.register_forward_hook(_post_forward_cell_hook)
                sub_cell.register_forward_hook(_pre_backward_cell_hook)
            sub_cell.register_forward_pre_hook(_post_backward_cell_hook)
            for sub_cell_param in sub_cell.get_parameters():
                self.zero3_parameters.append(sub_cell_param.name)

    def _parameter_status_split(self):
        """split parameters and status"""
        for i, param in enumerate(self._parameters):
            if self.zero_level == 'z3':
                if param.name in self.zero3_parameters:
                    self._parameter_splited[i] = True
                else:
                    if param.shape[0] % self.shard_size == 0:
                        self._status_splited[i] = True
            else:
                if param.shape[0] % self.shard_size == 0:
                    self._status_splited[i] = True

    def _regist_hook_for_params(self):
        """Register hook for model parameters for optimizer parallel."""

        def reduce_scatter_hook(grad):
            # allreduce = P.AllReduce(group=get_dp_group())
            # split = P.Split(0, self.shard_size)
            # res = split(allreduce(grad))[self.shard_id].contiguous()
            res = self.reduce_scatter(grad)
            if not self.grad_allreduce_sum:
                res = res / self.shard_size
            return res

        def reduce_hook(grad):
            res = self.allreduce(grad)
            if not self.grad_allreduce_sum:
                res = res / self.shard_size
            return res

        for i, param in enumerate(self._parameters):
            if (self._parameter_splited[i] or self._status_splited[i]) and self.zero_level in ["z2", "z3"]:
                param.register_hook(reduce_scatter_hook)
            else:
                param.register_hook(reduce_hook)

    def _init_momentum(self, params, prefix, init="zeros"):
        """Init momentum or variance for adamw optimizer."""
        moments_list = []
        for i, param in enumerate(params):
            param_shape = param.shape
            if self._status_splited[i]:
                param_shape = list(param_shape)
                param_shape[0] = param_shape[0] // self.shard_size
                param_shape = tuple(param_shape)
            moment = ms.Parameter(initializer(init, shape=param_shape, dtype=mstype.float32),
                                  name=prefix + "." + param.name)
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
        if self.zero_level == "z1":
            grads = self.hyper_map(F.partial(_split_params, self.shard_id, self.split),
                                   grads, self._status_splited)
        if self.allreduce_after_grad_accumulation:
            grads = self.hyper_map(F.partial(_inner_grad_reduce_scatter, self.reduce_scatter), grads)

        params = self.hyper_map(F.partial(_split_params, self.shard_id, self.split),
                                self._parameters, self._status_splited)
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
                    F.partial(_adamw_opt, self.op_mul, self.op_pow, self.op_sqrt,
                              self.addcmul, self.op_cast, self.beta1, self.beta2, self.eps, self.global_step),
                    lr,
                    weight_decay,
                    params,
                    grads,
                    self.moments1,
                    self.moments2,
                )
            else:
                optim_result = self.hyper_map(
                    F.partial(_adamw_opt, self.op_mul, self.op_pow, self.op_sqrt,
                              self.addcmul, self.op_cast, self.beta1, self.beta2, self.eps, self.global_step, lr),
                    weight_decay,
                    params,
                    grads,
                    self.moments1,
                    self.moments2
                )
        else:
            optim_result = self.hyper_map(
                F.partial(_adamw_opt, self.op_mul, self.op_pow, self.op_sqrt,
                          self.addcmul, self.op_cast, self.beta1, self.beta2, self.eps, self.global_step, lr,
                          weight_decay),
                params,
                grads,
                self.moments1,
                self.moments2
            )

        self.hyper_map(_update_params, self._parameters, optim_result, self.all_gather_ops)
