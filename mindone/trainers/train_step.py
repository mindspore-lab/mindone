"""Train step wrapper supporting setting drop overflow update, ema etc"""
import logging

from packaging import version

import mindspore as ms
import mindspore.context as context
from mindspore import Tensor, nn, ops
from mindspore.boost.grad_accumulation import gradient_accumulation_op as _grad_accum_op
from mindspore.boost.grad_accumulation import gradient_clear_op as _grad_clear_op
from mindspore.common import RowTensor
from mindspore.common import dtype as mstype
from mindspore.communication import get_group_size, get_rank
from mindspore.context import ParallelMode
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.parallel._utils import _get_parallel_mode

_logger = logging.getLogger(__name__)

_grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()
_grad_overflow = C.MultitypeFuncGraph("_grad_overflow")


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * F.cast(reciprocal(scale), F.dtype(grad))


@_grad_scale.register("Tensor", "RowTensor")
def tensor_grad_scale_row_tensor(scale, grad):
    return RowTensor(
        grad.indices,
        grad.values * F.cast(reciprocal(scale), F.dtype(grad.values)),
        grad.dense_shape,
    )


_optim_allgather = ops.MultitypeFuncGraph("optim_allgather")


@_optim_allgather.register("Function", "Bool", "Tensor", "Tensor", "Bool")
def _run_optim_allgather(allgather, last_assign, variable, value, need_allgather):
    if need_allgather:
        value = allgather(value)
    if last_assign:
        ops.assign(variable, value)
    return True


_dp_allreduce = ops.MultitypeFuncGraph("dp_allreduce")


@_dp_allreduce.register("Function", "Tensor", "Tensor")
def _run_dp_allreduce(dp_allreduce, dp_group_size, gradient):
    gradient = dp_allreduce(gradient) / dp_group_size
    return gradient


_stage2_reduce_scatter = ops.MultitypeFuncGraph("stage2_reduce_scatter")


@_stage2_reduce_scatter.register("Function", "Tensor", "Tensor", "Bool")
def _run_stage2_reduce_scatter(reduce_scatter, op_group_size, gradient, need_reduce_scatter):
    if need_reduce_scatter:
        gradient = reduce_scatter(gradient) / op_group_size
    return gradient


_stage1_split_grad = ops.MultitypeFuncGraph("stage1_split_grad")


@_stage1_split_grad.register("Function", "Int", "Tensor", "Bool")
def _run_stage1_split_grad(split, op_rank_id, gradient, need_split):
    if need_split:
        gradient = split(gradient)[op_rank_id]
    return gradient


class TrainOneStepWrapper(nn.TrainOneStepWithLossScaleCell):
    """TrainStep with ema and clip grad.

    Args:
        drop_overflow_update: if True, network will not be updated when gradient is overflow.
        scale_sense (Union[Tensor, Cell]): If this value is a Cell, it will be called
            to update loss scale. If this value is a Tensor, the loss scale can be modified by `set_sense_scale`,
            the shape should be :math:`()` or :math:`(1,)`.
        zero_stage (`int`, *optional*): Stage setting of ZeRO, default is 0.
            - zero_stage is 0: Normal optimizer update.
            - zero_stage is 1: Split optimizer parameters and gradients, manually updating optimizer parameters.
            - zero_stage is 2: Split optimizer parameters, replace gradients allreduce with reducescatter,
                manually updating optimizer parameters.
            - zero_stage is 3: Split optimizer parameters, normal optimizer update.
        optimizer_offload (`bool`, *optional*): Only take effect when optimizer is AdamWeightDecay, default is False.
        op_group (`str`, *optional*): The name of the optimizer parallel communication group, default is None.
        dp_group (`str`, *optional*): The name of the data parallel communication group, default is None.

    Returns:
        Tuple of 3 Tensor, the loss, overflow flag and current loss scale value.
        loss (Tensor) -  A scalar, the loss value.
        overflow (Tensor) -  A scalar, whether overflow occur or not, the type is bool.
        loss scale (Tensor) -  The loss scale value, the shape is :math:`()` or :math:`(1,)`.

    """

    def __init__(
        self,
        network,
        optimizer,
        scale_sense=1.0,
        ema=None,
        updates=0,
        drop_overflow_update=True,
        gradient_accumulation_steps=1,
        clip_grad=False,
        clip_norm=1.0,
        verbose=False,
        zero_stage: int = 0,
        optimizer_offload: bool = False,
        op_group: str = None,
        dp_group: str = None,
    ):
        super().__init__(network, optimizer, scale_sense)
        self.ema = ema
        self.drop_overflow_update = drop_overflow_update

        assert isinstance(clip_grad, bool), f"Invalid type of clip_grad, got {type(clip_grad)}, expected bool"
        assert clip_norm > 0.0 and isinstance(clip_norm, float), f"clip_norm must be float > 1.0, but got {clip_norm}"
        self.clip_grad = clip_grad
        self.clip_norm = clip_norm

        assert gradient_accumulation_steps >= 1
        self.accum_steps = gradient_accumulation_steps
        if gradient_accumulation_steps > 1:
            self.accumulated_grads = optimizer.parameters.clone(prefix="grad_accumulated_", init="zeros")

            self.cur_accum_step = ms.Parameter(ms.Tensor(0, dtype=ms.int32), name="accum_step")
            self.zero = Tensor(0, ms.int32)

        self.verbose = verbose
        self.is_cpu_device = context.get_context("device_target") == "CPU"  # to support CPU in CI
        self.skip_start_overflow_check = version.parse(ms.__version__) >= version.parse("2.1")

        self.map = ops.Map()
        self.partial = ops.Partial()
        # init ZeRO settings
        self.is_data_parallel = _get_parallel_mode() == ParallelMode.DATA_PARALLEL
        self.zero_stage = zero_stage
        if not self.is_data_parallel:
            self.zero_stage = 0
        if self.zero_stage != 0:
            self.op_group = op_group
            self.ori_parameters = self.optimizer._parameters
            self.split_op = ops.Identity()
            self.op_allgather = ops.Identity()
            self.op_reduce_scatter = ops.Identity()
            self.dp_allreduce = ops.Identity()
            self.op_group_size = get_group_size(self.op_group) if self.is_data_parallel else 1
            self.op_rank_id = get_rank(self.op_group) if self.is_data_parallel else 0
            self.need_dp = False
            self.last_assign = False
            self.dp_group_size = 1
            self.need_allgather = [False] * len(self.optimizer._parameters)
            if self.zero_stage in [2, 3]:
                self.grad_reducer = nn.Identity()
            if self.zero_stage in [1, 2]:
                _logger.info("Clone optimizer.parameters, will increase memory.")
                # Because the first input of MindSpore optimizer must be ms.Parameter,
                # copy optimizer.parameters for optimizer parameters update.
                # It will increase 1/n parameters' memory.
                self.optimizer.parameters = self.optimizer.parameters.clone(prefix="wrapper", init="same")
                self.optimizer._parameters = self.optimizer.parameters
                self.last_assign = True
            if self.zero_stage in [1, 2, 3]:
                if self.zero_stage == 2:
                    self.op_reduce_scatter = ops.ReduceScatter(op=ops.ReduceOp.SUM, group=self.op_group)
                if self.zero_stage in [1, 2]:
                    # AllGather the parameters after optimizer calculate to update the parameters in train network.
                    self.op_allgather = ops.AllGather(group=self.op_group)
                self.need_dp = dp_group is not None
                if self.need_dp:
                    # Set it when op_group is not the WORLD_COMM_GROUP.
                    self.dp_allreduce = ops.AllReduce(op=ops.ReduceOp.SUM, group=dp_group)
                    self.dp_group_size = ms.Tensor(get_group_size(group=dp_group), ms.float32)
                self.split_op = ops.Split(0, self.op_group_size)  # optimizer parallel split
                self.split_params()
            self.need_allgather = tuple(self.need_allgather)
            if optimizer_offload:
                if isinstance(self.optimizer, nn.AdamWeightDecay):
                    nn.AdamWeightDecay.target("CPU")
                    _logger.info("Set optimizer run offload.")
                else:
                    _logger.warning("optimizer_offload only take effect when optimizer is AdamWeightDecay.")
                    optimizer_offload = False
            _logger.info(
                f"Build TrainOneStepWrapper with ZeRO stage: {self.zero_stage}, "
                f"optimizer_offload: {optimizer_offload}, "
                f"op_group_size: {self.op_group_size} "
                f"op_rank_id: {self.op_rank_id} "
                f"dp_group_size: {self.dp_group_size} "
            )

    def split_param(self, param):
        return self.split_op(param)[self.op_rank_id]

    def get_optimizer_param_tuples(self):
        param_tuples = []
        if ms.get_context("mode") == ms.PYNATIVE_MODE:
            for name in self.optimizer._params_list:
                if name in ["_parameters", "parameters"]:
                    continue
                _logger.debug(f"Add optimizer param_tuples {name}")
                param_tuples.append(getattr(self.optimizer, name))
        else:
            for attr in self.optimizer.__dict__:
                if isinstance(getattr(self.optimizer, attr), ms.ParameterTuple):
                    if attr in ["_parameters", "parameters"]:
                        continue
                    _logger.debug(f"Add optimizer param_tuples {attr}")
                    param_tuples.append(getattr(self.optimizer, attr))
        return param_tuples

    def split_params(self):
        param_tuples = self.get_optimizer_param_tuples()
        for i, param in enumerate(self.optimizer._parameters):
            _logger.debug(f"Split optimizer param {param.name} {param.shape}")
            # If zero_stage is 3, the parameters in train network have been split,
            # use parameter in param_tuples to get batch size.
            if self.zero_stage == 3:
                if param_tuples:
                    B = param_tuples[0][i].shape[0]
                else:
                    continue
            else:
                B = param.shape[0]
            _logger.debug(f"Do split with zero_stage {self.zero_stage}")
            if param.parallel_optimizer and B >= self.op_group_size and B % self.op_group_size == 0:
                if self.zero_stage in [1, 2]:
                    self.need_allgather[i] = True
                    ori_shape = param.shape
                    param.assign_value(self.split_param(param))
                    _logger.debug(f"Optimizer {param.name} from {ori_shape} to {param.shape}")
                for param_tuple in param_tuples:
                    ori_shape = param_tuple[i].shape
                    param_tuple[i].assign_value(self.split_param(param_tuple[i]))
                    _logger.debug(f"Optimizer {param_tuple[i].name} from {ori_shape} to {param_tuple[i].shape}")

    def reduce_scatter_gradients(self, gradients):
        dtype = gradients[0].dtype
        gradients = self.hyper_map(
            ops.partial(
                _stage2_reduce_scatter,
                self.op_reduce_scatter,
                ms.Tensor(self.op_group_size, dtype),
            ),
            gradients,
            self.need_allgather,
        )
        return gradients

    def dp_allreduce_gradients(self, gradients):
        dtype = gradients[0].dtype
        gradients = self.hyper_map(
            ops.partial(
                _dp_allreduce,
                self.dp_allreduce,
                ms.Tensor(self.dp_group_size, dtype),
            ),
            gradients,
        )
        return gradients

    def split_gradients(self, gradients):
        gradients = self.hyper_map(
            ops.partial(
                _stage1_split_grad,
                self.split_op,
                self.op_rank_id,
            ),
            gradients,
            self.need_allgather,
        )
        return gradients

    def run_optimizer(self, grads):
        optim_result = self.optimizer(grads)
        if self.zero_stage == 1 or self.zero_stage == 2:
            optim_result = ops.depend(
                self.hyper_map(
                    ops.partial(_optim_allgather, self.op_allgather, self.last_assign),
                    self.ori_parameters,
                    self.optimizer._parameters,
                    self.need_allgather,
                ),
                optim_result,
            )
        return optim_result

    def construct(self, *inputs):
        # compute loss
        weights = self.weights
        loss = self.network(*inputs)  # mini-batch loss
        scaling_sens = self.scale_sense

        # check loss overflow. (after ms2.1, it's done together with gradient overflow checking)
        if self.skip_start_overflow_check:
            status = Tensor([0] * 8, mstype.int32)
        else:
            if not self.is_cpu_device:
                status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
            else:
                status = None

        scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))  # loss scale value

        # 1. compute gradients (of the up-scaled loss w.r.t. the model weights)
        grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)
        if self.zero_stage == 1:
            grads = self.split_gradients(grads)
        if self.zero_stage == 2:
            grads = self.reduce_scatter_gradients(grads)
        if self.need_dp:
            grads = self.dp_allreduce_gradients(grads)
        if self.accum_steps == 1:
            grads = self.grad_reducer(grads)
            scaling_sens = ops.depend(scaling_sens, grads)

        # 2. down-scale gradients by loss_scale. grads = grads / scaling_sense  / grad_accum_steps
        # also divide gradients by accumulation steps to avoid taking mean of  the accumulated gradients later
        grads = self.hyper_map(F.partial(_grad_scale, scaling_sens), grads)  # accum_steps division is done later

        # 3. check gradient overflow
        if not self.is_cpu_device:
            cond = self.get_overflow_status(status, grads)
            overflow = self.process_loss_scale(cond)
        else:
            overflow = ms.Tensor(False)
            cond = ms.Tensor(False)

        # accumulate gradients and update model weights if no overflow or allow to update even when overflow
        if (not self.drop_overflow_update) or (not overflow):
            # 4. gradient accumulation if enabled
            if self.accum_steps > 1:
                # self.accumulated_grads += grads / accum_steps
                loss = F.depend(
                    loss, self.hyper_map(F.partial(_grad_accum_op, self.accum_steps), self.accumulated_grads, grads)
                )

                # self.cur_accum_step += 1
                loss = F.depend(loss, ops.assign_add(self.cur_accum_step, Tensor(1, ms.int32)))

                if self.cur_accum_step >= self.accum_steps:
                    # 5. gradient reduction on distributed GPUs/NPUs
                    grads = self.grad_reducer(self.accumulated_grads)

                    # 6. clip grad
                    if self.clip_grad:
                        grads = ops.clip_by_global_norm(grads, self.clip_norm)
                    # 7. optimize
                    loss = F.depend(loss, self.run_optimizer(grads))

                    # clear gradient accumulation states
                    loss = F.depend(loss, self.hyper_map(F.partial(_grad_clear_op), self.accumulated_grads))
                    # self.cur_accum_step = 0
                    loss = F.depend(loss, ops.assign(self.cur_accum_step, self.zero))
                else:
                    # update LR in each gradient step but not optimize net parameter
                    # to ensure the LR curve is consistent
                    # FIXME: for ms>=2.2, get_lr() will not increase global step by 1. we need to do it manually.
                    loss = F.depend(loss, self.optimizer.get_lr())
            else:
                # 5. gradient reduction on distributed GPUs/NPUs
                # 6. clip grad
                if self.clip_grad:
                    grads = ops.clip_by_global_norm(grads, self.clip_norm)
                # 7. optimize
                loss = F.depend(loss, self.run_optimizer(grads))

            # 8.ema
            if self.ema is not None:
                self.ema.ema_update()
        # else:
        #    print("WARNING: Gradient overflow! update skipped.") # TODO: recover it after 910B in-graph print issue fixed

        return loss, cond, scaling_sens
