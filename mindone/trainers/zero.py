import logging
from typing import List, Optional
import mindspore as ms
from mindspore import nn, ops
from mindspore.communication import get_group_size, get_rank
from mindspore.communication.management import GlobalComm
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode

from .train_step import TrainOneStepWrapper
from mindone.models.modules.parallel import PARALLEL_MODULE

_logger = logging.getLogger(__name__)


hyper_map = ops.HyperMap()

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


@_stage2_reduce_scatter.register("Function", "Function", "Tensor", "Tensor", "Bool")
def _run_stage2_reduce_scatter(reduce_scatter, allreduce, op_group_size, gradient, need_reduce_scatter):
    if need_reduce_scatter:
        gradient = reduce_scatter(gradient) / op_group_size
    else:
        gradient = allreduce(gradient) / op_group_size
    return gradient


_stage1_split_grad = ops.MultitypeFuncGraph("stage1_split_grad")


@_stage1_split_grad.register("Function", "Function", "Int", "Int", "Tensor", "Bool")
def _run_stage1_split_grad(allreduce, split, op_group_size, op_rank_id, gradient, need_split):
    gradient = allreduce(gradient) / op_group_size
    if need_split:
        gradient = split(gradient)[op_rank_id]
    return gradient


@ms.ms_class
class ZeroHelper:
    """
    Zero redundancy optimizer(ZeRO) build helper.

    - zero_stage is 0: Normal optimizer update.
    - zero_stage is 1: Split optimizer parameters and gradients, manually updating optimizer parameters.
    - zero_stage is 2: Split optimizer parameters, replace gradients allreduce with reducescatter,
        manually updating optimizer parameters.
    - zero_stage is 3: Split optimizer parameters, normal optimizer update.

    Args:
        optimizer (`nn.Optimizer`): Must be the subclass of MindSpore Optimizer.
        zero_stage (`int`, *optional*): Stage setting of ZeRO, default is 0.
        op_group (`str`, *optional*): The name of the optimizer parallel communication group, default is None.
        dp_group (`str`, *optional*): The name of the data parallel communication group, default is None.
        optimizer_offload (`bool`, *optional*): Only take effect when optimizer is AdamWeightDecay, default is False.
        comm_fusion (`dict`, *optional*): A dict contains the types and configurations
            for setting the communication fusion, default is None, turn off the communication fusion. If set a dict,
            turn on the communication fusion.
            Examples: {"allreduce": {"openstate": True, "bucket_size": 5e8},
                       "reduce_scatter": {"openstate": True, "bucket_size": 5e8},
                       "allgather": {"openstate": False, "bucket_size": 5e8},}
        params_list (`List[str]`, *optional*), List of params used for communication in the network,
            default is None, get the params_list use `network.trainable_params()`.
    """

    def __init__(
        self,
        optimizer: nn.Optimizer,
        zero_stage: int = 0,
        op_group: str = None,
        dp_group: str = None,
        optimizer_offload: bool = False,
        comm_fusion: dict = None,
        params_list: Optional[List[str]] = None,
    ):
        self.optimizer = optimizer
        self.zero_stage = zero_stage
        self.op_group = op_group
        self.ori_parameters = self.optimizer._parameters
        # Init parallel settings
        self.is_parallel = _get_parallel_mode() == ParallelMode.DATA_PARALLEL
        if not self.is_parallel and self.zero_stage != 0:
            _logger.warning("Not in DATA_PARALLEL, set zero_stage to 0.")
            self.zero_stage = 0
        self.split_op = ops.Identity()
        self.op_allgather = ops.Identity()
        self.op_reduce_scatter = ops.Identity()
        self.op_allreduce = ops.Identity()
        self.dp_allreduce = ops.Identity()
        self.op_group_size = get_group_size(self.op_group) if self.is_parallel else 1
        self.op_rank_id = get_rank(self.op_group) if self.is_parallel else 0
        self.need_dp = False
        self.dp_group = dp_group
        self.last_assign = False
        self.dp_group_size = 1
        self.need_allgather = tuple([False] * len(self.optimizer._parameters))

        if self.zero_stage in [1, 2, 3] and self.is_parallel:
            if comm_fusion is None:
                self.set_comm_ops()
            self.split_op = ops.Split(0, self.op_group_size)  # optimizer parallel split

        self.hyper_map = ops.HyperMap()
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
            f"op_group_size: {self.op_group_size}, "
            f"op_rank_id: {self.op_rank_id}, "
            f"dp_group_size: {self.dp_group_size}."
        )

    def set_comm_ops(self,):
        self.op_allreduce = ops.AllReduce(op=ops.ReduceOp.SUM, group=self.op_group)
        if self.zero_stage == 2:
            self.op_reduce_scatter = ops.ReduceScatter(op=ops.ReduceOp.SUM, group=self.op_group)
        if self.zero_stage in [1, 2]:
            # AllGather the parameters after optimizer calculate to update the parameters in train network.
            self.op_allgather = ops.AllGather(group=self.op_group)
        self.need_dp = self.dp_group is not None
        if self.need_dp:
            # Set it when op_group is not the WORLD_COMM_GROUP.
            self.dp_allreduce = ops.AllReduce(op=ops.ReduceOp.SUM, group=self.dp_group)
            self.dp_group_size = ms.Tensor(get_group_size(group=self.dp_group), ms.float32)

    def set_fusion_comm_ops(self, params_list):


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
        if self.zero_stage in [1, 2] and self.is_parallel:
            _logger.info("Clone optimizer.parameters, will increase memory.")
            # Because the first input of MindSpore optimizer must be ms.Parameter,
            # copy optimizer.parameters for optimizer parameters update.
            # It will increase 1/n parameters' memory.
            self.optimizer.parameters = self.optimizer.parameters.clone(prefix="wrapper", init="same")
            self.optimizer._parameters = self.optimizer.parameters
            self.last_assign = True

        self.need_allgather = [False] * len(self.optimizer._parameters)
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
                    _logger.debug(f"Optimizer {param_tuple[i].name} " f"from {ori_shape} to {param_tuple[i].shape}")
        self.need_allgather = tuple(self.need_allgather)

    def reduce_scatter_gradients(self, gradients):
        dtype = gradients[0].dtype
        gradients = self.hyper_map(
            ops.partial(
                _stage2_reduce_scatter,
                self.op_reduce_scatter,
                self.op_allreduce,
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
                self.op_allreduce,
                self.split_op,
                self.op_group_size,
                self.op_rank_id,
            ),
            gradients,
            self.need_allgather,
        )
        return gradients

    def cal_gradients(self, gradients):
        if self.zero_stage == 1:
            gradients = self.split_gradients(gradients)
        if self.zero_stage == 2:
            gradients = self.reduce_scatter_gradients(gradients)
        if self.need_dp:
            gradients = self.dp_allreduce_gradients(gradients)
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


def get_cell_dtype(cell):
    if getattr(cell, "fp16", False):
        return ms.float16
    if getattr(cell, "fp32", False):
        return ms.float32
    if getattr(cell, "bf16", False):
        return ms.bfloat16
    return None


def _init_parallel_settings(net, op_group):
    for module, parallel_module in PARALLEL_MODULE.items():
        if isinstance(net, module):
            cell_type = get_cell_dtype(net)
            new_net = parallel_module(net, 3, op_group)
            if cell_type is not None:
                new_net.to_float(cell_type)
            return new_net
    return None


def _prepare_network(network: nn.Cell, op_group: str):
    new_net = _init_parallel_settings(network, op_group)
    if new_net is not None:
        return new_net
    for name, sub_net in network._cells.items():
        if not sub_net:
            continue
        new_sub_net = _init_parallel_settings(sub_net, op_group)
        if new_sub_net is not None:
            network.__setattr__(name, new_sub_net)
            continue
        if sub_net._params:
            for param_name in sub_net._params:
                param = getattr(sub_net, param_name)
                _logger.warning(f"Set param {param.name} parallel_optimizer False, param shape {param.shape}")
                param.parallel_optimizer = False
        _prepare_network(sub_net, op_group)
    return network


def prepare_network(network: nn.Cell, zero_stage: int = 0, op_group: str = None):
    if zero_stage != 3 or _get_parallel_mode() != ParallelMode.DATA_PARALLEL:
        _logger.info("No need rewrite network and return original network.")
        return network
    _logger.info("Rewrite the network, please wait...")
    network = _prepare_network(network, op_group)
    return network


def prepare_train_network(
    network: nn.Cell,
    optimizer: nn.Optimizer,
    scale_sense: float = 1.0,
    ema: nn.Cell = None,
    updates: int = 0,
    drop_overflow_update: bool = True,
    gradient_accumulation_steps: int = 1,
    clip_grad: bool = False,
    clip_norm: float = 1.0,
    verbose: bool = False,
    zero_stage: int = 0,
    optimizer_offload: bool = False,
    op_group: str = None,
    dp_group: str = None,
    comm_fusion: dict = None,
    params_list: Optional[List[str]] = None,
):
    """
    Prepare network and optimizer for distributed training.

    Args:
        network (`nn.Cell`): train network, not include grad function,
            grad function must be built after rewrite train network.
        optimizer (`nn.Optimizer`): Must be the subclass of MindSpore Optimizer.
        scale_sense (Union[Tensor, Cell]): If this value is a Cell, it will be called
            to update loss scale. If this value is a Tensor, the loss scale can be modified by `set_sense_scale`,
            the shape should be :math:`()` or :math:`(1,)`.
        zero_stage (`int`, *optional*): Stage setting of ZeRO, default is 0.
        optimizer_offload (`bool`, *optional*): Only take effect when optimizer is AdamWeightDecay, default is False.
        op_group (`str`, *optional*): The name of the optimizer parallel communication group, default is None.
        dp_group (`str`, *optional*): The name of the data parallel communication group, default is None.
        comm_fusion (`dict`, *optional*): A dict contains the types and configurations
            for setting the communication fusion, default is None, turn off the communication fusion. If set a dict,
            turn on the communication fusion.
            Examples: {"allreduce": {"openstate": True, "bucket_size": 5e8},
                       "reduce_scatter": {"openstate": True, "bucket_size": 5e8},
                       "allgather": {"openstate": False, "bucket_size": 5e8},}
        params_list (`List[str]`, *optional*), List of params used for communication in the network,
            default is None, get the params_list use `network.trainable_params()`.
    """
    is_parallel = _get_parallel_mode() == ParallelMode.DATA_PARALLEL
    if not is_parallel and zero_stage == 0:
        _logger.info("No need prepare train_network with zero.")
        return network, optimizer

    if zero_stage not in [0, 1, 2, 3]:
        raise ValueError("Not support zero_stage {zero_stage}")
    if op_group is None:
        _logger.warning("Not set zero group, set it WORLD_COMM_GROUP.")
        op_group = GlobalComm.WORLD_COMM_GROUP
    if op_group != GlobalComm.WORLD_COMM_GROUP and dp_group is None:
        raise ValueError("op_group {op_group} and dp_group {dp_group} not full network hccl group coverage")

    new_network = prepare_network(network, zero_stage, op_group)
    zero_helper = ZeroHelper(optimizer, zero_stage, op_group, dp_group, optimizer_offload, comm_fusion, params_list)
    if isinstance(scale_sense, float):
        scale_sense = ms.Tensor(scale_sense, ms.float32)
    train_network = TrainOneStepWrapper(
        new_network,
        optimizer,
        scale_sense=scale_sense,
        ema=ema,
        updates=updates,
        drop_overflow_update=drop_overflow_update,
        gradient_accumulation_steps=gradient_accumulation_steps,
        clip_grad=clip_grad,
        clip_norm=clip_norm,
        verbose=verbose,
        zero_helper=zero_helper,
    )
    return train_network
