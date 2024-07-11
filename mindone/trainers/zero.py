import logging

import mindspore as ms
from mindspore import nn, ops
from mindspore.communication import get_group_size, get_rank
from mindspore.communication.management import GlobalComm
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode
from mindspore.rewrite import NodeType, SymbolTree

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


class ZeroOptimizerWrapper(nn.Cell):
    """
    Building a zero optimizer with zero_stage.

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
    """

    def __init__(self, optimizer: nn.Optimizer, zero_stage: int = 0, op_group: str = None, dp_group: str = None):
        super().__init__(auto_prefix=False)
        self.optimizer = optimizer
        self.zero_stage = zero_stage
        self.op_group = op_group
        self.is_parallel = _get_parallel_mode() == ParallelMode.DATA_PARALLEL
        self.split_op = ops.Identity()
        self.need_allgather = [False] * len(self.optimizer._parameters)
        self.op_allgather = ops.Identity()
        self.op_reduce_scatter = ops.Identity()
        self.dp_allreduce = ops.Identity()
        self.op_group_size = get_group_size(self.op_group) if self.is_parallel else 1
        self.op_rank_id = get_rank(self.op_group) if self.is_parallel else 0
        self.ori_parameters = self.optimizer._parameters
        self.need_dp = False
        self.last_assign = False
        self.dp_group_size = 1
        if self.zero_stage in [1, 2] and self.is_parallel:
            _logger.info("Clone optimizer.parameters, will increase memory.")
            self.optimizer.parameters = self.optimizer.parameters.clone(prefix="wrapper", init="same")
            self.optimizer._parameters = self.optimizer.parameters
            self.last_assign = True
        if self.zero_stage in [1, 2, 3] and self.is_parallel:
            if self.zero_stage == 2:
                self.op_reduce_scatter = ops.ReduceScatter(op=ops.ReduceOp.SUM, group=self.op_group)
            if self.zero_stage in [1, 2]:
                self.op_allgather = ops.AllGather(group=self.op_group)
            self.need_dp = dp_group != None
            if self.need_dp:
                self.dp_allreduce = ops.AllReduce(op=ops.ReduceOp.SUM, group=dp_group)
                self.dp_group_size = ms.Tensor(get_group_size(group=dp_group), ms.float32)
            self.split_op = ops.Split(0, self.op_group_size)
            self.split_params()
        self.need_allgather = tuple(self.need_allgather)
        self.hyper_map = ops.HyperMap()
        if self.zero_stage != 2:
            self._parameters = self.ori_parameters
            self.parameters = self.ori_parameters
        else:
            self._parameters = self.ori_parameters
            self.parameters = self.ori_parameters

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

    def construct(self, gradients):
        if self.zero_stage == 1:
            gradients = self.split_gradients(gradients)
        if self.zero_stage == 2:
            gradients = self.reduce_scatter_gradients(gradients)
        if self.need_dp:
            gradients = self.dp_allreduce_gradients(gradients)
        optim_result = self.optimizer(gradients)
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


class ZeroParamWrapper(nn.Cell):
    """
    a cell to Insert communication operators before and after parameters when `zero_stage == 3`.
    """

    def __init__(self, param: ms.Parameter, zero_stage: int = 0, op_group: str = GlobalComm.WORLD_COMM_GROUP):
        super().__init__(auto_prefix=False)
        self.op_group = op_group
        self.zero_stage = zero_stage
        if zero_stage not in [2, 3]:
            raise ValueError(f"ZeroParamWrapper not support zero_stage {zero_stage}.")
        self.zero_stage = zero_stage
        self.is_parallel = _get_parallel_mode() == ParallelMode.DATA_PARALLEL
        self.op_group_size = get_group_size(self.op_group) if self.is_parallel else 1
        self.need_rewrite = self.check_rewrite(param)
        self.allgather = ops.Identity()
        self.reduce_scatter = None
        if self.need_rewrite and self.zero_stage == 3:
            self.op_allgather = ops.AllGather(group=self.op_group)
            self.op_reduce_scatter = ops.ReduceScatter(group=self.op_group, op=ops.ReduceOp.SUM)

    def check_rewrite(self, param):
        need_rewrite = self.is_parallel and self.zero_stage == 3
        B = param.shape[0]
        if not param.parallel_optimizer or B < self.op_group_size or B % self.op_group_size != 0:
            need_rewrite = False
        return need_rewrite

    def construct(self, param):
        if self.need_rewrite:
            return self.op_allgather(param)
        return param

    def bprop(self, param, out, dout):
        if self.need_rewrite:
            r = self.op_reduce_scatter(dout) / self.op_group_size
            return (r,)
        return (dout,)


def get_cell_dtype(cell):
    if cell.fp16:
        return ms.float16
    if cell.fp32:
        return ms.float32
    if cell.bf16:
        return ms.bfloat16
    return None


def rewrite_node(node, cell):
    rewrite_params = []
    for i, arg in enumerate(node.get_args()):
        if arg.scope == "self" and isinstance(getattr(cell, arg.value), ms.Parameter):
            node.set_arg(i, f"self.param_w_{arg.value}(self.{arg.value})")
            _logger.debug(f"Rewrite {arg.value} with ZeroParamWrapper.")
            rewrite_params.append(arg.value)
    return rewrite_params


def rewrite_cell(cell: nn.Cell):
    """
    Rewrite the cell. Add ZeroParamWrapper to all parameters.
    """
    stree = SymbolTree.create(cell)
    rewrite_params = []
    for node in stree.nodes():
        rewrite_params = rewrite_params + rewrite_node(node, cell)
        if node.get_node_type() == NodeType.ControlFlow:
            all_nodes = [ms.rewrite.Node(n) for n in node.get_handler().nodes()]
            for sub_node in all_nodes:
                rewrite_params = rewrite_params + rewrite_node(sub_node, cell)
    if rewrite_params:
        return rewrite_params, stree.get_network()
    return None


def get_cell_params_fullname_dict(cell: nn.Cell):
    fullname_dict = {}
    for param_name in cell._params:
        fullname_dict[param_name] = getattr(cell, param_name).name
    return fullname_dict


def _prepare_network(network: nn.Cell, op_group: str, op_group_size: int = 1, op_rank_id: int = 0):
    for name, sub_net in network._cells.items():
        if not sub_net:
            continue
        if sub_net._params:
            params_fullname_dict = get_cell_params_fullname_dict(sub_net)
            rewrite_res = rewrite_cell(sub_net)
            if rewrite_res is not None:
                rewrite_params, new_cell = rewrite_res
                _logger.debug(f"Rewrite cell {name} with params {rewrite_params}")
                network.__setattr__(name, new_cell)

                # parameter name will update after __setattr__, reset to ori parameter name.
                for param_name in rewrite_params:
                    getattr(new_cell, param_name).name = params_fullname_dict[param_name]

                for param_name in rewrite_params:
                    param = getattr(sub_net, param_name)
                    # Set zero_param_wrapper same type with sub_net
                    cell_type = get_cell_dtype(sub_net)
                    if cell_type:
                        zero_param_wrapper = ZeroParamWrapper(param, zero_stage=3, op_group=op_group).to_float(
                            cell_type
                        )
                    new_cell.__setattr__(f"param_w_{param_name}", zero_param_wrapper)
                    if zero_param_wrapper.need_rewrite:
                        split_op = ops.Split(0, op_group_size)
                        ori_shape = param.shape
                        new_cell.__getattr__(param_name).assign_value(split_op(param)[op_rank_id])
                        _logger.debug(f"Cell {name} split {param_name} from {ori_shape} to {param.shape}")

        _prepare_network(sub_net, op_group, op_group_size, op_rank_id)


def prepare_network(network: nn.Cell, zero_stage: int = 0, op_group: int = None):
    if zero_stage != 3 or _get_parallel_mode() != ParallelMode.DATA_PARALLEL:
        _logger.info("No need rewrite network and return original network.")
        return network
    op_rank_id = get_rank(op_group)
    op_group_size = get_group_size(op_group)
    _prepare_network(network, op_group, op_group_size, op_rank_id)
    return network


def prepare_optimizer(optimizer: nn.Optimizer, zero_stage: int = 0, op_group: str = None, dp_group: str = None):
    if zero_stage == 0 or _get_parallel_mode() != ParallelMode.DATA_PARALLEL:
        _logger.info(f"No need prepare optimizer with zero.")
        return optimizer
    if not isinstance(optimizer, nn.Optimizer):
        raise ValueError("optimizer must be MindSpore Optimizer.")
    optimizer = ZeroOptimizerWrapper(optimizer, zero_stage, op_group, dp_group)
    return optimizer


def prepare_train_network(
    network: nn.Cell, optimizer: nn.Optimizer, zero_stage: int = 0, op_group: str = None, dp_group: str = None
):
    """
    Prepare network and optimizer for distributed training.

    Args:
        network (`nn.Cell`): train network, not include grad function,
            grad function must be built after rewrite train network.
        optimizer (`nn.Optimizer`): Must be the subclass of MindSpore Optimizer.
        zero_stage (`int`, *optional*): Stage setting of ZeRO, default is 0.
        op_group (`str`, *optional*): The name of the optimizer parallel communication group, default is None.
        dp_group (`str`, *optional*): The name of the data parallel communication group, default is None.
    """
    is_parallel = _get_parallel_mode() == ParallelMode.DATA_PARALLEL
    if not is_parallel and zero_stage == 0:
        _logger.info(f"No need prepare train_network with zero.")
        return network, optimizer

    if zero_stage not in [0, 1, 2, 3]:
        raise ValueError("Not support zero_stage {zero_stage}")
    if op_group is None:
        _logger.warning("Not set zero group, set it WORLD_COMM_GROUP.")
        op_group = GlobalComm.WORLD_COMM_GROUP
    if op_group != GlobalComm.WORLD_COMM_GROUP and dp_group is None:
        raise ValueError("op_group {op_group} and dp_group {dp_group} not full network hccl group coverage")

    new_network = prepare_network(network, zero_stage, op_group)
    new_optimizer = prepare_optimizer(optimizer, zero_stage, op_group, dp_group)
    return new_network, new_optimizer
