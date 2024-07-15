import logging

import mindspore as ms
from mindspore import nn, ops
from mindspore.communication import get_group_size, get_rank
from mindspore.communication.management import GlobalComm
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode
from mindspore.rewrite import NodeType, SymbolTree

from .train_step import TrainOneStepWrapper

_logger = logging.getLogger(__name__)


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
        self.need_rewrite = self.check_rewrite(param)
        # Init parallel settings
        self.is_parallel = _get_parallel_mode() == ParallelMode.DATA_PARALLEL
        self.op_group_size = get_group_size(self.op_group) if self.is_parallel else 1
        self.allgather = ops.Identity()
        self.reduce_scatter = None
        if self.need_rewrite and self.zero_stage == 3:
            self.op_allgather = ops.AllGather(group=self.op_group)
            self.op_reduce_scatter = ops.ReduceScatter(group=self.op_group, op=ops.ReduceOp.SUM)

    def check_rewrite(self, param):
        """Check the parameter need to split or not."""
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


def prepare_network(network: nn.Cell, zero_stage: int = 0, op_group: str = None):
    if zero_stage != 3 or _get_parallel_mode() != ParallelMode.DATA_PARALLEL:
        _logger.info("No need rewrite network and return original network.")
        return network
    op_rank_id = get_rank(op_group)
    op_group_size = get_group_size(op_group)
    _logger.info("Rewrite the network, please wait...")
    _prepare_network(network, op_group, op_group_size, op_rank_id)
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
        zero_stage=zero_stage,
        optimizer_offload=optimizer_offload,
        op_group=op_group,
        dp_group=dp_group,
    )
    return train_network
