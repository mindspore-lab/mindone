import json
import logging
import os
from typing import Literal

import mindspore as ms
from mindspore import nn, ops
from mindspore.communication import get_group_size, get_rank
from mindspore.communication.management import GlobalComm
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode

from mindone.models.modules.parallel import PARALLEL_MODULES

from .train_step import TrainOneStepWrapper

_logger = logging.getLogger(__name__)


hyper_map = ops.HyperMap()

_optim_allgather = ops.MultitypeFuncGraph("optim_allgather")


@_optim_allgather.register("Bool", "Function", "Tensor", "Tensor", "Bool")
def _run_optim_allgather(last_assign, allgather, variable, value, need_parameter_split):
    if need_parameter_split:
        value = allgather(value)
    if last_assign:
        ops.assign(variable, value)
    return True


_dp_allreduce = ops.MultitypeFuncGraph("dp_allreduce")


@_dp_allreduce.register("Tensor", "Function", "Tensor")
def _run_dp_allreduce(dp_group_size, dp_allreduce, gradient):
    gradient = dp_allreduce(gradient) / dp_group_size
    return gradient


_stage2_reduce_scatter = ops.MultitypeFuncGraph("stage2_reduce_scatter")


@_stage2_reduce_scatter.register("Tensor", "Function", "Function", "Tensor", "Bool")
def _run_stage2_reduce_scatter(op_group_size, reduce_scatter, allreduce, gradient, need_reduce_scatter):
    if need_reduce_scatter:
        gradient = reduce_scatter(gradient) / op_group_size
    else:
        gradient = allreduce(gradient) / op_group_size
    return gradient


_stage1_split_grad = ops.MultitypeFuncGraph("stage1_split_grad")


@_stage1_split_grad.register("Function", "Int", "Int", "Function", "Tensor", "Bool")
def _run_stage1_split_grad(split, op_group_size, op_rank_id, allreduce, gradient, need_split):
    gradient = allreduce(gradient) / op_group_size
    if need_split:
        gradient = split(gradient)[op_rank_id]
    return gradient


def split_np(x, num, idx):
    b = x.shape[0]
    sp_len = b // num
    start = sp_len * idx
    end = sp_len * (idx + 1)
    return ms.Tensor(x.asnumpy()[start:end])


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
            Examples: {"allreduce": {"bucket_size": 5e8},
                       "reduce_scatter": {"bucket_size": 5e8},
                       "allgather": {"bucket_size": 5e8},}
        params_split_info (`str`, *optional*): A json path of the optimizer parallel communication group,
            default is `params_info`.
    """

    def __init__(
        self,
        optimizer: nn.Optimizer,
        zero_stage: int = 0,
        op_group: str = None,
        dp_group: str = None,
        optimizer_offload: bool = False,
        comm_fusion: dict = None,
        params_split_info: str = "params_info",
    ):
        self.optimizer = optimizer
        self.zero_stage = zero_stage
        self.op_group = op_group
        if isinstance(optimizer, ms.experimental.optim.optimizer.Optimizer):
            self.optimizer._parameters = self.optimizer.parameters
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
        self.need_parameter_split = tuple([False] * len(self.optimizer._parameters))
        self.use_comm_fusion = False
        if self.zero_stage in [1, 2, 3] and self.is_parallel:
            self.split_op = ops.Split(0, self.op_group_size)  # optimizer parallel split
            self.get_need_parameter_split()
            if comm_fusion is None:
                self.set_comm_ops()
            else:
                self.use_comm_fusion = True
                self.max_fusion_id = 0
                if self.zero_stage == 1:
                    self.set_zero1_allreduce_fusion_comm_list(comm_fusion)
                    self.set_optimizer_allgather_fusion_comm_list(comm_fusion)
                if self.zero_stage == 2:
                    self.set_zero2_reduce_scatter_fusion_comm_list(comm_fusion)
                    self.set_optimizer_allgather_fusion_comm_list(comm_fusion)
                if self.need_dp:
                    self.set_dp_allreduce_comm_list(comm_fusion)
            if not os.path.exists(params_split_info):
                os.makedirs(params_split_info, exist_ok=True)
            if not os.path.isdir(params_split_info):
                ValueError(f"params_split_info must be a folder, params_split_info: {params_split_info}")
            self.dump_params_split_info(params_split_info)

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

    def set_comm_ops(
        self,
    ):
        self.op_allreduce = ops.AllReduce(op=ops.ReduceOp.SUM, group=self.op_group)
        self.op_reduce_scatter = ops.ReduceScatter(op=ops.ReduceOp.SUM, group=self.op_group)
        # AllGather the parameters after optimizer calculate to update the parameters in train network.
        self.op_allgather = ops.AllGather(group=self.op_group)

        self.need_dp = self.dp_group is not None
        if self.need_dp:
            # Set it when op_group is not the WORLD_COMM_GROUP.
            self.dp_allreduce = ops.AllReduce(op=ops.ReduceOp.SUM, group=self.dp_group)
            self.dp_group_size = ms.Tensor(get_group_size(group=self.dp_group), ms.float32)

    def update_comm_op_info(self, comm_op_info, bucket_size, param_size, param_name):
        if comm_op_info[-1]["size"] + param_size <= bucket_size or len(comm_op_info[-1]["params"]) == 0:
            comm_op_info[-1]["size"] += param_size
            comm_op_info[-1]["params"].append(param_name)
        else:
            fusion_id = self.max_fusion_id + 1
            self.max_fusion_id += 1
            comm_op_info.append({"size": param_size, "fusion_id": fusion_id, "params": [param_name]})

    def set_zero1_allreduce_fusion_comm_list(self, comm_fusion):
        allreduce_info = [{"size": 0, "fusion_id": self.max_fusion_id + 1, "params": []}]
        self.max_fusion_id += 1
        self.zero1_allreduce_list = []
        for i, param in enumerate(self.ori_parameters):
            param_size = param.itemsize * param.size
            param_name = param.name
            self.update_comm_op_info(allreduce_info, comm_fusion["allreduce"]["bucket_size"], param_size, param_name)
            comm_op = ops.AllReduce(op=ops.ReduceOp.SUM, group=self.op_group)
            comm_op.add_prim_attr("fusion", allreduce_info[-1]["fusion_id"])
            self.zero1_allreduce_list.append(comm_op)
        _logger.info(f"zero1_allreduce_fusion: {allreduce_info}")

    def set_zero2_reduce_scatter_fusion_comm_list(self, comm_fusion):
        reduce_scatter_info = [{"size": 0, "fusion_id": self.max_fusion_id + 1, "params": []}]
        self.max_fusion_id += 1
        allreduce_info = [{"size": 0, "fusion_id": self.max_fusion_id + 1, "params": []}]
        self.max_fusion_id += 1
        self.zero2_reduce_scatter_list = []
        self.zero2_allreduce_list = []
        for i, param in enumerate(self.ori_parameters):
            param_size = param.itemsize * param.size
            param_name = param.name
            if self.need_parameter_split[i]:
                self.update_comm_op_info(
                    reduce_scatter_info, comm_fusion["reduce_scatter"]["bucket_size"], param_size, param_name
                )
            else:
                self.update_comm_op_info(
                    allreduce_info, comm_fusion["allreduce"]["bucket_size"], param_size, param_name
                )
            comm_op = ops.ReduceScatter(op=ops.ReduceOp.SUM, group=self.op_group)
            comm_op.add_prim_attr("fusion", reduce_scatter_info[-1]["fusion_id"])
            self.zero2_reduce_scatter_list.append(comm_op)

            comm_op = ops.AllReduce(op=ops.ReduceOp.SUM, group=self.op_group)
            comm_op.add_prim_attr("fusion", allreduce_info[-1]["fusion_id"])
            self.zero2_allreduce_list.append(comm_op)
        _logger.info(f"zero2_reduce_scatter_fusion: {reduce_scatter_info}")
        _logger.info(f"zero2_reduce_scatter_fusion: {allreduce_info}")

    def set_optimizer_allgather_fusion_comm_list(self, comm_fusion):
        allgather_info = [{"size": 0, "fusion_id": self.max_fusion_id + 1, "params": []}]
        self.max_fusion_id += 1
        self.optimizer_allgather_list = []
        for i, param in enumerate(self.ori_parameters):
            param_size = param.itemsize * param.size
            param_name = param.name
            if self.need_parameter_split[i]:
                self.update_comm_op_info(
                    allgather_info, comm_fusion["allgather"]["bucket_size"], param_size, param_name
                )
            comm_op = ops.AllGather(group=self.op_group)
            comm_op.add_prim_attr("fusion", allgather_info[-1]["fusion_id"])
            self.optimizer_allgather_list.append(comm_op)
        _logger.info(f"optimizer_allgather_fusion: {allgather_info}")

    def set_dp_allreduce_comm_list(self, comm_fusion):
        dp_allreduce_info = [{"size": 0, "fusion_id": self.max_fusion_id + 1, "params": []}]
        self.max_fusion_id += 1
        self.dp_allreduce_list = []
        for i, param in enumerate(self.ori_parameters):
            param_size = param.itemsize * param.size
            param_name = param.name
            if self.need_parameter_split[i]:
                self.update_comm_op_info(
                    dp_allreduce_info, comm_fusion["allreduce"]["bucket_size"], param_size, param_name
                )
            comm_op = ops.AllGather(group=self.op_group)
            comm_op.add_prim_attr("fusion", dp_allreduce_info[-1]["fusion_id"])
            self.dp_allreduce_list.append(comm_op)
        _logger.info(f"dp_allreduce_fusion: {dp_allreduce_info}")

    def split_param(self, param):
        return split_np(param, self.op_group_size, self.op_rank_id)

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

    def dump_params_split_info(self, params_split_info):
        params_split_info_path = os.path.join(params_split_info, f"params_split_info_{self.op_rank_id}.json")
        params_split_info_dict = {}
        for i, param in enumerate(self.optimizer._parameters):
            param_split_info = {
                "split": self.need_parameter_split[i],
                "group_size": self.op_group_size,
                "rank_id": self.op_rank_id,
            }
            params_split_info_dict[param.name] = param_split_info
        params_split_info_json = json.dumps(params_split_info_dict, indent=2)
        with open(params_split_info_path, "w") as f:
            f.write(params_split_info_json)

    def get_need_parameter_split(self):
        self.need_parameter_split = [False] * len(self.optimizer._parameters)
        param_tuples = self.get_optimizer_param_tuples()
        for i, param in enumerate(self.optimizer._parameters):
            if self.zero_stage == 3:
                if param_tuples:
                    B = param_tuples[0][i].shape[0]
                else:
                    continue
            else:
                B = param.shape[0]
            if param.parallel_optimizer and B >= self.op_group_size and B % self.op_group_size == 0:
                if self.zero_stage in [1, 2]:
                    self.need_parameter_split[i] = True
        self.need_parameter_split = tuple(self.need_parameter_split)

    def split_params(self):
        if self.zero_stage in [1, 2] and self.is_parallel:
            _logger.info("Clone optimizer.parameters, will increase memory.")
            # Because the first input of MindSpore optimizer must be ms.Parameter,
            # copy optimizer.parameters for optimizer parameters update.
            # It will increase 1/n parameters' memory.
            self.optimizer.parameters = self.optimizer.parameters.clone(prefix="wrapper", init="same")
            self.optimizer._parameters = self.optimizer.parameters
            self.last_assign = True

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
                    ori_shape = param.shape
                    param.assign_value(self.split_param(param))
                    _logger.debug(f"Optimizer {param.name} from {ori_shape} to {param.shape}")
                for param_tuple in param_tuples:
                    ori_shape = param_tuple[i].shape
                    param_tuple[i].assign_value(self.split_param(param_tuple[i]))
                    _logger.debug(f"Optimizer {param_tuple[i].name} " f"from {ori_shape} to {param_tuple[i].shape}")

    def reduce_scatter_gradients(self, gradients):
        dtype = gradients[0].dtype
        if self.use_comm_fusion:
            gradients = self.hyper_map(
                ops.partial(
                    _stage2_reduce_scatter,
                    ms.Tensor(self.op_group_size, dtype),
                ),
                self.zero2_reduce_scatter_list,
                self.zero2_allreduce_list,
                gradients,
                self.need_parameter_split,
            )
        else:
            gradients = self.hyper_map(
                ops.partial(
                    _stage2_reduce_scatter,
                    ms.Tensor(self.op_group_size, dtype),
                    self.op_reduce_scatter,
                    self.op_allreduce,
                ),
                gradients,
                self.need_parameter_split,
            )
        return gradients

    def dp_allreduce_gradients(self, gradients):
        dtype = gradients[0].dtype
        if self.use_comm_fusion:
            gradients = self.hyper_map(
                ops.partial(
                    _dp_allreduce,
                    ms.Tensor(self.dp_group_size, dtype),
                ),
                self.dp_allreduce_list,
                gradients,
            )
        else:
            gradients = self.hyper_map(
                ops.partial(
                    _dp_allreduce,
                    ms.Tensor(self.dp_group_size, dtype),
                    self.dp_allreduce,
                ),
                gradients,
            )
        return gradients

    def split_gradients(self, gradients):
        if self.use_comm_fusion:
            gradients = self.hyper_map(
                ops.partial(
                    _stage1_split_grad,
                    self.split_op,
                    self.op_group_size,
                    self.op_rank_id,
                ),
                self.zero1_allreduce_list,
                gradients,
                self.need_parameter_split,
            )
        else:
            gradients = self.hyper_map(
                ops.partial(
                    _stage1_split_grad,
                    self.split_op,
                    self.op_group_size,
                    self.op_rank_id,
                    self.op_allreduce,
                ),
                gradients,
                self.need_parameter_split,
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
            if self.use_comm_fusion:
                optim_result = ops.depend(
                    self.hyper_map(
                        ops.partial(_optim_allgather, self.last_assign),
                        self.optimizer_allgather_list,
                        self.ori_parameters,
                        self.optimizer._parameters,
                        self.need_parameter_split,
                    ),
                    optim_result,
                )
            else:
                optim_result = ops.depend(
                    self.hyper_map(
                        ops.partial(_optim_allgather, self.last_assign, self.op_allgather),
                        self.ori_parameters,
                        self.optimizer._parameters,
                        self.need_parameter_split,
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


def _init_parallel_settings(net, op_group, parallel_modules=None):
    for module, parallel_module in parallel_modules.items():
        if isinstance(net, module):
            cell_type = get_cell_dtype(net)
            new_net = parallel_module(net, 3, op_group)
            if cell_type is not None:
                new_net.to_float(cell_type)
            return new_net
    return None


def get_cell_params_fullname_dict(cell: nn.Cell):
    fullname_dict = {}
    for param_name in cell._params:
        fullname_dict[param_name] = getattr(cell, param_name).name
    return fullname_dict


def _prepare_network(network: nn.Cell, op_group: str, parallel_modules=None):
    new_net = _init_parallel_settings(network, op_group, parallel_modules)
    if new_net is not None:
        return new_net
    for name, sub_net in network._cells.items():
        if not sub_net:
            continue
        new_sub_net = _init_parallel_settings(sub_net, op_group, parallel_modules)
        if new_sub_net is not None:
            params_fullname_dict = get_cell_params_fullname_dict(sub_net)
            if isinstance(network, (nn.CellList, nn.SequentialCell)):
                network._cells[name] = new_sub_net
                if isinstance(network, nn.SequentialCell):
                    network.cell_list = list(network._cells.values())
            else:
                network.__setattr__(name, new_sub_net)

            # parameter name will update after __setattr__, reset to ori parameter name.
            for param_name in new_sub_net.net._params:
                getattr(new_sub_net.net, param_name).name = params_fullname_dict[param_name]
            continue
        if sub_net._params:
            for param_name in sub_net._params:
                param = getattr(sub_net, param_name)
                _logger.warning(f"Set param {param.name} parallel_optimizer False, param shape {param.shape}")
                param.parallel_optimizer = False
        _prepare_network(sub_net, op_group, parallel_modules)
    return network


def prepare_network(network: nn.Cell, zero_stage: int = 0, op_group: str = None, parallel_modules=None):
    if zero_stage != 3 or _get_parallel_mode() != ParallelMode.DATA_PARALLEL:
        _logger.info("No need rewrite network and return original network.")
        return network
    _logger.info("Rewrite the network, please wait...")
    if parallel_modules is None:
        parallel_modules = PARALLEL_MODULES
    network = _prepare_network(network, op_group, parallel_modules)
    return network


def prepare_ema(ema, zero_stage: int = 0, op_group: str = None):
    is_parallel = _get_parallel_mode() == ParallelMode.DATA_PARALLEL
    if not is_parallel or zero_stage != 3:
        return ema
    op_group_size = get_group_size(op_group)
    op_rank_id = get_rank(op_group)
    _logger.info(f"Split EMA params: rank_id {op_rank_id}, rank_size {op_group_size}.")
    for net_weight, ema_weight, swap_cache in zip(ema.net_weight, ema.ema_weight, ema.swap_cache):
        if net_weight.shape == ema_weight.shape:
            continue
        ema_weight.set_data(split_np(ema_weight, op_group_size, op_rank_id), slice_shape=True)
        swap_cache.set_data(split_np(swap_cache, op_group_size, op_rank_id), slice_shape=True)
    return ema


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
    zero_stage: Literal[0, 1, 2, 3] = 0,
    optimizer_offload: bool = False,
    op_group: str = None,
    dp_group: str = None,
    comm_fusion: dict = None,
    parallel_modules=None,
) -> TrainOneStepWrapper:
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
        parallel_modules (`dict`, *optional*): A dict of Cells could split parameters in zero3, default is None.
            If None, use `PARALLEL_MODULES` from `mindone.models.modules.parallel`.
    """
    if zero_stage not in [0, 1, 2, 3]:
        raise ValueError("Not support zero_stage {zero_stage}")
    if op_group is None:
        _logger.warning("Not set zero group, set it WORLD_COMM_GROUP.")
        op_group = GlobalComm.WORLD_COMM_GROUP
    if op_group != GlobalComm.WORLD_COMM_GROUP and dp_group is None:
        raise ValueError("op_group {op_group} and dp_group {dp_group} not full network hccl group coverage")

    is_parallel = _get_parallel_mode() == ParallelMode.DATA_PARALLEL
    if not is_parallel and zero_stage == 0:
        _logger.info("No need prepare train_network with zero.")
        zero_helper = None
    else:
        network = prepare_network(network, zero_stage, op_group, parallel_modules=parallel_modules)
        zero_helper = ZeroHelper(optimizer, zero_stage, op_group, dp_group, optimizer_offload, comm_fusion)

    if ema is not None:
        ema = prepare_ema(ema, zero_stage, op_group)
    if isinstance(scale_sense, float):
        scale_sense = ms.Tensor(scale_sense, ms.float32)
    train_network = TrainOneStepWrapper(
        network,
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


def transform_checkpoints(src_checkpoint: str, src_param_split_info_json: str, group_size: int):
    """
    src_checkpoint (`str`): The path of checkpoints need to merge parameters. eg. "save_checkpoint_dir/ckpt_{}.ckpt",
        {} is placeholder of rank_id.
    src_param_split_info_json (`str`): The path of param_split_info_jsons. eg. "params_info/params_split_info_{}.json",
        {} is placeholder of rank_id.
    group_size (`int`): The rank size of the communication group.
    """

    def read_json(json_file):
        s = ""
        with open(json_file, "r") as f:
            for line in f.readlines():
                s += line
        return json.loads(s)

    new_params_list = []
    ckpts = []
    jsons = []
    for i in range(group_size):
        ckpts.append(ms.load_checkpoint(src_checkpoint.format(i)))
        jsons.append(read_json(src_param_split_info_json.format(i)))
    for param_name in ckpts[0].keys():
        param_value = None
        param_list = []
        for i in range(group_size):
            if param_name not in jsons[i]:
                _logger.warning(f"param {param_name} not in param_split_info_json, keep ori data.")
                if i:
                    raise ValueError("please check jsons, param name not same!")
                param_value = ckpts[0][param_name]
                break
            elif not jsons[i][param_name]["split"]:
                if i:
                    raise ValueError("please check jsons, param info not same!")
                param_value = ckpts[0][param_name]
                break
            else:
                param_list.append(ckpts[i][param_name])
        if param_value is None:
            param_value = ops.cat(param_list)
            _logger.debug("Merge {param_name} to {param_value.shape}")

        new_params_list.append({"name": param_name, "data": param_value})

    ms.save_checkpoint(new_params_list, src_checkpoint.format(f"all_{group_size}"))
