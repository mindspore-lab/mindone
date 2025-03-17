import mindspore as ms
from mindspore import nn, ops
from mindspore.communication import get_group_size
from mindspore.communication.management import GlobalComm
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode


class ZeroParamWrapper(nn.Cell):
    """
    a cell to Insert communication operators before and after parameters when `zero_stage == 3`.
    """

    def __init__(
        self,
        param: ms.Parameter,
        zero_stage: int = 0,
        optimizer_parallel_group: str = GlobalComm.WORLD_COMM_GROUP,
        cell_type=None,
    ):
        super().__init__(auto_prefix=False)
        self.optimizer_parallel_group = optimizer_parallel_group
        self.zero_stage = zero_stage
        self.cell_type = cell_type
        if zero_stage != 3:
            raise ValueError(f"ZeroParamWrapper not support zero_stage {zero_stage}.")

        # Init parallel settings
        self.is_parallel = _get_parallel_mode() == ParallelMode.DATA_PARALLEL
        self.op_group_size = get_group_size(self.optimizer_parallel_group) if self.is_parallel else 1
        self.allgather = ops.Identity()
        self.reduce_scatter = None
        self.dtype = param.dtype
        self.allreduce = ops.AllReduce(group=self.optimizer_parallel_group, op=ops.ReduceOp.SUM)

        self.need_rewrite = self.check_rewrite(param)
        if self.need_rewrite:
            self.op_allgather = ops.AllGather(group=self.optimizer_parallel_group)
            self.op_reduce_scatter = ops.ReduceScatter(group=self.optimizer_parallel_group, op=ops.ReduceOp.SUM)

    def check_rewrite(self, param):
        """Check the parameter need to split or not."""
        need_rewrite = self.is_parallel
        B = param.shape[0]
        if not param.parallel_optimizer or B < self.op_group_size or B % self.op_group_size != 0:
            need_rewrite = False
        param.parallel_optimizer = need_rewrite
        return need_rewrite

    def construct(self, param):
        if self.need_rewrite:
            if self.cell_type is not None:
                param = param.to(self.cell_type)
            return self.op_allgather(param)
        return param

    def bprop(self, param, out, dout):
        if self.need_rewrite:
            r = self.op_reduce_scatter(dout.to(self.dtype)) / self.op_group_size
            return (r,)
        dout = self.allreduce(dout.to(self.dtype)) / self.op_group_size
        return (dout,)
