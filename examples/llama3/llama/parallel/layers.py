import numbers
from typing import Callable, Optional, Tuple, Union

import mindspore as ms
import mindspore.mint as mint
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.common.initializer import Initializer
from mindspore.communication import GlobalComm, get_group_size, get_rank

__all__ = ["ColumnParallelLinear"]


def _communicate_along_dim(x: Tensor, dim: int, func: Callable[[Tensor], Tensor]) -> Tensor:
    x = x.swapaxes(0, dim)
    x = func(x)
    x = x.swapaxes(dim, 0)
    return x


def _split(x: Tensor, dim: int, rank: int, world_size: int) -> Tensor:
    dim_size = x.shape[dim]
    tensor_list = x.split(dim_size // world_size, axis=dim)
    x = tensor_list[rank]
    return x


class _CopyToModelParallelRegion(nn.Cell):
    def __init__(self, group: str = GlobalComm.WORLD_COMM_GROUP) -> None:
        super().__init__(auto_prefix=False)
        self.reduce = ops.AllReduce(op=ops.ReduceOp.SUM, group=group)

    def construct(self, x: Tensor) -> Tensor:
        return x

    def bprop(self, x: Tensor, out: Tensor, dout: Tensor) -> Tuple[Tensor]:
        dout = self.reduce(dout)
        return (dout,)


class _GatherFromModelParallelRegion(nn.Cell):
    def __init__(self, group: str = GlobalComm.WORLD_COMM_GROUP) -> None:
        super().__init__(auto_prefix=False)
        self.gather = ops.AllGather(group=group)
        self.rank = get_rank(group)
        self.world_size = get_group_size(group)

    def construct(self, x: Tensor) -> Tensor:
        return _communicate_along_dim(x, -1, self.gather)

    def bprop(self, x: Tensor, out: Tensor, dout: Tensor) -> Tuple[Tensor]:
        dout = _split(dout, -1, self.rank, self.world_size)
        return (dout,)


class ColumnParallelLinear(nn.Cell):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_init: Union[None, Tensor, str, Initializer, numbers.Number] = None,
        bias_init: Union[None, Tensor, str, Initializer, numbers.Number] = None,
        gather_output: bool = True,
        group: str = GlobalComm.WORLD_COMM_GROUP,
        dtype: Optional[ms.Type] = None,
    ):
        super().__init__(auto_prefix=False)

        self.group_size = get_group_size(group)
        assert out_features % self.group_size == 0
        self.out_features_per_partition = out_features // self.group_size
        self.gather_output = gather_output

        self.copy_to_model_parallel_region = _CopyToModelParallelRegion(group)
        self.linear = mint.nn.Linear(
            in_features,
            self.out_features_per_partition,
            bias=bias,
            weight_init=weight_init,
            bias_init=bias_init,
            dtype=dtype,
        )
        if self.gather_output:
            self.gather_from_model_parallel_region = _GatherFromModelParallelRegion(group)

    def construct(self, x: Tensor) -> Tensor:
        x = self.copy_to_model_parallel_region(x)
        x = self.linear(x)
        if self.gather_output:
            x = self.gather_from_model_parallel_region(x)
        return x
