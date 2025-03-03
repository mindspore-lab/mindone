from typing import Callable, Literal, Tuple

import mindspore.mint as mint
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.communication import GlobalComm, get_group_size, get_rank


def _split(x: Tensor, dim: int, rank: int, world_size: int) -> Tensor:
    dim_size = x.shape[dim]
    tensor_list = x.split(dim_size // world_size, axis=dim)
    x = tensor_list[rank]
    return x


def _communicate_along_dim(x: Tensor, dim: int, func: Callable[[Tensor], Tensor]) -> Tensor:
    x = x.swapaxes(0, dim)
    x = func(x)
    x = x.swapaxes(dim, 0)
    return x


class SplitForwardGatherBackward(nn.Cell):
    def __init__(
        self, dim: int = 0, grad_scale: Literal["up", "down"] = "down", group: str = GlobalComm.WORLD_COMM_GROUP
    ) -> None:
        super().__init__()
        self.dim = dim
        self.rank = get_rank(group)
        self.world_size = get_group_size(group)
        self.gather = ops.AllGather(group=group)

        if grad_scale == "up":
            self.scale = self.world_size
        else:
            self.scale = 1 / self.world_size

    def construct(self, x: Tensor) -> Tensor:
        return _split(x, self.dim, self.rank, self.world_size)

    def bprop(self, x: Tensor, out: Tensor, dout: Tensor) -> Tuple[Tensor]:
        dout = dout * self.scale
        dout = _communicate_along_dim(dout, self.dim, self.gather)
        return (dout,)


class GatherForwardSplitBackward(nn.Cell):
    def __init__(
        self, dim: int = 0, grad_scale: Literal["up", "down"] = "up", group: str = GlobalComm.WORLD_COMM_GROUP
    ) -> None:
        super().__init__()
        self.dim = dim
        self.rank = get_rank(group)
        self.world_size = get_group_size(group)
        self.gather = ops.AllGather(group=group)

        if grad_scale == "up":
            self.scale = self.world_size
        else:
            self.scale = 1 / self.world_size

    def construct(self, x: Tensor) -> Tensor:
        x = _communicate_along_dim(x, self.dim, self.gather)
        return x

    def bprop(self, x: Tensor, out: Tensor, dout: Tensor) -> Tuple[Tensor]:
        dout = dout * self.scale
        dout = _split(dout, self.dim, self.rank, self.world_size)
        return (dout,)


class AlltoAll(nn.Cell):
    def __init__(self, split_dim: int = 2, concat_dim: int = 1, group: str = GlobalComm.WORLD_COMM_GROUP) -> None:
        super().__init__()
        assert split_dim >= 0 and concat_dim >= 0
        self.split_dim = split_dim
        self.concat_dim = concat_dim
        self.group = group

    @staticmethod
    def _all_to_all(x: Tensor, split_dim: int, concat_dim: int, group: str = GlobalComm.WORLD_COMM_GROUP):
        world_size = get_group_size(group)
        input_list = list(mint.chunk(x, world_size, dim=split_dim))
        output_list = [mint.empty_like(input_list[0]) for _ in range(world_size)]
        mint.distributed.all_to_all(output_list, input_list, group=group)
        return mint.cat(output_list, dim=concat_dim)

    def construct(self, x: Tensor) -> Tensor:
        x = self._all_to_all(x, self.split_dim, self.concat_dim, group=self.group)
        return x

    def bprop(self, x: Tensor, out: Tensor, dout: Tensor) -> Tuple[Tensor]:
        dout = self._all_to_all(dout, self.concat_dim, self.split_dim, group=self.group)
        return (dout,)
