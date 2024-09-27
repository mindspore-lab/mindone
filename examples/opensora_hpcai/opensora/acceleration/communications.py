from typing import Callable, Tuple, Union

import mindspore.mint as mint
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.communication import GlobalComm, get_group_size, get_rank

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


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


class SplitFowardGatherBackward(nn.Cell):
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


class GatherFowardSplitBackward(nn.Cell):
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
        world_size = get_group_size(group)
        self.alltoall = ops.AlltoAll(
            split_count=world_size, split_dim=self.split_dim, concat_dim=self.concat_dim, group=group
        )

    def construct(self, x: Tensor, split_pad: Union[int, Tensor] = 0, concat_pad: Union[int, Tensor] = 0) -> Tensor:
        if split_pad > 0:
            padding = (len(x.shape) - self.split_dim - 1) * (0, 0) + (0, split_pad)
            x = mint.nn.functional.pad(x, padding)

        x = self.alltoall(x)

        if concat_pad > 0:
            x = x.narrow(self.concat_dim, 0, x.shape[self.concat_dim] - concat_pad)
        return x
