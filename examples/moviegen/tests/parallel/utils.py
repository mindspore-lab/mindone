from typing import Callable, Tuple

import numpy as np

import mindspore.ops as ops
from mindspore import Tensor
from mindspore.communication import GlobalComm, get_group_size


def _communicate_along_dim(x: Tensor, dim: int, func: Callable[[Tensor], Tensor]) -> Tensor:
    x = x.swapaxes(0, dim)
    x = func(x)
    x = x.swapaxes(dim, 0)
    return x


def gather_or_reduce_parallel_gradient(
    parallel_gradient: Tensor, non_parallel_gradient_shape: Tuple[int, ...], group: str = GlobalComm.WORLD_COMM_GROUP
) -> Tensor:
    if parallel_gradient.shape == non_parallel_gradient_shape:
        # Sequence Parallel / Context Parallel
        allreduce = ops.AllReduce(op=ops.ReduceOp.SUM, group=group)
        parallel_gradient = allreduce(parallel_gradient) / get_group_size(group)
        return parallel_gradient

    scales = np.array(non_parallel_gradient_shape) / np.array(parallel_gradient.shape)
    assert np.count_nonzero(scales - 1) == 1
    assert np.prod(scales) == get_group_size(group)
    dim = np.argmax(scales).item()
    allgather = ops.AllGather(group=group)
    parallel_gradient = _communicate_along_dim(parallel_gradient, dim, allgather)
    return parallel_gradient
