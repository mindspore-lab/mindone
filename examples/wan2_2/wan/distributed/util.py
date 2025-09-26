# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from typing import Any, List, Optional

import mindspore as ms
import mindspore.mint as mint
import mindspore.mint.distributed as dist


def init_distributed_group() -> None:
    """r initialize sequence parallel group."""
    if not dist.is_initialized():
        dist.init_process_group(backend="hccl")


def get_rank() -> int:
    return dist.get_rank()


def get_world_size() -> int:
    return dist.get_world_size()


def all_to_all(x: ms.Tensor, scatter_dim: int, gather_dim: int, group: Optional[Any] = None, **kwargs) -> ms.Tensor:
    """
    `scatter` along one dimension and `gather` along another.
    """
    world_size = get_world_size()
    if world_size > 1:
        inputs = [u.contiguous() for u in x.chunk(world_size, dim=scatter_dim)]
        outputs = [mint.empty_like(u) for u in inputs]
        dist.all_to_all(outputs, inputs, group=group, **kwargs)
        x = mint.cat(outputs, dim=gather_dim).contiguous()
    return x


def all_gather(tensor: ms.Tensor) -> List[ms.Tensor]:
    world_size = dist.get_world_size()
    if world_size == 1:
        return [tensor]
    tensor_list = [mint.empty_like(tensor) for _ in range(world_size)]
    mint.distributed.all_gather(tensor_list, tensor)
    return tensor_list


def gather_forward(input: ms.Tensor, dim: int) -> ms.Tensor:
    # skip if world_size == 1
    world_size = dist.get_world_size()
    if world_size == 1:
        return input

    # gather sequence
    output = all_gather(input)
    return mint.cat(output, dim=dim).contiguous()
