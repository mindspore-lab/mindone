# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
from typing import Any, Optional

import mindspore as ms
import mindspore.mint.distributed as dist
import mindspore.nn as nn
from mindspore.communication import GlobalComm

from mindone.trainers.zero import prepare_network


def shard_model(
    model: nn.Cell,
    device_id: Optional[int] = None,
    param_dtype: ms.dtype = ms.bfloat16,
    reduce_dtype: ms.dtype = ms.float32,
    buffer_dtype: ms.dtype = ms.float32,
    process_group: Optional[Any] = None,
    sharding_strategy: Optional[Any] = None,
    sync_module_states: bool = True,
) -> nn.Cell:
    model = prepare_network(model, zero_stage=3, optimizer_parallel_group=GlobalComm.WORLD_COMM_GROUP)
    return model


def free_model(obj: object, name: str, dist_barrier: bool = True) -> None:
    ms.synchronize()
    if dist.is_initialized() and dist_barrier:
        dist.barrier()
    delattr(obj, name)
    gc.collect()
    ms.empty_cache()
