from typing import Optional

from mindspore.communication import create_group, get_group_size, get_rank

_GLOBAL_PARALLEL_GROUPS = dict()


def set_tensor_parallel_group(group: str) -> None:
    _GLOBAL_PARALLEL_GROUPS["tensor"] = group


def get_tensor_parallel_group() -> Optional[str]:
    return _GLOBAL_PARALLEL_GROUPS.get("tensor", None)


def set_sequence_parallel_group(group: str) -> None:
    _GLOBAL_PARALLEL_GROUPS["sequence"] = group


def get_sequence_parallel_group() -> Optional[str]:
    return _GLOBAL_PARALLEL_GROUPS.get("sequence", None)


def create_parallel_group(tensor_parallel_shards: int = 1) -> None:
    device_num = get_group_size()
    if device_num % tensor_parallel_shards != 0:
        raise ValueError(
            f"Total number of devices ({device_num}) must be devisible by the number of tensors parallel shards ({tensor_parallel_shards})."
        )

    rank_id = get_rank()
    tp_group_id = rank_id // tensor_parallel_shards
    tp_group_rank_ids = list(range(tp_group_id * tensor_parallel_shards, (tp_group_id + 1) * tensor_parallel_shards))
    tp_group_name = f"tp_group_{tp_group_id}"
    create_group(tp_group_name, tp_group_rank_ids)
    set_tensor_parallel_group(tp_group_name)
