from typing import Optional

from mindspore.communication import create_group, get_group_size, get_rank

_GLOBAL_PARALLEL_GROUPS = dict()


def set_model_parallel_group(group: str) -> None:
    _GLOBAL_PARALLEL_GROUPS["model"] = group


def get_model_parallel_group() -> Optional[str]:
    return _GLOBAL_PARALLEL_GROUPS.get("model", None)


def set_sequence_parallel_group(group: str) -> None:
    _GLOBAL_PARALLEL_GROUPS["sequence"] = group


def get_sequence_parallel_group() -> Optional[str]:
    return _GLOBAL_PARALLEL_GROUPS.get("sequence", None)


def create_parallel_group(model_parallel_shards: int = 1) -> None:
    device_num = get_group_size()
    if device_num % model_parallel_shards != 0:
        raise ValueError(
            f"Total number of devices ({device_num}) must be devisible by the number of model parallel shards ({model_parallel_shards})."
        )

    rank_id = get_rank()
    mp_group_id = rank_id // model_parallel_shards
    mp_group_rank_ids = list(range(mp_group_id * model_parallel_shards, (mp_group_id + 1) * model_parallel_shards))
    mp_group_name = f"mp_group_{mp_group_id}"
    create_group(mp_group_name, mp_group_rank_ids)
    set_model_parallel_group(mp_group_name)
