from typing import Optional

from mindspore.communication import create_group, get_group_size, get_rank

__all__ = ["set_model_parallel_group", "get_model_parallel_group", "create_parallel_group"]


_GLOBAL_PARALLEL_GROUPS = dict()


def set_model_parallel_group(group: str) -> None:
    _GLOBAL_PARALLEL_GROUPS["model"] = group


def get_model_parallel_group() -> Optional[str]:
    return _GLOBAL_PARALLEL_GROUPS.get("model", None)


def create_parallel_group(model_parallel_shards: int = 1) -> None:
    if model_parallel_shards <= 1:
        raise ValueError(
            f"`model_parallel_shards` must be larger than 1 to enable model parallel, but get `{model_parallel_shards}`."
        )

    device_num = get_group_size()
    if device_num % model_parallel_shards != 0:
        raise ValueError(
            f"Total number of devices ({device_num}) must be divisible by the number of model parallel shards ({model_parallel_shards})."
        )

    rank_id = get_rank()

    if model_parallel_shards > 1:
        mp_group_id = rank_id // model_parallel_shards
        mp_group_rank_ids = list(range(mp_group_id * model_parallel_shards, (mp_group_id + 1) * model_parallel_shards))
        mp_group_name = f"mp_group_{mp_group_id}"
        create_group(mp_group_name, mp_group_rank_ids)
        set_model_parallel_group(mp_group_name)
