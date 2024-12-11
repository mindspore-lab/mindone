from typing import Optional

from mindspore.communication import create_group, get_group_size, get_rank

_GLOBAL_PARALLEL_GROUPS = dict()


def set_sequence_parallel_group(group: str) -> None:
    _GLOBAL_PARALLEL_GROUPS["sequence"] = group


def get_sequence_parallel_group() -> Optional[str]:
    return _GLOBAL_PARALLEL_GROUPS.get("sequence", None)


def create_parallel_group(sequence_parallel_shards: int) -> None:
    if sequence_parallel_shards <= 1:
        raise ValueError(
            f"`sequence_parallel_shards` must be larger than 1 to enable sequence parallel, but get `{sequence_parallel_shards}`."
        )

    device_num = get_group_size()
    if device_num % sequence_parallel_shards != 0:
        raise ValueError(
            f"Total number of devices `{device_num}` must be divisible by the number of sequence parallel shards `{sequence_parallel_shards}`."
        )

    rank_id = get_rank()
    sp_group_id = rank_id // sequence_parallel_shards
    sp_group_rank_ids = list(
        range(sp_group_id * sequence_parallel_shards, (sp_group_id + 1) * sequence_parallel_shards)
    )
    sp_group_name = f"sp_group_{sp_group_id}"
    create_group(sp_group_name, sp_group_rank_ids)
    set_sequence_parallel_group(sp_group_name)
