from typing import Optional

from mindspore.communication import GlobalComm, create_group, get_group_size, get_rank

_GLOBAL_PARALLEL_GROUPS = dict()


def set_sequence_parallel_group(group: str) -> None:
    _GLOBAL_PARALLEL_GROUPS["sequence"] = group


def get_sequence_parallel_group() -> Optional[str]:
    # TODO: change the default value to be None
    return _GLOBAL_PARALLEL_GROUPS.get("sequence", GlobalComm.WORLD_COMM_GROUP)


def set_model_parallel_group(group: str) -> None:
    _GLOBAL_PARALLEL_GROUPS["model"] = group


def get_model_parallel_group() -> Optional[str]:
    # TODO: change the default value to be None
    return _GLOBAL_PARALLEL_GROUPS.get("model", GlobalComm.WORLD_COMM_GROUP)


def create_parallel_group(sequence_parallel_shards: int = 1, model_parallel_shards: int = 1) -> None:
    if sequence_parallel_shards <= 1 and model_parallel_shards <= 1:
        raise ValueError(
            f"`sequence_parallel_shards`/`model_parallel_shards` must be larger than 1 "
            f"to enable sequence/model parallel, but get `{sequence_parallel_shards}` and `{model_parallel_shards}`."
        )

    device_num = get_group_size()
    if device_num % sequence_parallel_shards != 0 or device_num % model_parallel_shards != 0:
        raise ValueError(
            f"Total number of devices ({device_num}) must be divisible by the number of "
            f"sequence parallel shards ({sequence_parallel_shards}) and model parallel shards ({model_parallel_shards})."
        )

    rank_id = get_rank()

    if sequence_parallel_shards > 1:
        sp_group_id = rank_id // sequence_parallel_shards
        sp_group_rank_ids = list(
            range(sp_group_id * sequence_parallel_shards, (sp_group_id + 1) * sequence_parallel_shards)
        )
        sp_group_name = f"sp_group_{sp_group_id}"
        create_group(sp_group_name, sp_group_rank_ids)
        set_sequence_parallel_group(sp_group_name)
    elif model_parallel_shards > 1:  # not compatible with SP currently
        mp_group_id = rank_id // model_parallel_shards
        mp_group_rank_ids = list(range(mp_group_id * model_parallel_shards, (mp_group_id + 1) * model_parallel_shards))
        mp_group_name = f"mp_group_{mp_group_id}"
        create_group(mp_group_name, mp_group_rank_ids)
        set_model_parallel_group(mp_group_name)
