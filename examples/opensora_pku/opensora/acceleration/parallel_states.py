from mindspore.communication import create_group, get_group_size, get_rank

_SEQUENCE_PARALLEL_STATE = None


class HCCL_INFO:
    def __init__(self):
        self.group = None
        self.group_id = None
        self.world_size = 0
        self.rank = -1


hccl_info = HCCL_INFO()


def initialize_sequence_parallel_state(sequence_parallel_size):
    global _SEQUENCE_PARALLEL_STATE
    if sequence_parallel_size > 1:
        _SEQUENCE_PARALLEL_STATE = True
        _initialize_sequence_parallel_group(sequence_parallel_size)
    else:
        _SEQUENCE_PARALLEL_STATE = False


def set_sequence_parallel_state(state):
    global _SEQUENCE_PARALLEL_STATE
    _SEQUENCE_PARALLEL_STATE = state


def get_sequence_parallel_state():
    return _SEQUENCE_PARALLEL_STATE


def _initialize_sequence_parallel_group(sequence_parallel_size):
    """Initialize the sequence parallel group."""
    rank = get_rank()
    world_size = get_group_size()

    assert world_size % sequence_parallel_size == 0, "world_size must be divisible by sequence_parallel_size"

    hccl_info.world_size = sequence_parallel_size
    hccl_info.rank = rank

    g_id = rank // sequence_parallel_size
    ranks = list(range(g_id * sequence_parallel_size, (g_id + 1) * sequence_parallel_size))
    group = f"sp_group_{g_id}"
    create_group(group, ranks)
    hccl_info.group = group
    hccl_info.group_id = g_id
