from ..mindspore_adapter.utils import _is_ascend


def is_flash_attn_2_available():
    if _is_ascend():
        return True

    return False


def is_sdpa_available():
    return False
