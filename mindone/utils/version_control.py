import logging
import os
from functools import total_ordering

from packaging import version

import mindspore as ms

logger = logging.getLogger()


@total_ordering
class VersionComparator:
    """
    Package version comparison class.
    """

    def __init__(self, current: str):
        self.current = version.parse(current)

    def __eq__(self, other: str) -> bool:
        return self.current == version.parse(other)

    def __gt__(self, other: str) -> bool:
        return self.current > version.parse(other)

    def __str__(self):
        return str(self.current)


MS_VERSION = VersionComparator(ms.__version__)


def get_ascend_soc_version():
    """Get ascend soc version."""
    if MS_VERSION >= "2.2.0":
        from mindspore._c_expression import MSContext

        return MSContext.get_instance().get_ascend_soc_version()
    ascend_chip_type = os.getenv("ASCEND_CHIP_TYPE", "UNSET")
    if ascend_chip_type not in ["910a", "910b", "UNSET"]:
        raise EnvironmentError(f"ASCEND_CHIP_TYPE should be in ['910a', '910b'],but get {ascend_chip_type}")
    if ascend_chip_type == "UNSET":
        logger.info(
            "For MS version<2.2.0, environment variables need to be set manually to obtain the chip type,"
            "which can be set as follows: \n"
            "For 910A chip, run 'export ASCEND_CHIP_TYPE=910a' before the program runs.\n"
            "For 910B chip, run 'export ASCEND_CHIP_TYPE=910b' before the program runs.\n"
            "If you need to get chip information automatically, MindSpore 2.2 and above is recommended"
        )
    return ascend_chip_type


def is_910a():
    device = get_ascend_soc_version()
    return device in ["910a", "ascend910"]


def is_910b():
    device = get_ascend_soc_version()
    return device in ["910b", "ascend910b"]


def check_valid_flash_attention(import_fa_valid=True):
    """check mindspore version is valid for flash attention"""
    if MS_VERSION < "2.2.0":
        logger.warning("Current MindSpore do not support FlashAttention, please upgrade to 2.2.0 or later version.")
        logger.warning("Now running on self-attention mode.")
        result = False
    # ms 2.2.0 or latter version but import error is not support
    elif not import_fa_valid:
        logger.warning("Import FlashAttention ERROR, please upgrade your MindSpore to 2.2.0 or later version. ")
        logger.warning("Now running on self-attention mode.")
        result = False
    # both pass should return True
    else:
        result = True
    return result


def choose_flash_attention_dtype():
    """
    attention_mask dtype should be float16 on ms 2.2.0, uint8 on 2.2.10
    ms version below 2.2.0 won't be in this func
    """
    if MS_VERSION >= "2.2.1":
        return ms.uint8
    return ms.float16


def is_old_ms_version(last_old_version="1.10.1"):
    # some APIs are changed after ms 1.10.1 version, such as dropout
    return MS_VERSION <= last_old_version
