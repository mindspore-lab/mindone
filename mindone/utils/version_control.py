import logging
import os

from packaging import version

import mindspore as ms

logger = logging.getLogger()


def is_old_ms_version(last_old_version="1.10.1"):
    # some APIs are changed after ms 1.10.1 version, such as dropout
    return version.parse(ms.__version__) <= version.parse(last_old_version)


# For the following code, credits are to mindformers
def is_version_ge(current_version, base_version):
    """
    return current_version >= base_version.
    Check whether the current version is higher than or equal to the base version.
    for current_version: 1.8.1, base_version: 1.11.0, it return False.
    """
    version_split_char = "."
    if version_split_char not in base_version or version_split_char not in current_version:
        raise ValueError(
            "The version string will contain the `.`." "For example, current_version 1.8.1， base_version: 1.11.0."
        )
    for x, y in zip(current_version.split(version_split_char), base_version.split(version_split_char)):
        if not x.isdigit() or not y.isdigit():
            continue
        if int(x) != int(y):
            return int(x) >= int(y)
    return True


def get_ascend_soc_version():
    """Get ascend soc version."""
    if is_version_ge(ms.__version__, "2.2.0"):
        from mindspore._c_expression import MSContext

        return MSContext.get_instance().get_ascend_soc_version()
    ascend_chip_type = os.getenv("ASCEND_CHIP_TYPE", "UNSET")
    if ascend_chip_type not in ["910a", "910b", "UNSET"]:
        raise EnvironmentError(f"ASCEND_CHIP_TYPE should be in ['910a', '910b'],but get {ascend_chip_type}")
    if ascend_chip_type == "UNSET":
        logger.info(
            "Environment variables need to be set manually to obtain the chip type,"
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
    version_valid = is_version_ge(ms.__version__, "2.2.0")
    # below ms 2.2.0 is not support
    if not version_valid:
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
    fa_dtype = ms.uint8
    cur_ver = ms.__version__
    if is_version_ge(cur_ver, "2.2.0") and not is_version_ge(cur_ver, "2.2.1"):
        fa_dtype = ms.float16
    elif is_version_ge(cur_ver, "2.2.1"):
        fa_dtype = ms.uint8
    return fa_dtype
