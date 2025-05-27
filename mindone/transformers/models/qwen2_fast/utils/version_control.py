# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""MindSpore Version Control"""
import os
from functools import wraps
from transformers import logging

import mindspore as ms


logger = logging.get_logger(__name__)


def get_ascend_soc_version():
    """Get ascend soc version."""
    if is_version_ge(ms.__version__, "2.2.0"):
        from mindspore._c_expression import MSContext
        return MSContext.get_instance().get_ascend_soc_version()
    ascend_chip_type = os.getenv("ASCEND_CHIP_TYPE", "UNSET")
    if ascend_chip_type not in ["910a", "910b", "UNSET"]:
        raise EnvironmentError(f"ASCEND_CHIP_TYPE should be in ['910a', '910b'],but get {ascend_chip_type}")
    if ascend_chip_type == "UNSET":
        logger.info("Environment variables need to be set manually to obtain the chip type,"
                    "which can be set as follows: \n"
                    "For Atlas 800, run 'export ASCEND_CHIP_TYPE=910a' before the program runs.\n"
                    "For Atlas 800T A2, run 'export ASCEND_CHIP_TYPE=910b' before the program runs.\n"
                    "If you need to get chip information automatically, MindSpore 2.2 and above is recommended")
    return ascend_chip_type


def is_910a():
    device = get_ascend_soc_version()
    return device in ['910a', 'ascend910']


def is_910b():
    device = get_ascend_soc_version()
    return device in ['910b', 'ascend910b']


def need_nz():
    device = get_ascend_soc_version()
    return device in ['310p', 'ascend310p', '910a', 'ascend910']


def is_version_ge(current_version, base_version):
    """
        return current_version >= base_version.
        Check whether the current version is higher than or equal to the base version.
        for current_version: 1.8.1, base_version: 1.11.0, it return False.
    """
    version_split_char = '.'
    if version_split_char not in base_version or version_split_char not in current_version:
        raise ValueError("The version string will contain the `.`."
                         "For example, current_version 1.8.1， base_version: 1.11.0.")
    for x, y in zip(current_version.split(version_split_char), base_version.split(version_split_char)):
        if not x.isdigit() or not y.isdigit():
            continue
        if int(x) != int(y):
            return int(x) >= int(y)
    return True


def check_valid_big_kernel():
    """check mindspore version is valid for big kernel SiLU and LlamaRMSNorm Ops"""
    version_valid = is_version_ge(ms.__version__, "2.2.10")
    # below ms 2.2.10 is not support
    if not version_valid:
        logger.warning("Current MindSpore do not support fusion operator SiLU and RMSNorm, "
                       "please upgrade to 2.2.10 or later version.")
        result = False
    else:
        result = True
    return result


def check_rmsnorm_big_kernel_valid():
    """check whether rmsnorm big kernel is valid"""
    if check_valid_big_kernel() and not is_910a():
        return True
    return False
