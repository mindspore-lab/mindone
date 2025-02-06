import os
import random

import numpy as np

import mindspore as ms


def enable_full_determinism(seed: int):
    """
    Helper function for reproducible behavior during distributed training. See
    - https://www.mindspore.cn/docs/zh-CN/r2.3.1/index.html for MindSpore
    """
    # set seed first
    set_seed(seed)

    ms.set_context(deterministic="ON")
    print("WARNING: Set mindspore context `deterministic=ON`")

    os.environ["HCCL_DETERMINISTIC"] = "true"
    os.environ["ASCEND_LAUNCH_BLOCKING"] = "1"
    os.environ["TE_PARALLEL_COMPILER"] = "1"


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `mindspore`.

    Args:
        seed (`int`):
            The seed to set.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)
    ms.set_seed(seed)
