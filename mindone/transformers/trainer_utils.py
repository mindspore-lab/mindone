import copy
import functools
import gc
import inspect
import os
import random
import re
import threading
import time
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np

import mindspore as ms

from .utils.generic import ExplicitEnum

from transformers.trainer_utils import (
    IntervalStrategy,
    EvalPrediction,
    SchedulerType,
    EvaluationStrategy,
    HubStrategy,
    RemoveColumnsCollator,
    speed_metrics,
    number_of_arguments,
    has_length,
    get_last_checkpoint
)


def enable_full_determinism(seed: int):
    """
    Helper function for reproducible behavior during distributed training. See
    - https://pytorch.org/docs/stable/notes/randomness.html for pytorch
    - https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism for tensorflow
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
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    Args:
        seed (`int`):
            The seed to set.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)
    ms.set_seed(seed)
