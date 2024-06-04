import random

import numpy as np

import mindspore as ms


def set_random_seed(seed: int):
    """Set Random Seed"""
    random.seed(seed)
    np.random.seed(seed)
    ms.set_seed(seed)
