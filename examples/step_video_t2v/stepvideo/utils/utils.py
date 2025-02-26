import random

import numpy as np

import mindspore as ms


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    ms.set_seed(seed)
