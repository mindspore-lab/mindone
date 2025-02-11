import logging
import random

import numpy as np

import mindspore as ms

_logger = logging.getLogger(__name__)


class NoOp:
    """useful for distributed training No-Ops"""

    def __getattr__(self, name):
        return self.noop

    def noop(self, *args, **kwargs):
        return


def set_random_seed(seed):
    """Set Random Seed"""
    _logger.debug("Random seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    ms.set_seed(seed)


class Struct:
    def __init__(self, dict_):
        self.__dict__.update(dict_)
