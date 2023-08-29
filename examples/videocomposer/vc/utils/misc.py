import binascii
import os
import random

import numpy as np

import mindspore as ms

__all__ = [
    "setup_seed",
    "rand_name",
    "get_abspath_of_weights",
]


def setup_seed(seed):
    ms.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def rand_name(length=8, suffix=""):
    name = binascii.b2a_hex(os.urandom(length)).decode("utf-8")
    if suffix:
        if not suffix.startswith("."):
            suffix = "." + suffix
        name += suffix
    return name


def get_abspath_of_weights(file_or_dirname=None, cache_dir=None):
    if cache_dir is None:
        cache_dir = os.path.join("/".join(os.path.abspath(__file__).split("/")[:-3]), "model_weights")
    base_path = os.path.join(cache_dir, file_or_dirname)
    return base_path
