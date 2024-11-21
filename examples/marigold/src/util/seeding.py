import logging
import random

import numpy as np

from mindspore import set_seed


def seed_all(seed: int = 0):
    """
    Set random seeds of all components.
    """
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)


def generate_seed_sequence(
    initial_seed: int,
    length: int,
    min_val=0x0000_0000_0000_0000,
    max_val=0x0000_0000_FFFF_FFFF,
):
    if initial_seed is None:
        logging.warning("initial_seed is None, reproducibility is not guaranteed")
    random.seed(initial_seed)

    seed_sequence = []

    for _ in range(length):
        seed = random.randint(min_val, max_val)

        seed_sequence.append(seed)

    return seed_sequence
