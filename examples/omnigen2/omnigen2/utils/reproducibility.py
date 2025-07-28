import random

import numpy as np

from mindspore import initial_seed, manual_seed


def worker_init_fn(worker_id, num_processes, num_workers, process_index, seed, same_seed_per_epoch=False):
    if same_seed_per_epoch:
        worker_seed = seed + num_processes + num_workers * process_index + worker_id
    else:
        worker_seed = initial_seed()

    random.seed(worker_seed)
    np.random.seed(worker_seed % 2**32)
    manual_seed(worker_seed)
