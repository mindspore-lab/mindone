import random

import numpy as np
import torch

import mindspore as ms

global_rng = random.Random(42)  # fix seed for reproducibility


def ids_numpy(shape, vocab_size, rng=None, name=None):
    #  Creates a random int32 numpy array of the shape within the vocab size
    if rng is None:
        rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.randint(1, vocab_size - 1))

    return np.array(values, dtype=np.int64).reshape(shape)


def random_attention_mask(shape, rng=None, name=None):
    attn_mask = ids_numpy(shape, vocab_size=2, rng=None, name=None)
    # make sure that at least one token is attended to for each batch
    # we choose the 1st token so this property of `at least one being non-zero` still holds after applying causal mask
    attn_mask[:, 0] = 1
    return attn_mask


def floats_numpy(shape, scale=1.0, rng=None, name=None):
    """Creates a random float32 numpy"""
    if rng is None:
        rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.random() * scale)

    return np.array(values, dtype=np.float32).reshape(shape)


def set_random_seed_for_all(seed, python=True, numpy=True, mindspore=True, pytorch=True):
    """Set random seed for reproducibility if the seed value is valid."""
    seed = int(seed)
    assert seed > 0 and seed < 4294967295
    if python:
        random.seed(seed)
    if numpy:
        np.random.seed(seed)
    if mindspore:
        ms.set_seed(seed)
    if pytorch:
        torch.manual_seed(seed)
    print(
        f"set global seed for {'python' if python else ''}, {'numpy' if numpy else ''}, "
        f"{'mindspore' if mindspore else ''}, {'pytorch' if pytorch else ''} to {seed}"
    )
