import random

import numpy as np

import mindspore as ms


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    ms.set_seed(seed)


def pad_to_fit(h, w):
    BEST_H, BEST_W = 720, 1280

    if h < BEST_H:
        h1, h2 = _create_pad(h, BEST_H)
    elif h == BEST_H:
        h1 = h2 = 0
    else:
        h1 = 0
        h2 = int((h + 48) // 64 * 64) + 64 - 48 - h

    if w < BEST_W:
        w1, w2 = _create_pad(w, BEST_W)
    elif w == BEST_W:
        w1 = w2 = 0
    else:
        w1 = 0
        w2 = int(w // 64 * 64) + 64 - w
    return (w1, w2, h1, h2)


def _create_pad(h, max_len):
    h1 = int((max_len - h) // 2)
    h2 = max_len - h1 - h
    return h1, h2


def make_chunks(f_num, interp_f_num, chunk_overlap_ratio=0.5):
    MAX_CHUNK_LEN = 32
    MAX_O_LEN = MAX_CHUNK_LEN * chunk_overlap_ratio
    chunk_len = int((MAX_CHUNK_LEN - 1) // (1 + interp_f_num) * (interp_f_num + 1) + 1)
    o_len = int((MAX_O_LEN - 1) // (1 + interp_f_num) * (interp_f_num + 1) + 1)
    chunk_inds = sliding_windows_1d(f_num, chunk_len, o_len)
    return chunk_inds


def sliding_windows_1d(full_size, window_size, overlap_size):
    stride = window_size - overlap_size
    ind = 0
    coords = []
    while ind < full_size:
        if ind + round(window_size * 1.33) >= full_size or ind + window_size >= full_size:
            coords.append((ind, full_size))
            break
        else:
            coords.append((ind, ind + window_size))
            ind += stride
    return coords


# https://github.com/csslc/CCSR/blob/main/model/q_sampler.py#L503
def gaussian_weights(tile_width, tile_height):
    """Generates a gaussian mask of weights for tile contributions"""
    latent_width = tile_width
    latent_height = tile_height
    var = 0.02
    midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
    x_probs = [
        np.exp(-(x - midpoint) * (x - midpoint) / (latent_width * latent_width) / (2 * var)) / np.sqrt(2 * np.pi * var)
        for x in range(latent_width)
    ]
    midpoint = latent_height / 2
    y_probs = [
        np.exp(-(y - midpoint) * (y - midpoint) / (latent_height * latent_height) / (2 * var))
        / np.sqrt(2 * np.pi * var)
        for y in range(latent_height)
    ]
    weights = np.outer(y_probs, x_probs)
    return weights


def blend_time(a, b, blend_extent):
    blend_extent = min(a.shape[2], b.shape[2], blend_extent)
    for t in range(blend_extent):
        coef = t / blend_extent
        b[:, :, t, :, :] = a[:, :, -blend_extent + t, :, :] * (1 - coef) + b[:, :, t, :, :] * coef
    return b
