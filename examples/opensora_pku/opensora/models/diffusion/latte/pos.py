import numpy as np

from mindone.diffusers.models.embeddings import get_1d_sincos_pos_embed_from_grid


def get_1d_sincos_pos_embed(embed_dim, length, interpolation_scale=1.0, base_size=16):
    pos = np.arange(0, length)[:, None] / interpolation_scale
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, pos)
    return pos_embed
