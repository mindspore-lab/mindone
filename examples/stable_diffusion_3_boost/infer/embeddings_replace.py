import math

import mindspore as ms
from mindspore import nn, ops, mint

def get_timestep_embedding(
    timesteps: ms.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -ops.log(ms.Tensor(max_period, dtype=ms.float32)) * ops.arange(start=0, end=half_dim, dtype=ms.float32)
    exponent = mint.div(exponent, (half_dim - downscale_freq_shift))

    emb = ops.exp(exponent)
    emb = timesteps.expand_dims(1).float() * emb.expand_dims(0)

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = ops.cat([ops.sin(emb), ops.cos(emb)], axis=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        sin, cos = mint.split(emb, half_dim, dim=1)
        emb = ops.cat((cos, sin), axis=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = ops.pad(emb, (0, 1, 0, 0))
    return emb
