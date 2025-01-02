import numpy as np

import mindspore as ms
from mindspore import mint


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False, dtype=ms.float32):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = mint.exp(-mint.log(ms.Tensor(max_period, dtype)) * mint.arange(start=0, end=half, dtype=dtype) / half)
        args = timesteps[:, None].float() * freqs[None]
        embedding = mint.cat([mint.cos(args), mint.sin(args)], dim=-1)
        if dim % 2:
            embedding = mint.cat([embedding, mint.zeros_like(embedding[:, :1])], dim=-1)
    else:
        timesteps = timesteps.unsqueeze(1)
        embedding = timesteps.repeat_interleave(dim, 1)
        # embedding = repeat(timesteps, "b -> b d", d=dim)
    return embedding


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def make_beta_schedule(schedule="linear", n_timestep=1000, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        start = linear_start**0.5
        stop = linear_end**0.5
        num = n_timestep
        betas = (np.linspace(start, stop, num) ** 2).astype(np.float32)
    elif schedule == "squaredcos_cap_v2":
        return betas_for_alpha_bar(
            n_timestep,
            lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2,
        )
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")

    return betas
