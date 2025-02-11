import numpy as np


def make_beta_schedule(schedule="linear", n_timestep=1000, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        start = linear_start**0.5
        stop = linear_end**0.5
        num = n_timestep
        betas = (np.linspace(start, stop, num) ** 2).astype(np.float32)
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")

    return betas


def rescale_zero_terminal_snr(betas):
    """
    Rescales betas to have zero terminal SNR Based on https://arxiv.org/pdf/2305.08891.pdf (Algorithm 1)

    Args:
        betas (`numpy.ndarray`):
            the betas that the scheduler is being initialized with.

    Returns:
        `numpy.ndarray`: rescaled betas with zero terminal SNR
    """
    # Convert betas to alphas_bar_sqrt
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_bar_sqrt = np.sqrt(alphas_cumprod)

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].copy()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].copy()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt**2  # Revert sqrt
    alphas = alphas_bar[1:] / alphas_bar[:-1]  # Revert cumprod
    alphas = np.concatenate([alphas_bar[0:1], alphas])
    betas = 1 - alphas

    return betas
