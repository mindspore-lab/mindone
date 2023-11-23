import numpy as np
from gm.modules.diffusionmodules.sampling_utils import to_d
from gm.util import append_dims

import mindspore as ms
import mindspore.nn as nn
from mindspore import ops


class EulerEDMSampler(nn.Cell):
    def __init__(
        self,
        num_steps=None,
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        scale=5.0,
        num_timesteps=1000,
        linear_start=0.00085,
        linear_end=0.0120,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise
        self.scale = scale
        self.sigma_hat = ms.ops.zeros(1, ms.float32)
        self.num_timesteps = num_timesteps
        betas = ms.ops.linspace(linear_start**0.5, linear_end**0.5, num_timesteps).astype(ms.float32) ** 2
        alphas = 1.0 - betas
        self.alphas_cumprod = ms.ops.cumprod(alphas, dim=0)

    def prepare_sampling_loop(self, x):
        sigmas = self.get_sigmas()
        x = ms.ops.mul(x, ms.ops.sqrt(1.0 + sigmas[0] ** 2.0))
        s_in = ops.ones((x.shape[0],), x.dtype)
        return x, s_in

    def pre_model_input(self, iter_index, x, s_in):
        sigmas = self.get_sigmas()
        num_sigmas = self.num_steps
        if self.s_tmin <= sigmas[iter_index] and sigmas[iter_index] <= self.s_tmax:
            gamma = min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
        else:
            gamma = 0.0

        sigma = s_in * sigmas[iter_index]
        next_sigma = s_in * sigmas[iter_index + 1]

        sigma_hat = sigma * (gamma + 1.0)
        if gamma > 0:
            eps = ms.Tensor(np.random.randn(*x.shape), x.dtype) * self.s_noise
            x = x + eps * append_dims(sigma_hat**2 - s_in * sigmas[iter_index] ** 2, x.ndim) ** 0.5
        noised_input = ops.concat((x, x))
        sigma_hat_s = ops.concat((sigma_hat, sigma_hat))
        return noised_input, sigma_hat_s, next_sigma, sigma_hat

    def construct(self, model_output, c_out, noised_input, c_skip, scale, x, sigma_hat, next_sigma):
        denoised = model_output * c_out + noised_input * c_skip
        _id = denoised.shape[0] // 2
        x_u, x_c = denoised[:_id], denoised[_id:]
        denoised = x_u + scale * (x_c - x_u)
        d = to_d(x, sigma_hat, denoised)
        dt = append_dims(next_sigma - sigma_hat, x.ndim)
        euler_step = x + dt * d
        return euler_step

    def get_sigmas(self):
        n = self.num_steps
        if n < self.num_timesteps:
            timesteps = ms.ops.linspace(self.num_timesteps - 1, 0, n).astype(ms.int32)[::-1]
            alphas_cumprod = self.alphas_cumprod[timesteps]
        elif n == self.num_timesteps:
            alphas_cumprod = self.alphas_cumprod
        else:
            raise ValueError
        sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        sigmas = ms.ops.flip(sigmas, (0,))
        sigmas = append_zero(sigmas)
        return sigmas


def append_zero(x):
    return ms.ops.concat([x, ms.ops.zeros([1], dtype=x.dtype)])
