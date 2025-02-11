from functools import partial

import numpy as np
from ldm.modules.diffusionmodules.util import make_beta_schedule
from ldm.util import extract_into_tensor

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor


class AbstractLowScaleModel(nn.Cell):
    # for concatenating a downsampled image to the latent representation
    def __init__(self, noise_schedule_config=None):
        super().__init__()
        if noise_schedule_config is not None:
            self.register_schedule(**noise_schedule_config)

    def register_schedule(
        self, beta_schedule="linear", timesteps=1000, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3
    ):
        betas = make_beta_schedule(
            beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s
        )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, "alphas have to be defined for each timestep"

        to_ms = partial(Tensor, dtype=ms.float32)

        self.betas = to_ms(betas)
        self.alphas_cumprod = to_ms(alphas_cumprod)
        alphas_cumprod_prev = to_ms(alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = to_ms(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = to_ms(np.sqrt(1.0 - alphas_cumprod))
        self.log_one_minus_alphas_cumprod = to_ms(np.log(1.0 - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = to_ms(np.sqrt(1.0 / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = to_ms(np.sqrt(1.0 / alphas_cumprod - 1))

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = ops.randn_like(x_start)

        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def construct(self, x):
        return x, None

    def decode(self, x):
        return x


class ImageConcatWithNoiseAugmentation(AbstractLowScaleModel):
    def __init__(self, noise_schedule_config, max_noise_level=1000):
        super().__init__(noise_schedule_config=noise_schedule_config)
        self.max_noise_level = max_noise_level

    def construct(self, x, noise_level=None):
        if noise_level is None:
            noise_level = ms.numpy.randint(0, self.max_noise_level, (x.shape[0],), dtype=ms.int64)
        z = self.q_sample(x, noise_level)
        return z, noise_level
