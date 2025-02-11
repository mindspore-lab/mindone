from functools import partial
from typing import List

import numpy as np

import mindspore as ms
from mindspore import Parameter, Tensor, mint, nn

from .modules.diffusionmodules.util import extract_into_tensor, make_beta_schedule
from .util import default, exists, instantiate_from_config


class DiffusionWrapper(nn.Cell):
    def __init__(self, diffusion_model):
        super().__init__()
        self.diffusion_model = diffusion_model

    def construct(self, *args, **kwargs):
        return self.diffusion_model(*args, **kwargs)


class LatentDiffusionInterface(nn.Cell):
    """a simple interface class for LDM inference"""

    def __init__(
        self,
        unet_config,
        cond_stage_config,
        first_stage_config,
        parameterization="eps",
        scale_factor=0.18215,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=0.00085,
        linear_end=0.0120,
        cosine_s=8e-3,
        given_betas=None,
        *args,
        **kwargs,
    ):
        super().__init__()

        unet = instantiate_from_config(unet_config)
        self.model = DiffusionWrapper(unet)
        self.cond_stage_model = instantiate_from_config(cond_stage_config)
        self.first_stage_model = instantiate_from_config(first_stage_config)

        self.parameterization = parameterization
        self.scale_factor = scale_factor
        self.register_schedule(
            given_betas=given_betas,
            beta_schedule=beta_schedule,
            timesteps=timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )

    def register_buffer_ms(self, name: str, tensor: Tensor):
        return setattr(self, name, Parameter(default_input=tensor, requires_grad=False))

    def register_schedule(
        self,
        given_betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        if exists(given_betas):
            betas = given_betas
        else:
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

        to_ms_tensor = partial(Tensor, dtype=ms.float32)

        self.register_buffer_ms("betas", to_ms_tensor(betas))
        self.register_buffer_ms("alphas_cumprod", to_ms_tensor(alphas_cumprod))
        self.register_buffer_ms("alphas_cumprod_prev", to_ms_tensor(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer_ms("sqrt_alphas_cumprod", to_ms_tensor(np.sqrt(alphas_cumprod)))
        self.register_buffer_ms("sqrt_one_minus_alphas_cumprod", to_ms_tensor(np.sqrt(1.0 - alphas_cumprod)))
        self.register_buffer_ms("log_one_minus_alphas_cumprod", to_ms_tensor(np.log(1.0 - alphas_cumprod)))
        self.register_buffer_ms("sqrt_recip_alphas_cumprod", to_ms_tensor(np.sqrt(1.0 / alphas_cumprod)))
        self.register_buffer_ms("sqrt_recipm1_alphas_cumprod", to_ms_tensor(np.sqrt(1.0 / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.v_posterior = 0
        posterior_variance = (1 - self.v_posterior) * betas * (1.0 - alphas_cumprod_prev) / (
            1.0 - alphas_cumprod
        ) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer_ms("posterior_variance", to_ms_tensor(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer_ms(
            "posterior_log_variance_clipped", to_ms_tensor(np.log(np.maximum(posterior_variance, 1e-20)))
        )
        self.register_buffer_ms(
            "posterior_mean_coef1", to_ms_tensor(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        )
        self.register_buffer_ms(
            "posterior_mean_coef2", to_ms_tensor((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod))
        )

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: mint.normal(size=x_start.shape))
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def get_v(self, x, noise, t):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise
            - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_start_from_z_and_v(self, x_t, t, v):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def predict_eps_from_z_and_v(self, x_t, t, v):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * v
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * x_t
        )

    def apply_model(self, x_noisy, t, cond, **kwargs):
        assert isinstance(cond, dict)
        return self.model(x_noisy, t, **cond, **kwargs)

    def get_learned_conditioning(self, prompts: List[str]):
        return self.cond_stage_model(prompts)

    def get_first_stage_encoding(self, encoder_posterior):
        # if isinstance(encoder_posterior, DiagonalGaussianDistribution):
        z = encoder_posterior
        # elif isinstance(encoder_posterior, Tensor):
        #     z = encoder_posterior
        # else:
        #     raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    def encode_first_stage(self, x):
        return self.first_stage_model.encode(x)

    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        return self.first_stage_model.decode(z)
