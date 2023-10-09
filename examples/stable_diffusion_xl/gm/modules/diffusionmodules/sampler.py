"""
    Partially ported from https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py
    Reference to https://github.com/Stability-AI/generative-models
"""
from typing import Dict, Union

import numpy as np
from gm.modules.diffusionmodules.sampling_utils import to_d, to_neg_log_sigma, to_sigma
from gm.util import append_dims, default, instantiate_from_config
from omegaconf import ListConfig, OmegaConf
from tqdm import tqdm

import mindspore as ms
from mindspore import Tensor, ops

DEFAULT_GUIDER = {"target": "gm.modules.diffusionmodules.guiders.IdentityGuider"}


class BaseDiffusionSampler:
    def __init__(
        self,
        discretization_config: Union[Dict, ListConfig, OmegaConf],
        num_steps: Union[int, None] = None,
        guider_config: Union[Dict, ListConfig, OmegaConf, None] = None,
        verbose: bool = False,
    ):
        self.num_steps = num_steps
        self.discretization = instantiate_from_config(discretization_config)
        self.guider = instantiate_from_config(
            default(
                guider_config,
                DEFAULT_GUIDER,
            )
        )
        self.verbose = verbose

    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        sigmas = self.discretization(self.num_steps if num_steps is None else num_steps)

        uc = default(uc, cond)

        x *= Tensor(np.sqrt(1.0 + sigmas[0] ** 2.0), x.dtype)
        num_sigmas = len(sigmas)

        s_in = ops.ones((x.shape[0],), x.dtype)

        return x, s_in, sigmas, num_sigmas, cond, uc

    def denoise(self, x, model, sigma, cond, uc):
        noised_input, sigmas, cond = self.guider.prepare_inputs(x, sigma, cond, uc)
        cond = model.openai_input_warpper(cond)
        c_skip, c_out, c_in, c_noise = model.denoiser(sigmas, noised_input.ndim)
        model_output = model.model(ops.cast(noised_input * c_in, ms.float32), ops.cast(c_noise, ms.int32), **cond)
        model_output = model_output.astype(ms.float32)
        denoised = model_output * c_out + noised_input * c_skip
        denoised = self.guider(denoised, sigma)
        return denoised

    def get_sigma_gen(self, num_sigmas):
        sigma_generator = range(num_sigmas - 1)
        if self.verbose:
            print("#" * 30, " Sampling setting ", "#" * 30)
            print(f"Sampler: {self.__class__.__name__}")
            print(f"Discretization: {self.discretization.__class__.__name__}")
            print(f"Guider: {self.guider.__class__.__name__}")
            sigma_generator = tqdm(
                sigma_generator,
                total=(num_sigmas - 1),
                desc=f"Sampling with {self.__class__.__name__} for {(num_sigmas - 1)} steps",
            )
        return sigma_generator


class SingleStepDiffusionSampler(BaseDiffusionSampler):
    def sampler_step(self, sigma, next_sigma, model, x, cond, uc, *args, **kwargs):
        raise NotImplementedError

    def euler_step(self, x, d, dt):
        return x + dt * d


class EDMSampler(SingleStepDiffusionSampler):
    def __init__(self, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise

    def sampler_step(self, sigma, next_sigma, model, x, cond, uc=None, gamma=0.0):
        sigma_hat = sigma * (gamma + 1.0)
        if gamma > 0:
            eps = Tensor(np.random.randn(*x.shape), x.dtype) * self.s_noise
            x = x + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5

        denoised = self.denoise(x, model, sigma_hat, cond, uc)
        d = to_d(x, sigma_hat, denoised)
        dt = append_dims(next_sigma - sigma_hat, x.ndim)

        euler_step = self.euler_step(x, d, dt)
        x = self.possible_correction_step(euler_step, x, d, dt, next_sigma, model, cond, uc)
        return x

    def __call__(self, model, x, cond, uc=None, num_steps=None):
        x = ops.cast(x, ms.float32)
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(x, cond, uc, num_steps)

        for i in self.get_sigma_gen(num_sigmas):
            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1) if self.s_tmin <= sigmas[i] <= self.s_tmax else 0.0
            )
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                model,
                x,
                cond,
                uc,
                gamma,
            )

        return x


class EulerEDMSampler(EDMSampler):
    def possible_correction_step(self, euler_step, x, d, dt, next_sigma, model, cond, uc):
        return euler_step


class HeunEDMSampler(EDMSampler):
    def possible_correction_step(self, euler_step, x, d, dt, next_sigma, model, cond, uc):
        if ops.sum(next_sigma) < 1e-14:
            # Save a network evaluation if all noise levels are 0
            return euler_step
        else:
            denoised = self.denoise(euler_step, model, next_sigma, cond, uc)
            d_new = to_d(euler_step, next_sigma, denoised)
            d_prime = (d + d_new) / 2.0

            # apply correction if noise level is not 0
            x = ops.where(append_dims(next_sigma, x.ndim) > 0.0, x + d_prime * dt, euler_step)
            return x


class DPMPP2MSampler(BaseDiffusionSampler):
    def get_variables(self, sigma, next_sigma, previous_sigma=None):
        t, t_next = [to_neg_log_sigma(s) for s in (sigma, next_sigma)]
        h = t_next - t

        if previous_sigma is not None:
            h_last = t - to_neg_log_sigma(previous_sigma)
            r = h_last / h
            return h, r, t, t_next
        else:
            return h, None, t, t_next

    def get_mult(self, h, r, t, t_next, previous_sigma):
        mult1 = to_sigma(t_next) / to_sigma(t)
        mult2 = (-h).expm1()

        if previous_sigma is not None:
            mult3 = 1 + 1 / (2 * r)
            mult4 = 1 / (2 * r)
            return mult1, mult2, mult3, mult4
        else:
            return mult1, mult2

    def sampler_step(
        self,
        old_denoised,
        previous_sigma,
        sigma,
        next_sigma,
        model,
        x,
        cond,
        uc=None,
    ):
        denoised = self.denoise(x, model, sigma, cond, uc)

        h, r, t, t_next = self.get_variables(sigma, next_sigma, previous_sigma)
        mult = [append_dims(mult, x.ndim) for mult in self.get_mult(h, r, t, t_next, previous_sigma)]

        x_standard = mult[0] * x - mult[1] * denoised
        if old_denoised is None or ops.sum(next_sigma) < 1e-14:
            # Save a network evaluation if all noise levels are 0 or on the first step
            return x_standard, denoised
        else:
            denoised_d = mult[2] * denoised - mult[3] * old_denoised
            x_advanced = mult[0] * x - mult[1] * denoised_d

            # apply correction if noise level is not 0 and not first step
            x = ops.where(append_dims(next_sigma, x.ndim) > 0.0, x_advanced, x_standard)

        return x, denoised

    def __call__(self, model, x, cond, uc=None, num_steps=None, **kwargs):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(x, cond, uc, num_steps)

        old_denoised = None
        for i in self.get_sigma_gen(num_sigmas):
            x, old_denoised = self.sampler_step(
                old_denoised,
                None if i == 0 else s_in * sigmas[i - 1],
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                model,
                x,
                cond,
                uc=uc,
            )

        return x
