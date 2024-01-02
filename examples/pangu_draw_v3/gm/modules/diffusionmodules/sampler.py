"""
    Partially ported from https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py
    Reference to https://github.com/Stability-AI/generative-models
"""
from typing import Dict, Union

import numpy as np
from gm.modules.diffusionmodules.sampling_utils import get_ancestral_step, to_d, to_neg_log_sigma, to_sigma
from gm.util import append_dims, default, instantiate_from_config
from omegaconf import ListConfig, OmegaConf
from scipy import integrate
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

    def denoise(self, x, model, sigma, cond, uc, **kwargs):
        noised_input, sigmas, cond = self.guider.prepare_inputs(x, sigma, cond, uc)
        cond = model.openai_input_warpper(cond)
        c_skip, c_out, c_in, c_noise = model.denoiser(sigmas, noised_input.ndim)
        model_output = model.model(
            ops.cast(noised_input * c_in, ms.float32), ops.cast(c_noise, ms.int32), **cond, **kwargs
        )
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
    def sampler_step(self, sigma, next_sigma, model, x, cond, uc=None, gamma=0.0, **kwargs):
        sigma_hat = sigma * (gamma + 1.0)
        if gamma > 0:
            eps = Tensor(np.random.randn(*x.shape), x.dtype) * self.s_noise
            x = x + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5

        denoised = self.denoise(x, model, sigma_hat, cond, uc, **kwargs)
        d = to_d(x, sigma_hat, denoised)
        dt = append_dims(next_sigma - sigma_hat, x.ndim)

        euler_step = self.euler_step(x, d, dt)
        x = euler_step
        return x

    def euler_step(self, x, d, dt):
        return x + dt * d


class AncestralSampler(SingleStepDiffusionSampler):
    def __init__(self, eta=1.0, s_noise=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.eta = eta
        self.s_noise = s_noise
        self.noise_sampler = lambda x: Tensor(np.random.randn(*x.shape), x.dtype)

    def ancestral_euler_step(self, x, denoised, sigma, sigma_down):
        d = to_d(x, sigma, denoised)
        dt = append_dims(sigma_down - sigma, x.ndim)

        return self.euler_step(x, d, dt)

    def ancestral_step(self, x, sigma, next_sigma, sigma_up):
        x = ops.where(
            append_dims(next_sigma, x.ndim) > 0.0,
            x + self.noise_sampler(x) * self.s_noise * append_dims(sigma_up, x.ndim),
            x,
        )
        return x

    def __call__(self, model, x, cond, uc=None, num_steps=None, **kwargs):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(x, cond, uc, num_steps)

        for i in self.get_sigma_gen(num_sigmas):
            x = self.sampler_step(s_in * sigmas[i], s_in * sigmas[i + 1], model, x, cond, uc, **kwargs)

        return x


class LinearMultistepSampler(BaseDiffusionSampler):
    def __init__(self, order=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.order = order

    def linear_multistep_coeff(self, order, t, i, j, epsrel=1e-4):
        if order - 1 > i:
            raise ValueError(f"Order {order} too high for step {i}")

        def fn(tau):
            prod = 1.0
            for k in range(order):
                if j == k:
                    continue
                prod *= (tau - t[i - k]) / (t[i - j] - t[i - k])
            return prod

        return integrate.quad(fn, t[i], t[i + 1], epsrel=epsrel)[0]

    def __call__(self, model, x, cond, uc=None, num_steps=None, **kwargs):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(x, cond, uc, num_steps)

        ds = []
        sigmas_cpu = sigmas  # 转换为NumPy数组
        for i in self.get_sigma_gen(num_sigmas):
            sigma = s_in * sigmas[i]
            denoised = self.denoise(x, model, sigma, cond, uc, **kwargs)
            d = to_d(x, sigma, denoised)  # 假定有一个名为to_d的函数用于计算d
            ds.append(d)
            if len(ds) > self.order:
                ds.pop(0)
            cur_order = min(i + 1, self.order)
            coeffs = [self.linear_multistep_coeff(cur_order, sigmas_cpu, i, j) for j in range(cur_order)]
            x = x + ms.Tensor(np.sum([coeff * d for coeff, d in zip(coeffs, reversed(ds))], axis=0))
            x = x + sum(coeff * d for coeff, d in zip(coeffs, reversed(ds)))

        return x


class EDMSampler(SingleStepDiffusionSampler):
    def __init__(self, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise

    def sampler_step(self, sigma, next_sigma, model, x, cond, uc=None, gamma=0.0, **kwargs):
        sigma_hat = sigma * (gamma + 1.0)
        if gamma > 0:
            eps = Tensor(np.random.randn(*x.shape), x.dtype) * self.s_noise
            x = x + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5

        denoised = self.denoise(x, model, sigma_hat, cond, uc, **kwargs)
        d = to_d(x, sigma_hat, denoised)
        dt = append_dims(next_sigma - sigma_hat, x.ndim)

        euler_step = self.euler_step(x, d, dt)
        x = self.possible_correction_step(euler_step, x, d, dt, next_sigma, model, cond, uc)
        return x

    def __call__(self, model, x, cond, uc=None, num_steps=None, **kwargs):
        x = ops.cast(x, ms.float32)
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(x, cond, uc, num_steps)

        for i in self.get_sigma_gen(num_sigmas):
            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1) if self.s_tmin <= sigmas[i] <= self.s_tmax else 0.0
            )
            x = self.sampler_step(s_in * sigmas[i], s_in * sigmas[i + 1], model, x, cond, uc, gamma, **kwargs)

        return x


class EulerAncestralSampler(AncestralSampler):
    def __init__(self, eta, *args, **kwargs):
        super(EulerAncestralSampler, self).__init__(*args, **kwargs)
        self.eta = eta

    def sampler_step(self, sigma, next_sigma, model, x, cond, uc, **kwargs):
        sigma_down, sigma_up = get_ancestral_step(sigma, next_sigma)
        denoised = self.denoise(x, model, sigma, cond, uc, **kwargs)
        x = self.ancestral_euler_step(x, denoised, sigma, sigma_down)
        x = self.ancestral_step(x, sigma, next_sigma, sigma_up)

        return x


class DPMPP2SAncestralSampler(AncestralSampler):
    def get_variables(self, sigma, sigma_down):
        t, t_next = [to_neg_log_sigma(s) for s in (sigma, sigma_down)]
        h = t_next - t
        s = t + 0.5 * h
        return h, s, t, t_next

    def get_mult(self, h, s, t, t_next):
        mult1 = to_sigma(s) / to_sigma(t)
        mult2 = (-0.5 * h).expm1()
        mult3 = to_sigma(t_next) / to_sigma(t)
        mult4 = (-h).expm1()

        return mult1, mult2, mult3, mult4

    def sampler_step(self, sigma, next_sigma, model, x, cond, uc=None, **kwargs):
        sigma_down, sigma_up = get_ancestral_step(sigma, next_sigma, eta=self.eta)
        denoised = self.denoise(x, model, sigma, cond, uc)
        x_euler = self.ancestral_euler_step(x, denoised, sigma, sigma_down)

        if ops.sum(sigma_down) < 1e-14:
            # Save a network evaluation if all noise levels are 0
            x = x_euler
        else:
            h, s, t, t_next = self.get_variables(sigma, sigma_down)
            mult = [append_dims(mult, x.ndim) for mult in self.get_mult(h, s, t, t_next)]

            x2 = mult[0] * x - mult[1] * denoised
            denoised2 = self.denoise(x2, model, to_sigma(s), cond, uc)
            x_dpmpp2s = mult[2] * x - mult[3] * denoised2

            # apply correction if noise level is not 0
            x = ops.where(append_dims(sigma_down, x.ndim) > 0.0, x_dpmpp2s, x_euler)

        x = self.ancestral_step(x, sigma, next_sigma, sigma_up)
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

    def sampler_step(self, old_denoised, previous_sigma, sigma, next_sigma, model, x, cond, uc=None, **kwargs):
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
                **kwargs,
            )

        return x


# ===========
# PanGu Draw
# ===========


class PanGuEDMSampler(SingleStepDiffusionSampler):
    def __init__(self, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise

    def denoise(self, x, model, sigma, cond, uc, other_c, **kwargs):
        noised_input, sigmas, cond = self.guider.prepare_inputs(x, sigma, cond, uc, other_c)
        cond = model.openai_input_warpper(cond)
        c_skip, c_out, c_in, c_noise = model.denoiser(sigmas, noised_input.ndim)
        model_output = model.model(
            ops.cast(noised_input * c_in, ms.float32), ops.cast(c_noise, ms.int32), **cond, **kwargs
        )
        model_output = model_output.astype(ms.float32)
        denoised = model_output * c_out + noised_input * c_skip
        denoised = self.guider(denoised, sigma)
        return denoised

    def sampler_step(self, sigma, next_sigma, model, x, cond, uc=None, other_c=None, gamma=0.0, **kwargs):
        if other_c is None:
            other_c = []
        sigma_hat = sigma * (gamma + 1.0)
        if gamma > 0:
            eps = Tensor(np.random.randn(*x.shape), x.dtype) * self.s_noise
            x = x + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5

        denoised = self.denoise(x, model, sigma_hat, cond, uc, other_c)
        step_idx = model.denoiser.sigma_to_idx(sigma_hat).numpy()[0].item()
        d = to_d(x, sigma_hat, denoised)
        dt = append_dims(next_sigma - sigma_hat, x.ndim)

        euler_step = self.euler_step(x, d, dt)
        x = self.possible_correction_step(euler_step, x, d, dt, next_sigma, model, cond, uc, other_c)
        return x, step_idx

    def __call__(self, model, high_timestamp_model, x, cond=None, uc=None, other_c=None, num_steps=None, **kwargs):
        x = ops.cast(x, ms.float32)
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(x, cond, uc, num_steps)
        step_idx = 1000

        for i in self.get_sigma_gen(num_sigmas):
            if high_timestamp_model is not None and step_idx >= 500:
                sample_model = high_timestamp_model
                print("using high timestamp model")
            else:
                sample_model = model
                print("using low timestamp model")
            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1) if self.s_tmin <= sigmas[i] <= self.s_tmax else 0.0
            )
            x, step_idx = self.sampler_step(
                s_in * sigmas[i], s_in * sigmas[i + 1], sample_model, x, cond, uc, other_c, gamma, **kwargs
            )

        return x


class PanGuEulerEDMSampler(PanGuEDMSampler):
    def possible_correction_step(self, euler_step, x, d, dt, next_sigma, model, cond, uc, other_c):
        return euler_step
