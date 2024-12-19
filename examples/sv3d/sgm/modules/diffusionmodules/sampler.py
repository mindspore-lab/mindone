"""
    Partially ported from https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py
    Reference to https://github.com/Stability-AI/generative-models

    This sampler has been tested to pass graph mode modular.
"""
# for modular debugging only
import sys
from typing import Dict, Union

import numpy as np

sys.path.append("/mnt/disk4/fredhong/mindone_master/examples/sv3d/")
sys.path.append("/mnt/disk4/fredhong/mindone_master/")

from omegaconf import ListConfig, OmegaConf
from sgm.helpers import create_model_sv3d
from sgm.modules.diffusionmodules.sampling_utils import to_d
from sgm.util import append_dims, default, instantiate_from_config
from tqdm import tqdm

import mindspore as ms
from mindspore import Tensor, ops

DEFAULT_GUIDER = {"target": "sgm.modules.diffusionmodules.guiders.IdentityGuider"}


class BaseDiffusionSampler:
    def __init__(
        self,
        discretization_config: Union[Dict, ListConfig, OmegaConf],
        num_steps: Union[int, None] = None,
        guider_config: Union[Dict, ListConfig, OmegaConf, None] = None,
        verbose: bool = False,
        network_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        network_ckpt: str = None,
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

        self.model, _ = create_model_sv3d(network_config, checkpoints=network_ckpt, freeze=True)

    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        sigmas = self.discretization(self.num_steps if num_steps is None else num_steps)

        uc = default(uc, cond)

        x *= Tensor(np.sqrt(1.0 + sigmas[0] ** 2.0), x.dtype)
        num_sigmas = len(sigmas)

        s_in = ops.ones((x.shape[0],), x.dtype)

        return x, s_in, sigmas, num_sigmas, cond, uc

    def denoise(self, x, sigma, cond, uc, **kwargs):
        noised_input, sigmas, cond = self.guider.prepare_inputs(x, sigma, cond, uc)
        cond = self.model.openai_input_warpper(cond)
        c_skip, c_out, c_in, c_noise = self.model.denoiser(sigmas, noised_input.ndim)
        model_output = self.model.model(noised_input * c_in, c_noise, **cond, **kwargs)
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
            if self.guider.__class__.__name__ == "VanillaCFG":
                print(f"Thresholding: {self.guider.dyn_thresh.__class__.__name__}")
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


class EDMSampler(SingleStepDiffusionSampler):
    def __init__(self, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise

    def sampler_step(self, sigma, next_sigma, x, cond, uc=None, gamma=0.0, **kwargs):
        sigma_hat = sigma * (gamma + 1.0)
        if gamma > 0:
            eps = Tensor(np.random.randn(*x.shape), x.dtype) * self.s_noise
            x = x + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5

        denoised = self.denoise(x, sigma_hat, cond, uc, **kwargs)
        d = to_d(x, sigma_hat, denoised)
        dt = append_dims(next_sigma - sigma_hat, x.ndim)

        euler_step = self.euler_step(x, d, dt)
        x = self.possible_correction_step(euler_step, x, d, dt, next_sigma, cond, uc)
        return x

    def __call__(self, input, cond, uc=None, num_steps=None, **kwargs):
        x = input
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(x, cond, uc, num_steps)

        for i in self.get_sigma_gen(num_sigmas):
            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1) if self.s_tmin <= sigmas[i] <= self.s_tmax else 0.0
            )
            x = self.sampler_step(s_in * sigmas[i], s_in * sigmas[i + 1], x, cond, uc, gamma, **kwargs)
        return x


class EulerEDMSampler(EDMSampler):
    def _notify_model_while_sampling(self, model):
        """This for graph mode, to avoid passing model as arg"""
        self.model = model

    def possible_correction_step(self, euler_step, x, d, dt, next_sigma, cond, uc):
        return euler_step
