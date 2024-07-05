# reference to https://github.com/Stability-AI/generative-models

from abc import abstractmethod

import numpy as np
from sgm.modules.diffusionmodules.util import make_beta_schedule
from sgm.util import append_zero


def generate_roughly_equally_spaced_steps(num_substeps: int, max_step: int) -> np.ndarray:
    return np.linspace(max_step - 1, 0, num_substeps, endpoint=False).astype(int)[::-1]


class Discretization:
    def __call__(self, n, do_append_zero=True, flip=False):
        sigmas = self.get_sigmas(n)
        sigmas = append_zero(sigmas) if do_append_zero else sigmas
        sigmas = sigmas if not flip else np.flip(sigmas, (0,))
        return sigmas

    @abstractmethod
    def get_sigmas(self, n):
        pass


class EDMDiscretization(Discretization):
    def __init__(self, sigma_min=0.002, sigma_max=80.0, rho=7.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def get_sigmas(self, n):
        ramp = np.linspace(0, 1, n)
        min_inv_rho = self.sigma_min ** (1 / self.rho)
        max_inv_rho = self.sigma_max ** (1 / self.rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
        return sigmas


class LegacyDDPMDiscretization(Discretization):
    def __init__(
        self,
        linear_start=0.00085,
        linear_end=0.0120,
        num_timesteps=1000,
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        betas = make_beta_schedule("linear", num_timesteps, linear_start=linear_start, linear_end=linear_end)
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)

    def get_sigmas(self, n):
        if n < self.num_timesteps:
            timesteps = generate_roughly_equally_spaced_steps(n, self.num_timesteps)
            alphas_cumprod = self.alphas_cumprod[timesteps]
        elif n == self.num_timesteps:
            alphas_cumprod = self.alphas_cumprod
        else:
            raise ValueError

        sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        return np.flip(sigmas, (0,))


class DiffusersDDPMDiscretization(Discretization):
    """Get the sigmas which strictly follows Diffusers's scheduler pipeline"""

    def __init__(
        self,
        linear_start=0.00085,
        linear_end=0.0120,
        num_timesteps=1000,
        interpolation_type="linear",
        timestep_spacing="leading",
        steps_offset=1,
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.interpolation_type = interpolation_type
        self.timestep_spacing = timestep_spacing
        self.steps_offset = steps_offset
        betas = make_beta_schedule("linear", num_timesteps, linear_start=linear_start, linear_end=linear_end)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        self.sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5

    def get_sigmas(self, n):
        if self.timestep_spacing == "linspace":
            timesteps = np.linspace(0, self.num_timesteps - 1, n, dtype=np.float32)[::-1]
        elif self.timestep_spacing == "leading":
            step_ratio = self.num_timesteps // n
            timesteps = (np.arange(0, n) * step_ratio).round()[::-1].astype(np.float32)
            timesteps += self.steps_offset
        else:
            raise NotImplementedError(f"Unsupported type '{self.timestep_spacing}'")

        if n < self.num_timesteps:
            if self.interpolation_type == "linear":
                sigmas = np.interp(timesteps, np.arange(0, len(self.sigmas)), self.sigmas)
            elif self.interpolation_type == "log_linear":
                sigmas = np.linspace(np.log(self.sigmas[-1]), np.log(self.sigmas[0]), n + 1)
                sigmas = np.exp(sigmas)
            else:
                raise NotImplementedError(f"Unsupported type '{self.interpolation_type}'")
        elif n == self.num_timesteps:
            sigmas = np.flip(self.sigmas, (0,))
        else:
            raise ValueError

        return sigmas


class Img2ImgDiscretizationWrapper:
    """
    wraps a discretizer, and prunes the sigmas
    params:
        strength: float between 0.0 and 1.0. 1.0 means full sampling (all sigmas are returned)
    """

    def __init__(self, discretization, strength: float = 1.0):
        self.discretization = discretization
        self.strength = strength
        assert 0.0 <= self.strength <= 1.0

    def __call__(self, *args, **kwargs):
        # sigmas start large first, and decrease then
        sigmas = self.discretization(*args, **kwargs)
        print("sigmas after discretization, before pruning img2img: ", sigmas)
        sigmas = np.flip(sigmas, (0,))
        sigmas = sigmas[: max(int(self.strength * len(sigmas)), 1)]
        print("prune index:", max(int(self.strength * len(sigmas)), 1))
        sigmas = np.flip(sigmas, (0,))
        print("sigmas after pruning: ", sigmas)
        return sigmas


class Txt2NoisyDiscretizationWrapper:
    """
    wraps a discretizer, and prunes the sigmas
    params:
        strength: float between 0.0 and 1.0. 0.0 means full sampling (all sigmas are returned)
    """

    def __init__(self, discretization, strength: float = 0.0, original_steps=None):
        self.discretization = discretization
        self.strength = strength
        self.original_steps = original_steps
        assert 0.0 <= self.strength <= 1.0

    def __call__(self, *args, **kwargs):
        # sigmas start large first, and decrease then
        sigmas = self.discretization(*args, **kwargs)
        print("sigmas after discretization, before pruning img2img: ", sigmas)
        sigmas = np.flip(sigmas, (0,))
        if self.original_steps is None:
            steps = len(sigmas)
        else:
            steps = self.original_steps + 1
        prune_index = max(min(int(self.strength * steps) - 1, steps - 1), 0)
        sigmas = sigmas[prune_index:]
        print("prune index:", prune_index)
        sigmas = np.flip(sigmas, (0,))
        print("sigmas after pruning: ", sigmas)
        return sigmas
