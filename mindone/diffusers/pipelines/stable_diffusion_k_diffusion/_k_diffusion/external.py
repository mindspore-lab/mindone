# Copyright (c) 2022 Katherine Crowson
# Licensed under the MIT License.
# Source: https://github.com/crowsonkb/k-diffusion

# Adapted from https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/external.py to work with MindSpore.
# This file is part of a derivative work and is also licensed under the MIT License.

import mindspore as ms
from mindspore import mint, nn

from . import sampling


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


class DiscreteSchedule(nn.Cell):
    """A mapping between continuous noise levels (sigmas) and a list of discrete noise
    levels."""

    def __init__(self, sigmas, quantize):
        super().__init__()
        self.sigmas = ms.Parameter(sigmas, name="sigmas", requires_grad=False)
        self.log_sigmas = ms.Parameter(sigmas.log(), name="log_sigmas", requires_grad=False)
        self.quantize = quantize

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def get_sigmas(self, n=None):
        if n is None:
            return sampling.append_zero(self.sigmas.flip([0]))
        t_max = len(self.sigmas) - 1
        t = mint.linspace(t_max, 0, n)
        return sampling.append_zero(self.t_to_sigma(t))

    def sigma_to_t(self, sigma, quantize=None):
        quantize = self.quantize if quantize is None else quantize
        log_sigma = sigma.log()
        dists = log_sigma - self.log_sigmas[:, None]
        if quantize:
            return dists.abs().argmin(dim=0).view(sigma.shape)
        low_idx = dists.ge(0).cumsum(dim=0).argmax(dim=0).clamp(max=self.log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1
        low, high = self.log_sigmas[low_idx], self.log_sigmas[high_idx]
        w = (low - log_sigma) / (low - high)
        w = w.clamp(0, 1)
        t = (1 - w) * low_idx + w * high_idx
        return t.view(sigma.shape)

    def t_to_sigma(self, t):
        t = t.float()
        low_idx, high_idx, w = t.floor().long(), t.ceil().long(), t.frac()
        log_sigma = (1 - w) * self.log_sigmas[low_idx] + w * self.log_sigmas[high_idx]
        return log_sigma.exp()


class DiscreteEpsDDPMDenoiser(DiscreteSchedule):
    """A wrapper for discrete schedule DDPM models that output eps (the predicted
    noise)."""

    def __init__(self, model, alphas_cumprod, quantize):
        super().__init__(((1 - alphas_cumprod) / alphas_cumprod) ** 0.5, quantize)
        self.inner_model = model
        self.sigma_data = 1.0

    def get_scalings(self, sigma):
        c_out = -sigma
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_out, c_in

    def get_eps(self, *args, **kwargs):
        return self.inner_model(*args, **kwargs)

    def loss(self, input, noise, sigma, **kwargs):
        c_out, c_in = [append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        noised_input = input + noise * append_dims(sigma, input.ndim)
        eps = self.get_eps(noised_input * c_in, self.sigma_to_t(sigma), **kwargs)
        return (eps - noise).pow(2).flatten(start_dim=1).mean(1)

    def construct(self, input, sigma, **kwargs):
        c_out, c_in = [append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        eps = self.get_eps(input * c_in, self.sigma_to_t(sigma), **kwargs)
        return input + eps * c_out


class CompVisDenoiser(DiscreteEpsDDPMDenoiser):
    """A wrapper for CompVis diffusion models."""

    def __init__(self, model, quantize=False):
        super().__init__(model, model.alphas_cumprod, quantize=quantize)

    def get_eps(self, *args, **kwargs):
        return self.inner_model.apply_model(*args, **kwargs)


class DiscreteVDDPMDenoiser(DiscreteSchedule):
    """A wrapper for discrete schedule DDPM models that output v."""

    def __init__(self, model, alphas_cumprod, quantize):
        super().__init__(((1 - alphas_cumprod) / alphas_cumprod) ** 0.5, quantize)
        self.inner_model = model
        self.sigma_data = 1.0

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = -sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_v(self, *args, **kwargs):
        return self.inner_model(*args, **kwargs)

    def loss(self, input, noise, sigma, **kwargs):
        c_skip, c_out, c_in = [append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        noised_input = input + noise * append_dims(sigma, input.ndim)
        model_output = self.get_v(noised_input * c_in, self.sigma_to_t(sigma), **kwargs)
        target = (input - c_skip * noised_input) / c_out
        return (model_output - target).pow(2).flatten(start_dim=1).mean(1)

    def construct(self, input, sigma, **kwargs):
        c_skip, c_out, c_in = [append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        return self.get_v(input * c_in, self.sigma_to_t(sigma), **kwargs) * c_out + input * c_skip


class CompVisVDenoiser(DiscreteVDDPMDenoiser):
    """A wrapper for CompVis diffusion models that output v."""

    def __init__(self, model, quantize=False):
        super().__init__(model, model.alphas_cumprod, quantize=quantize)

    def get_v(self, x, t, cond, **kwargs):
        return self.inner_model.apply_model(x, t, cond)
