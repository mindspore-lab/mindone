#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import mindspore.nn as nn
import mindspore.ops as ops

from ..sampler import sampling, utils


class DiscreteSchedule(nn.Cell):
    """A mapping between continuous noise levels (sigmas) and a list of discrete noise
    levels."""

    def __init__(self, sigmas, quantize):
        super().__init__()
        self.register_buffer("sigmas", sigmas)
        self.register_buffer("log_sigmas", sigmas.log())
        self.quantize = quantize

    def register_buffer(self, name, attr):
        setattr(self, name, attr)

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def get_sigmas(self, n=None):
        if n is None:
            return sampling.append_zero(self.sigmas.flip(0))
        t_max = len(self.sigmas) - 1
        t = ops.linspace(t_max, 0, n)
        return sampling.append_zero(self.t_to_sigma(t))

    def sigma_to_t(self, sigma, quantize=None):
        quantize = self.quantize if quantize is None else quantize
        log_sigma = sigma.log()
        dists = log_sigma - self.log_sigmas[:, None]
        if quantize:
            return dists.abs().argmin(axis=0).view(sigma.shape)
        low_idx = dists.ge(0).cumsum(axis=0).argmax(axis=0).clamp(max=self.log_sigmas.shape[0] - 2)
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
        c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        noised_input = input + noise * utils.append_dims(sigma, input.ndim)
        eps = self.get_eps(noised_input * c_in, self.sigma_to_t(sigma), **kwargs)
        return (eps - noise).pow(2).flatten(1).mean(1)

    def construct(self, input, sigma, **kwargs):
        c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
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
        c_skip, c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        noised_input = input + noise * utils.append_dims(sigma, input.ndim)
        model_output = self.get_v(noised_input * c_in, self.sigma_to_t(sigma), **kwargs)
        target = (input - c_skip * noised_input) / c_out
        return (model_output - target).pow(2).flatten(1).mean(1)

    def construct(self, input, sigma, **kwargs):
        c_skip, c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        return self.get_v(input * c_in, self.sigma_to_t(sigma), **kwargs) * c_out + input * c_skip


class CompVisVDenoiser(DiscreteVDDPMDenoiser):
    """A wrapper for CompVis diffusion models that output v."""

    def __init__(self, model, quantize=False):
        super().__init__(model, model.alphas_cumprod, quantize=quantize)

    def get_v(self, x, t, cond, **kwargs):
        return self.inner_model.apply_model(x, t, cond)
