# reference to https://github.com/Stability-AI/generative-models
from typing import Tuple

from mindspore import Tensor, nn, ops


class EDMScaling(nn.Cell):
    def __init__(self, sigma_data=0.5):
        super(EDMScaling, self).__init__()
        self.sigma_data = sigma_data

    def construct(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        c_noise = 0.25 * ops.log(sigma)
        return c_skip, c_out, c_in, c_noise


class EpsScaling(nn.Cell):
    def construct(self, sigma):
        c_skip = ops.ones_like(sigma)
        c_out = -sigma
        c_in = 1 / (sigma**2 + 1.0) ** 0.5
        c_noise = sigma.copy()
        return c_skip, c_out, c_in, c_noise


class EpsScaling2(nn.Cell):
    def construct(self, sigma):
        c_skip = ops.ones_like(sigma)
        c_out = -sigma
        c_in = ops.ones_like(sigma)  # c_in = 1 / (sigma**2 + 1.0) ** 0.5
        c_noise = sigma.copy()
        return c_skip, c_out, c_in, c_noise


class VScaling(nn.Cell):
    def construct(self, sigma):
        c_skip = 1.0 / (sigma**2 + 1.0)
        c_out = -sigma / (sigma**2 + 1.0) ** 0.5
        c_in = 1.0 / (sigma**2 + 1.0) ** 0.5
        c_noise = sigma.copy()
        return c_skip, c_out, c_in, c_noise


class VScalingWithEDMcNoise(nn.Cell):
    def construct(self, sigma: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        c_skip = 1.0 / (sigma**2 + 1.0)
        c_out = -sigma / (sigma**2 + 1.0) ** 0.5
        c_in = 1.0 / (sigma**2 + 1.0) ** 0.5
        c_noise = 0.25 * sigma.log()
        return c_skip, c_out, c_in, c_noise
