# reference to https://github.com/Stability-AI/generative-models

import numpy as np

import mindspore as ms
from mindspore import nn, ops


class DiagonalGaussianDistribution(nn.Cell):
    def __init__(self, deterministic=False):
        super(DiagonalGaussianDistribution, self).__init__()
        self.deterministic = deterministic
        self.pi = np.pi

    def get_mean_and_var(self, input):
        mean, logvar = input.chunk(2, axis=1)
        logvar = ops.clamp(logvar, -30.0, 20.0)
        std = ops.exp(0.5 * logvar)
        var = ops.exp(logvar)
        if self.deterministic:
            std = ops.zeros_like(mean)
            var = ops.zeros_like(mean)
        return mean, logvar, var, std

    @ms.jit  # FIXME: ops.randn causes 'dynamic shape' error in PyNative mode
    def sample(self, input):
        mean, _, _, std = self.get_mean_and_var(input)
        x = mean + std * ops.randn(*mean.shape)
        return x

    def kl(self, input, other=None):
        mean, logvar, var, _ = self.get_mean_and_var(input)

        if self.deterministic:
            return ops.zeros((1,), ms.float32)
        else:
            if other is None:
                return 0.5 * ops.sum(
                    ops.pow(mean, 2) + var - 1.0 - logvar,
                    dim=[1, 2, 3],
                )
            else:
                return 0.5 * ops.sum(
                    ops.pow(mean - other.mean, 2) / other.var + var / other.var - 1.0 - logvar + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, input, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return ops.zeros((1,), ms.float32)

        mean, logvar, var, _ = self.get_mean_and_var(input)
        logtwopi = ops.log(2.0 * self.pi)
        return 0.5 * ops.sum(
            logtwopi + logvar + ops.pow(sample - mean, 2) / var,
            dim=dims,
        )

    def mode(self, input):
        mean, _, _, _ = self.get_mean_and_var(input)
        return mean
