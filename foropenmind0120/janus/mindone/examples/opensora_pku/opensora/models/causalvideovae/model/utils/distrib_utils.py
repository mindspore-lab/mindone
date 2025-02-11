import mindspore as ms
from mindspore import Tensor, mint


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.mean, self.logvar = mint.split(parameters, [parameters.shape[1] // 2, parameters.shape[1] // 2], dim=1)
        self.logvar = mint.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = mint.exp(0.5 * self.logvar)
        self.var = mint.exp(self.logvar)
        self.stdnormal = mint.normal
        if self.deterministic:
            self.var = self.std = mint.zeros_like(self.mean, dtype=self.mean.dtype)

    def sample(self):
        x = self.mean + self.std * self.stdnormal(size=self.mean.shape)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return Tensor([0.0])
        else:
            if other is None:
                return 0.5 * mint.sum(mint.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1, 2, 3])
            else:
                return 0.5 * mint.sum(
                    mint.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return Tensor([0.0])
        logtwopi = ms.numpy.log(2.0 * ms.numpy.pi)
        return 0.5 * mint.sum(logtwopi + self.logvar + mint.pow(sample - self.mean, 2) / self.var, dim=dims)

    def mode(self):
        return self.mean
