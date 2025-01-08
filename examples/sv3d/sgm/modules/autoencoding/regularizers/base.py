from typing import Tuple

from sgm.modules.distributions.distributions import DiagonalGaussianDistribution

from mindspore import Tensor, nn, ops


class IdentityRegularizer(nn.Cell):
    def construct(self, z: Tensor) -> Tuple[Tensor, dict]:
        return z, dict()


class DiagonalGaussianRegularizer(nn.Cell):
    def __init__(self, sample: bool = True):
        super().__init__()
        self.sample = sample
        self.posterior = DiagonalGaussianDistribution()

    def construct(self, z: Tensor) -> Tuple[Tensor, dict]:
        z = self.posterior.sample(z) if self.sample else self.posterior.mode(z)
        kl_loss = self.posterior.kl(z)
        kl_loss = ops.sum(kl_loss) / kl_loss.shape[0]
        return z, {"kl_loss": kl_loss}
