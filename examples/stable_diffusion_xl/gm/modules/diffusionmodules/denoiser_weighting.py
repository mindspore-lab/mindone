# reference to https://github.com/Stability-AI/generative-models

from mindspore import nn, ops


class UnitWeighting(nn.Cell):
    def construct(self, sigma):
        return ops.ones_like(sigma)


class EDMWeighting(nn.Cell):
    def __init__(self, sigma_data=0.5):
        super(EDMWeighting, self).__init__()
        self.sigma_data = sigma_data

    def construct(self, sigma):
        return (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2


class VWeighting(EDMWeighting):
    def __init__(self):
        super().__init__(sigma_data=1.0)


class EpsWeighting(nn.Cell):
    def construct(self, sigma):
        return sigma**-2.0
