# reference to https://github.com/Stability-AI/generative-models

from mindspore import ops


class UnitWeighting:
    def __call__(self, sigma):
        return ops.ones_like(sigma)


class EDMWeighting:
    def __init__(self, sigma_data=0.5):
        self.sigma_data = sigma_data

    def __call__(self, sigma):
        return (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2


class VWeighting(EDMWeighting):
    def __init__(self):
        super().__init__(sigma_data=1.0)


class EpsWeighting:
    def __call__(self, sigma):
        return sigma**-2.0
