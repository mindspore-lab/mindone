import numpy as np

import mindspore as ms
from mindspore import mint

from .....utils._brownian import BrownianInterval


def append_zero(x):
    return mint.cat([x, x.new_zeros([1])])


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = mint.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas)


class BatchedBrownianTree:
    """A wrapper around torchsde.BrownianTree that enables batches of entropy."""

    def __init__(self, x, t0, t1, seed=None, **kwargs):
        t0, t1, self.sign = self.sort(t0, t1)
        w0 = kwargs.get("w0", mint.zeros_like(x))
        if seed is None:
            seed = np.random.randint(0, 2**63 - 1, ()).item()
        self.batched = True
        try:
            assert len(seed) == x.shape[0]
            w0 = w0[0]
        except TypeError:
            seed = [seed]
            self.batched = False
        self.trees = [
            BrownianInterval(
                t0=t0,
                t1=t1,
                size=w0.shape,
                dtype=w0.dtype,
                entropy=s,
                tol=1e-6,
                pool_size=24,
                halfway_tree=True,
            )
            for s in seed
        ]

    @staticmethod
    def sort(a, b):
        return (a, b, 1) if a < b else (b, a, -1)

    def __call__(self, t0, t1):
        t0, t1, sign = self.sort(t0, t1)
        w = mint.stack([tree(t0, t1) for tree in self.trees]) * (self.sign * sign)
        return w if self.batched else w[0]


class BrownianTreeNoiseSampler:
    """A noise sampler backed by a torchsde.BrownianTree.

    Args:
        x (Tensor): The tensor whose shape, device and dtype to use to generate
            random samples.
        sigma_min (float): The low end of the valid interval.
        sigma_max (float): The high end of the valid interval.
        seed (int or List[int]): The random seed. If a list of seeds is
            supplied instead of a single integer, then the noise sampler will use one BrownianTree per batch item, each
            with its own seed.
        transform (callable): A function that maps sigma to the sampler's
            internal timestep.
    """

    def __init__(self, x, sigma_min, sigma_max, seed=None, transform=lambda x: x):
        self.transform = transform
        t0, t1 = self.transform(ms.Tensor(sigma_min)), self.transform(ms.Tensor(sigma_max))
        self.tree = BatchedBrownianTree(x, t0, t1, seed)

    def __call__(self, sigma, sigma_next):
        t0, t1 = self.transform(ms.Tensor(sigma)), self.transform(ms.Tensor(sigma_next))
        return self.tree(t0, t1) / (t1 - t0).abs().sqrt()
