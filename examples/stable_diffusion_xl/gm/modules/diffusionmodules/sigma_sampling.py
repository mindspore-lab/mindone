# reference to https://github.com/Stability-AI/generative-models
import numpy as np
from gm.util import default, instantiate_from_config

import mindspore as ms
from mindspore import Tensor, nn, ops


class EDMSampling(nn.Cell):
    def __init__(self, p_mean=-1.2, p_std=1.2):
        super(EDMSampling, self).__init__()
        self.p_mean = p_mean
        self.p_std = p_std

    def construct(self, n_samples, rand=None):
        log_sigma = self.p_mean + self.p_std * default(rand, ops.randn((n_samples,)))
        return log_sigma.exp()


class DiscreteSampling(nn.Cell):
    def __init__(self, discretization_config, num_idx, do_append_zero=False, flip=True):
        super(DiscreteSampling, self).__init__()
        self.num_idx = num_idx
        self.sigmas = Tensor(
            instantiate_from_config(discretization_config)(num_idx, do_append_zero=do_append_zero, flip=flip),
            ms.float32,
        )

    def idx_to_sigma(self, idx):
        return self.sigmas[idx]

    def construct(self, n_samples, rand=None):
        if rand is not None:
            idx = rand
        else:
            idx = ops.randint(0, self.num_idx, (n_samples,))
        return self.idx_to_sigma(idx)


class DiscreteSamplingForZeRO3(DiscreteSampling):
    def __init__(self, discretization_config, num_idx, do_append_zero=False, flip=True, quantize_c_noise=True):
        super(DiscreteSamplingForZeRO3, self).__init__(discretization_config, num_idx, do_append_zero, flip)
        sigmas_np = instantiate_from_config(discretization_config)(num_idx, do_append_zero=do_append_zero, flip=flip)
        self.sigmas_np = sigmas_np
        self.quantize_c_noise = quantize_c_noise
        c_skip, c_out, c_in, c_noise = self.c_compute(sigmas_np)
        self.c_skip = Tensor(c_skip)
        self.c_out = Tensor(c_out)
        self.c_in = Tensor(c_in)
        self.c_noise = Tensor(c_noise)

    def c_compute(self, sigmas_np):
        sigmas_np = sigmas_np.astype(np.float32)
        sigmas_np = self.possibly_quantize_sigma(sigmas_np)
        c_skip = np.ones_like(sigmas_np)
        c_out = -sigmas_np
        c_in = 1 / (sigmas_np**2 + 1.0) ** 0.5
        c_noise = sigmas_np.copy()
        c_noise = self.possibly_quantize_c_noise(c_noise)
        return c_skip, c_out, c_in, c_noise

    def sigma_to_idx(self, sigma):
        dists = sigma - self.sigmas_np[:, None]
        return np.abs(dists).argmin(axis=1)

    def idx_to_sigma(self, idx):
        return self.sigmas_np[idx]

    def possibly_quantize_sigma(self, sigma):
        return self.idx_to_sigma(self.sigma_to_idx(sigma))

    def possibly_quantize_c_noise(self, c_noise):
        if self.quantize_c_noise:
            return self.sigma_to_idx(c_noise)
        else:
            return c_noise
