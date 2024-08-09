# reference to https://github.com/Stability-AI/generative-models

from sgm.util import append_dims, instantiate_from_config

import mindspore as ms
from mindspore import Tensor, nn, ops


class Denoiser(nn.Cell):
    def __init__(self, weighting_config, scaling_config):
        super(Denoiser, self).__init__()
        self.weighting = instantiate_from_config(weighting_config) if weighting_config is not None else None
        self.scaling = instantiate_from_config(scaling_config)

    def possibly_quantize_sigma(self, sigma):
        return sigma

    def possibly_quantize_c_noise(self, c_noise):
        return c_noise

    def w(self, sigma):
        return self.weighting(sigma)

    def construct(self, sigma, input_dim):
        sigma = ops.cast(sigma, ms.float32)
        sigma = self.possibly_quantize_sigma(sigma)
        sigma_shape = sigma.shape
        sigma = append_dims(sigma, input_dim)
        c_skip, c_out, c_in, c_noise = self.scaling(sigma)
        c_noise = self.possibly_quantize_c_noise(c_noise.reshape(sigma_shape))

        return c_skip, c_out, c_in, c_noise


class DiscreteDenoiser(Denoiser):
    def __init__(
        self,
        weighting_config,
        scaling_config,
        num_idx,
        discretization_config,
        do_append_zero=False,
        quantize_c_noise=True,
        flip=True,
    ):
        super().__init__(weighting_config, scaling_config)
        sigmas = instantiate_from_config(discretization_config)(num_idx, do_append_zero=do_append_zero, flip=flip)
        self.sigmas = Tensor(sigmas, ms.float32)
        self.quantize_c_noise = quantize_c_noise

    def sigma_to_idx(self, sigma):
        dists = sigma - self.sigmas[:, None]
        return ops.abs(dists).argmin(axis=0).reshape(sigma.shape)
        # return np.abs(dists).argmin(axis=0).reshape(sigma.shape)

    def idx_to_sigma(self, idx):
        return self.sigmas[idx]

    def possibly_quantize_sigma(self, sigma):
        return self.idx_to_sigma(self.sigma_to_idx(sigma))

    def possibly_quantize_c_noise(self, c_noise):
        if self.quantize_c_noise:
            return self.sigma_to_idx(c_noise)
        else:
            return c_noise
