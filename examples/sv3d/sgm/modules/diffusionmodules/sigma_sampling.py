# reference to https://github.com/Stability-AI/generative-models

from sgm.util import instantiate_from_config

import mindspore as ms
from mindspore import Tensor, nn, ops


class EDMSampling(nn.Cell):
    def __init__(self, p_mean=-1.2, p_std=1.2):
        super(EDMSampling, self).__init__()
        self.p_mean = p_mean
        self.p_std = p_std

    @ms.jit
    def construct(self, n_samples, rand=None):
        rand = rand if rand else ops.randn((n_samples,))
        log_sigma = self.p_mean + self.p_std * rand
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
