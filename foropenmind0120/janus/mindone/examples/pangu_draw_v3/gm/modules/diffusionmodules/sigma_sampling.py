# reference to https://github.com/Stability-AI/generative-models

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
    def __init__(self, discretization_config, num_idx, do_append_zero=False, flip=True, min_idx=-1, max_idx=-1):
        super(DiscreteSampling, self).__init__()
        self.num_idx = num_idx
        self.sigmas = Tensor(
            instantiate_from_config(discretization_config)(num_idx, do_append_zero=do_append_zero, flip=flip),
            ms.float32,
        )
        self.min_idx = min_idx
        self.max_idx = max_idx

    def idx_to_sigma(self, idx):
        return self.sigmas[idx]

    def construct(self, n_samples, rand=None):
        if rand is not None:
            idx = rand
        elif self.min_idx >= 0 and self.max_idx >= 0:
            idx = ops.randint(self.min_idx, self.max_idx, (n_samples,))
        else:
            idx = ops.randint(0, self.num_idx, (n_samples,))
        return self.idx_to_sigma(idx)
