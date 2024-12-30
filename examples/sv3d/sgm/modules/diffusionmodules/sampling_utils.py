# reference to https://github.com/Stability-AI/generative-models

import numpy as np
from sgm.util import append_dims

import mindspore as ms
from mindspore import Tensor, ops


class NoDynamicThresholding:
    def __call__(self, x):
        return x


class DynamicThresholding:
    """
    "Dynamic thresholding: At each sampling step we set s to a certain percentile absolute pixel value in xt0 (the
    prediction of x_0 at timestep t), and if s > 1, then we threshold xt0 to the range [-s, s] and then divide by
    s. Dynamic thresholding pushes saturated pixels (those near -1 and 1) inwards, thereby actively preventing
    pixels from saturation at each step. We find that dynamic thresholding results in significantly better
    photorealism as well as better image-text alignment, especially when using very large guidance weights."

    https://arxiv.org/abs/2205.11487
    """

    def __init__(self, dynamic_thresholding_ratio=0.995, sample_max_value=1.0):
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.sample_max_value = sample_max_value

    def __call__(self, x):
        batch_size, channels, *remaining_dims = x.shape
        # Flatten sample for doing quantile calculation along each image
        x = x.reshape(batch_size, channels * np.prod(remaining_dims))
        # "a certain percentile absolute pixel value"
        abs_x = x.abs()
        s = Tensor(np.quantile(abs_x.asnumpy(), self.dynamic_thresholding_ratio, axis=1), ms.float32)
        # When clamped to min=1, equivalent to standard clipping to [-1, 1]
        s = ops.clamp(s, min=1, max=self.sample_max_value)
        s = s.unsqueeze(1)  # (batch_size, 1) because clamp will broadcast along dim=0
        x = ops.clamp(x, -s, s) / s
        x = x.reshape(batch_size, channels, *remaining_dims)

        return x


def to_d(x, sigma, denoised):
    return (x - denoised) / append_dims(sigma, x.ndim)


def to_neg_log_sigma(sigma):
    return sigma.log().neg()


def to_sigma(neg_log_sigma):
    return neg_log_sigma.neg().exp()


def get_ancestral_step(sigma_from, sigma_to, eta=1.0):
    if not eta:
        return sigma_to, 0.0
    sigma_up = ops.minimum(
        sigma_to,
        eta * (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5,
    )
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up
