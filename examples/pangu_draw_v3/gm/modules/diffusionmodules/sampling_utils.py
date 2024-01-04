# reference to https://github.com/Stability-AI/generative-models

from gm.util import append_dims

from mindspore import ops


class NoDynamicThresholding:
    def __call__(self, uncond, cond, scale):
        return uncond + scale * (cond - uncond)


class PanGuNoDynamicThresholding:
    def __call__(self, x_list, scale, other_scale):
        ret = x_list[0] + scale * (x_list[1] - x_list[0])
        for x, s in zip(x_list[2:], other_scale):
            ret += s * (x - x_list[1])
        return ret


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
