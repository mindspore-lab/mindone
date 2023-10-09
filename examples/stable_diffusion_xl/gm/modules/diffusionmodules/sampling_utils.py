# reference to https://github.com/Stability-AI/generative-models

from gm.util import append_dims


class NoDynamicThresholding:
    def __call__(self, uncond, cond, scale):
        return uncond + scale * (cond - uncond)


def to_d(x, sigma, denoised):
    return (x - denoised) / append_dims(sigma, x.ndim)


def to_neg_log_sigma(sigma):
    return sigma.log().neg()


def to_sigma(neg_log_sigma):
    return neg_log_sigma.neg().exp()
