# reference to https://github.com/Stability-AI/generative-models

from gm.util import append_dims


class NoDynamicThresholding:
    def __call__(self, uncond, cond, scale):
        return uncond + scale * (cond - uncond)


def to_d(x, sigma, denoised):
    return (x - denoised) / append_dims(sigma, x.ndim)
