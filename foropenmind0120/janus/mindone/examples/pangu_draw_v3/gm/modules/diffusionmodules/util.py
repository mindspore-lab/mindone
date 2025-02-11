# reference to https://github.com/Stability-AI/generative-models

import numpy as np

import mindspore as ms
from mindspore import nn, ops


class ZeroInitModule(nn.Cell):
    def __init__(self, module):
        super(ZeroInitModule, self).__init__(auto_prefix=False)
        self.module = module
        for n, p in self.parameters_and_names():
            ops.assign(p, ops.zeros_like(p))

    def construct(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)


def make_beta_schedule(
    schedule,
    n_timestep,
    linear_start=1e-4,
    linear_end=2e-2,
):
    if schedule == "linear":
        betas = np.linspace(linear_start**0.5, linear_end**0.5, n_timestep, dtype=np.float32) ** 2
    else:
        raise NotImplementedError

    return betas


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False, dtype=ms.float32):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = ops.exp(
            -ops.log(ops.ones(1, dtype=ms.float32) * max_period)
            * ops.arange(start=0, end=half, dtype=ms.float32)
            / half
        )
        args = timesteps[:, None].astype(ms.float32) * freqs[None]
        embedding = ops.concat((ops.cos(args), ops.sin(args)), axis=-1)
        if dim % 2:
            embedding = ops.concat((embedding, ops.zeros_like(embedding[:, :1])), axis=-1)
    else:
        embedding = ops.broadcast_to(timesteps[:, None], (-1, dim))
    return embedding.astype(dtype)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for n, p in module.parameters_and_names():
        ops.assign(p, ops.zeros_like(p))
    return module


def normalization(channels, eps=1e-5):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return nn.GroupNorm(32, channels, eps)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(
            *args, has_bias=kwargs.pop("has_bias", True), pad_mode=kwargs.pop("pad_mode", "valid"), **kwargs
        )
    elif dims == 2:
        return nn.Conv2d(
            *args, has_bias=kwargs.pop("has_bias", True), pad_mode=kwargs.pop("pad_mode", "valid"), **kwargs
        )
    elif dims == 3:
        return nn.Conv3d(
            *args, has_bias=kwargs.pop("has_bias", True), pad_mode=kwargs.pop("pad_mode", "valid"), **kwargs
        )
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Dense(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")
