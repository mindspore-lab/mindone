# reference to https://github.com/Stability-AI/generative-models

from typing import Optional

import numpy as np

import mindspore as ms
from mindspore import Tensor, float16, mint, nn, ops

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # FIXME: python 3.7


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
        freqs = mint.exp(
            -mint.log(mint.ones(1, dtype=ms.float32) * max_period)
            * mint.arange(start=0, end=half, dtype=ms.float32)
            / half
        )
        args = timesteps[:, None].astype(ms.float32) * freqs[None]
        embedding = mint.cat((mint.cos(args), mint.sin(args)), dim=-1)
        if dim % 2:
            embedding = mint.cat((embedding, mint.zeros_like(embedding[:, :1])), dim=-1)
    else:
        embedding = mint.broadcast_to(timesteps[:, None], (-1, dim))
    return embedding.astype(dtype)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for n, p in module.parameters_and_names():
        ops.assign(p, ops.zeros_like(p))
    return module


class GroupNorm(nn.GroupNorm):
    """
    Convert temporal 5D tensors to 4D as MindSpore supports (N, C, H, W) input only
    """

    def construct(self, x: Tensor) -> Tensor:
        if x.ndim == 5:
            return super().construct(x.view(x.shape[0], x.shape[1], x.shape[2], -1)).view(x.shape)
        else:
            return super().construct(x)


def normalization(channels, eps=1e-5):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm(32, channels, eps)


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
        ).to_float(float16)
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


class AlphaBlender(nn.Cell):
    """
    Blend spatial and temporal network branches using a learnable alpha value:
    blended = spatial * alpha + temporal * (1 - alpha)

    Args:
        alpha: a blending coefficient between 0 and 1.
        merge_strategy: merge strategy to use for spatial and temporal blending.
                        Options: "fixed" - alpha remains constant, "learned" - alpha is learned during training,
                        "learned_with_images" - alpha is learned for video frames only during hybrid (images and videos)
                        training. Default: "learned".
    """

    def __init__(
        self,
        alpha: float,
        merge_strategy: Literal["fixed", "learned", "learned_with_images"] = "learned",
    ):
        super().__init__()
        self.merge_strategy = merge_strategy

        if self.merge_strategy == "fixed":
            self.mix_factor = ms.Tensor([alpha])
        elif self.merge_strategy in ["learned", "learned_with_images"]:
            self.mix_factor = ms.Parameter([alpha])
        else:
            raise ValueError(f"Unknown branch merge strategy {self.merge_strategy}")

    def _get_alpha(self, image_only_indicator: ms.Tensor, ndim: int) -> ms.Tensor:
        if self.merge_strategy == "fixed":
            alpha = self.mix_factor
        elif self.merge_strategy == "learned":
            alpha = ops.sigmoid(self.mix_factor)
        elif self.merge_strategy == "learned_with_images":
            alpha = ops.where(
                image_only_indicator.bool(),
                ops.ones((1, 1)),
                ops.expand_dims(ops.sigmoid(self.mix_factor), -1),
            )

            if ndim == 5:  # apply alpha on the frame axis
                alpha = alpha[:, None, :, None, None]
            elif ndim == 3:  # apply alpha on the batch x frame axis
                alpha = alpha.reshape(-1)[:, None, None]
            else:
                raise ValueError(f"Unexpected ndim {ndim}. Dimensions should be 3 or 5")

        else:
            raise NotImplementedError(f"Unknown branch merge strategy {self.merge_strategy}")
        return alpha

    def construct(
        self, x_spatial: ms.Tensor, x_temporal: ms.Tensor, image_only_indicator: Optional[ms.Tensor] = None
    ) -> ms.Tensor:
        alpha = self._get_alpha(image_only_indicator, x_spatial.ndim)
        x = alpha.to(x_spatial.dtype) * x_spatial + (1.0 - alpha).to(x_spatial.dtype) * x_temporal
        return x
