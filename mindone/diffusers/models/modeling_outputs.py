"""Adapted from https://github.com/huggingface/diffusers/tree/main/src/diffusers/models/modeling_outputs.py."""

from dataclasses import dataclass

import mindspore as ms

from ..utils import BaseOutput


@dataclass
class AutoencoderKLOutput(BaseOutput):
    """
    Output of AutoencoderKL encoding method.

    Args:
        latent (`ms.Tensor`):
            Encoded outputs of `Encoder` represented as the mean and logvar of `DiagonalGaussianDistribution`.
            `DiagonalGaussianDistribution` allows for sampling latents from the distribution.
    """

    latent: ms.Tensor


@dataclass
class Transformer2DModelOutput(BaseOutput):
    """
    The output of [`Transformer2DModel`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` or
        `(batch size, num_vector_embeds - 1, num_latent_pixels)` if [`Transformer2DModel`] is discrete):
            The hidden states output conditioned on the `encoder_hidden_states` input. If discrete, returns probability
            distributions for the unnoised latent pixels.
    """

    sample: "ms.Tensor"  # noqa: F821
