# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import mindspore as ms
from mindspore import ops, nn, Parameter
import mindspore.numpy as mnp

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import BaseOutput

from ...utils.mindspore_utils import randn_tensor
from ..modeling_utils import ModelMixin

def norm_except_dim(v, pow, dim):
    if dim == -1:
        return mnp.norm(v, pow)
    elif dim == 0:
        output_size = (v.shape[0],) + (1,) * (v.ndim - 1)
        return mnp.norm(v.view((v.shape[0], -1)), pow, 1).view(output_size)
    elif dim == (v.ndim - 1):
        output_size = (1,) * (v.ndim - 1) + (v.shape[v.ndim - 1])
        return mnp.norm(v.view((-1, v.shape[v.ndim - 1])), pow, 0).view(output_size)
    else:
        return norm_except_dim(v.swapaxes(0, dim), pow, dim).swapaxes(0, dim)


def _weight_norm(v, g, dim):
    return v * (g / norm_except_dim(v, 2, dim))


class WeightNorm(nn.Cell):
    def __init__(self, module, dim: int = 0):
        super().__init__()

        if dim is None:
            dim = -1

        self.dim = dim
        self.module = module

        self.assign = ops.Assign()
        # add g and v as new parameters and express w as g/||v|| * v
        self.param_g = Parameter(ms.Tensor(norm_except_dim(self.module.weight, 2, dim)))
        self.param_v = Parameter(ms.Tensor(self.module.weight.data))
        self.module.weight.set_data(_weight_norm(self.param_v, self.param_g, self.dim))

        self.use_weight_norm = True
        self.kernel_size = module.kernel_size
        self.stride = module.stride
        self.dilation = module.dilation

    def construct(self, *inputs, **kwargs):
        if not self.use_weight_norm:
            return self.module(*inputs, **kwargs)

        self.assign(self.module.weight, _weight_norm(self.param_v, self.param_g, self.dim))
        return self.module(*inputs, **kwargs)

    def remove_weight_norm(self):
        self.assign(self.module.weight, _weight_norm(self.param_v, self.param_g, self.dim))
        self.use_weight_norm = False


class Snake1d(nn.Cell):
    """
    A 1-dimensional Snake activation function module.
    """

    def __init__(self, hidden_dim, logscale=True):
        super().__init__()
        self.alpha = nn.Parameter(ops.zeros(1, hidden_dim, 1))
        self.beta = nn.Parameter(ops.zeros(1, hidden_dim, 1))

        self.alpha.requires_grad = True
        self.beta.requires_grad = True
        self.logscale = logscale

    def construct(self, hidden_states):
        shape = hidden_states.shape

        alpha = self.alpha if not self.logscale else ops.exp(self.alpha)
        beta = self.beta if not self.logscale else ops.exp(self.beta)

        hidden_states = hidden_states.reshape(shape[0], shape[1], -1)
        hidden_states = hidden_states + (beta + 1e-9).reciprocal() * ops.sin(alpha * hidden_states).pow(2)
        hidden_states = hidden_states.reshape(shape)
        return hidden_states


class OobleckResidualUnit(nn.Cell):
    """
    A residual unit composed of Snake1d and weight-normalized Conv1d layers with dilations.
    """

    def __init__(self, dimension: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2

        self.snake1 = Snake1d(dimension)
        self.conv1 = WeightNorm(nn.Conv1d(dimension, dimension, kernel_size=7, dilation=dilation, padding=pad))
        self.snake2 = Snake1d(dimension)
        self.conv2 = WeightNorm(nn.Conv1d(dimension, dimension, kernel_size=1))

    def construct(self, hidden_state):
        """
        Forward pass through the residual unit.

        Args:
            hidden_state (`ms.Tensor` of shape `(batch_size, channels, time_steps)`):
                Input tensor .

        Returns:
            output_tensor (`ms.Tensor` of shape `(batch_size, channels, time_steps)`)
                Input tensor after passing through the residual unit.
        """
        output_tensor = hidden_state
        output_tensor = self.conv1(self.snake1(output_tensor))
        output_tensor = self.conv2(self.snake2(output_tensor))

        padding = (hidden_state.shape[-1] - output_tensor.shape[-1]) // 2
        if padding > 0:
            hidden_state = hidden_state[..., padding:-padding]
        output_tensor = hidden_state + output_tensor
        return output_tensor


class OobleckEncoderBlock(nn.Cell):
    """Encoder block used in Oobleck encoder."""

    def __init__(self, input_dim, output_dim, stride: int = 1):
        super().__init__()

        self.res_unit1 = OobleckResidualUnit(input_dim, dilation=1)
        self.res_unit2 = OobleckResidualUnit(input_dim, dilation=3)
        self.res_unit3 = OobleckResidualUnit(input_dim, dilation=9)
        self.snake1 = Snake1d(input_dim)
        self.conv1 = WeightNorm(
            nn.Conv1d(input_dim, output_dim, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2))
        )

    def construct(self, hidden_state):
        hidden_state = self.res_unit1(hidden_state)
        hidden_state = self.res_unit2(hidden_state)
        hidden_state = self.snake1(self.res_unit3(hidden_state))
        hidden_state = self.conv1(hidden_state)

        return hidden_state


class OobleckDecoderBlock(nn.Cell):
    """Decoder block used in Oobleck decoder."""

    def __init__(self, input_dim, output_dim, stride: int = 1):
        super().__init__()

        self.snake1 = Snake1d(input_dim)
        self.conv_t1 = WeightNorm(
            nn.ConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            )
        )
        self.res_unit1 = OobleckResidualUnit(output_dim, dilation=1)
        self.res_unit2 = OobleckResidualUnit(output_dim, dilation=3)
        self.res_unit3 = OobleckResidualUnit(output_dim, dilation=9)

    def construct(self, hidden_state):
        hidden_state = self.snake1(hidden_state)
        hidden_state = self.conv_t1(hidden_state)
        hidden_state = self.res_unit1(hidden_state)
        hidden_state = self.res_unit2(hidden_state)
        hidden_state = self.res_unit3(hidden_state)

        return hidden_state


class OobleckDiagonalGaussianDistribution(object):
    def __init__(self, parameters: ms.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.scale = parameters.chunk(2, dim=1)
        self.std = nn.functional.softplus(self.scale) + 1e-4
        self.var = self.std * self.std
        self.logvar = ops.log(self.var)
        self.deterministic = deterministic

    def sample(self, generator: Optional[np.random.Generator] = None) -> ms.Tensor:
        # make sure sample is on the same device as the parameters and has same dtype
        sample = randn_tensor(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        x = self.mean + self.std * sample
        return x

    def kl(self, other: "OobleckDiagonalGaussianDistribution" = None) -> ms.Tensor:
        if self.deterministic:
            return ms.Tensor([0.0])
        else:
            if other is None:
                return (self.mean * self.mean + self.var - self.logvar - 1.0).sum(1).mean()
            else:
                normalized_diff = ops.pow(self.mean - other.mean, 2) / other.var
                var_ratio = self.var / other.var
                logvar_diff = self.logvar - other.logvar

                kl = normalized_diff + var_ratio + logvar_diff - 1

                kl = kl.sum(1).mean()
                return kl

    def mode(self) -> ms.Tensor:
        return self.mean


@dataclass
class AutoencoderOobleckOutput(BaseOutput):
    """
    Output of AutoencoderOobleck encoding method.

    Args:
        latent_dist (`OobleckDiagonalGaussianDistribution`):
            Encoded outputs of `Encoder` represented as the mean and standard deviation of
            `OobleckDiagonalGaussianDistribution`. `OobleckDiagonalGaussianDistribution` allows for sampling latents
            from the distribution.
    """

    latent_dist: "OobleckDiagonalGaussianDistribution"  # noqa: F821


@dataclass
class OobleckDecoderOutput(BaseOutput):
    r"""
    Output of decoding method.

    Args:
        sample (`ms.Tensor` of shape `(batch_size, audio_channels, sequence_length)`):
            The decoded output sample from the last layer of the model.
    """

    sample: ms.Tensor


class OobleckEncoder(nn.Cell):
    """Oobleck Encoder"""

    def __init__(self, encoder_hidden_size, audio_channels, downsampling_ratios, channel_multiples):
        super().__init__()

        strides = downsampling_ratios
        channel_multiples = [1] + channel_multiples

        # Create first convolution
        self.conv1 = WeightNorm(nn.Conv1d(audio_channels, encoder_hidden_size, kernel_size=7, pad_mode='pad', padding=3))

        self.block = []
        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride_index, stride in enumerate(strides):
            self.block += [
                OobleckEncoderBlock(
                    input_dim=encoder_hidden_size * channel_multiples[stride_index],
                    output_dim=encoder_hidden_size * channel_multiples[stride_index + 1],
                    stride=stride,
                )
            ]

        self.block = nn.CellList(self.block)
        d_model = encoder_hidden_size * channel_multiples[-1]
        self.snake1 = Snake1d(d_model)
        self.conv2 = WeightNorm(nn.Conv1d(d_model, encoder_hidden_size, kernel_size=3, padding=1))

    def construct(self, hidden_state):
        hidden_state = self.conv1(hidden_state)

        for module in self.block:
            hidden_state = module(hidden_state)

        hidden_state = self.snake1(hidden_state)
        hidden_state = self.conv2(hidden_state)

        return hidden_state


class OobleckDecoder(nn.Cell):
    """Oobleck Decoder"""

    def __init__(self, channels, input_channels, audio_channels, upsampling_ratios, channel_multiples):
        super().__init__()

        strides = upsampling_ratios
        channel_multiples = [1] + channel_multiples

        # Add first conv layer
        self.conv1 = WeightNorm(nn.Conv1d(input_channels, channels * channel_multiples[-1], kernel_size=7, padding=3))

        # Add upsampling + MRF blocks
        block = []
        for stride_index, stride in enumerate(strides):
            block += [
                OobleckDecoderBlock(
                    input_dim=channels * channel_multiples[len(strides) - stride_index],
                    output_dim=channels * channel_multiples[len(strides) - stride_index - 1],
                    stride=stride,
                )
            ]

        self.block = nn.CellList(block)
        output_dim = channels
        self.snake1 = Snake1d(output_dim)
        self.conv2 = WeightNorm(nn.Conv1d(channels, audio_channels, kernel_size=7, padding=3, bias=False))

    def construct(self, hidden_state):
        hidden_state = self.conv1(hidden_state)

        for layer in self.block:
            hidden_state = layer(hidden_state)

        hidden_state = self.snake1(hidden_state)
        hidden_state = self.conv2(hidden_state)

        return hidden_state


class AutoencoderOobleck(ModelMixin, ConfigMixin):
    r"""
    An autoencoder for encoding waveforms into latents and decoding latent representations into waveforms. First
    introduced in Stable Audio.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        encoder_hidden_size (`int`, *optional*, defaults to 128):
            Intermediate representation dimension for the encoder.
        downsampling_ratios (`List[int]`, *optional*, defaults to `[2, 4, 4, 8, 8]`):
            Ratios for downsampling in the encoder. These are used in reverse order for upsampling in the decoder.
        channel_multiples (`List[int]`, *optional*, defaults to `[1, 2, 4, 8, 16]`):
            Multiples used to determine the hidden sizes of the hidden layers.
        decoder_channels (`int`, *optional*, defaults to 128):
            Intermediate representation dimension for the decoder.
        decoder_input_channels (`int`, *optional*, defaults to 64):
            Input dimension for the decoder. Corresponds to the latent dimension.
        audio_channels (`int`, *optional*, defaults to 2):
            Number of channels in the audio data. Either 1 for mono or 2 for stereo.
        sampling_rate (`int`, *optional*, defaults to 44100):
            The sampling rate at which the audio waveform should be digitalized expressed in hertz (Hz).
    """

    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(
        self,
        encoder_hidden_size=128,
        downsampling_ratios=[2, 4, 4, 8, 8],
        channel_multiples=[1, 2, 4, 8, 16],
        decoder_channels=128,
        decoder_input_channels=64,
        audio_channels=2,
        sampling_rate=44100,
    ):
        super().__init__()

        self.encoder_hidden_size = encoder_hidden_size
        self.downsampling_ratios = downsampling_ratios
        self.decoder_channels = decoder_channels
        self.upsampling_ratios = downsampling_ratios[::-1]
        self.hop_length = int(np.prod(downsampling_ratios))
        self.sampling_rate = sampling_rate

        self.encoder = OobleckEncoder(
            encoder_hidden_size=encoder_hidden_size,
            audio_channels=audio_channels,
            downsampling_ratios=downsampling_ratios,
            channel_multiples=channel_multiples,
        )

        self.decoder = OobleckDecoder(
            channels=decoder_channels,
            input_channels=decoder_input_channels,
            audio_channels=audio_channels,
            upsampling_ratios=self.upsampling_ratios,
            channel_multiples=channel_multiples,
        )

        self.use_slicing = False

    def enable_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.use_slicing = True

    def disable_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_slicing` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_slicing = False

    def encode(
        self, x: ms.Tensor, return_dict: bool = True
    ) -> Union[AutoencoderOobleckOutput, Tuple[OobleckDiagonalGaussianDistribution]]:
        """
        Encode a batch of images into latents.

        Args:
            x (`ms.Tensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
                The latent representations of the encoded images. If `return_dict` is True, a
                [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain `tuple` is returned.
        """
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self.encoder(x_slice) for x_slice in x.split(1)]
            h = ops.cat(encoded_slices)
        else:
            h = self.encoder(x)

        posterior = OobleckDiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)

        return AutoencoderOobleckOutput(latent_dist=posterior)

    def _decode(self, z: ms.Tensor, return_dict: bool = True) -> Union[OobleckDecoderOutput, ms.Tensor]:
        dec = self.decoder(z)

        if not return_dict:
            return (dec,)

        return OobleckDecoderOutput(sample=dec)

    def decode(
        self, z: ms.Tensor, return_dict: bool = True, generator=None
    ) -> Union[OobleckDecoderOutput, ms.Tensor]:
        """
        Decode a batch of images.

        Args:
            z (`ms.Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.vae.OobleckDecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.OobleckDecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.OobleckDecoderOutput`] is returned, otherwise a plain `tuple`
                is returned.

        """
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            decoded = ops.cat(decoded_slices)
        else:
            decoded = self._decode(z).sample

        if not return_dict:
            return (decoded,)

        return OobleckDecoderOutput(sample=decoded)

    def construct(
        self,
        sample: ms.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[np.random.Generator] = None,
    ) -> Union[OobleckDecoderOutput, ms.Tensor]:
        r"""
        Args:
            sample (`ms.Tensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`OobleckDecoderOutput`] instead of a plain tuple.
        """
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z).sample

        if not return_dict:
            return (dec,)

        return OobleckDecoderOutput(sample=dec)