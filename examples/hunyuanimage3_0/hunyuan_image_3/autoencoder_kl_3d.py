#
# This code is adapted from https://github.com/Tencent-Hunyuan/HunyuanImage-3.0
# with modifications to run diffusers on mindspore.
#
# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

import mindspore as ms
import mindspore.mint.nn.functional as F
from mindspore import mint, nn

from mindone.diffusers.configuration_utils import ConfigMixin, register_to_config
from mindone.diffusers.models.activations import get_activation
from mindone.diffusers.models.modeling_utils import ModelMixin
from mindone.diffusers.utils import BaseOutput
from mindone.diffusers.utils.mindspore_utils import randn_tensor
from mindone.transformers.mindspore_adapter import scaled_dot_product_attention


def forward_with_checkpointing(module, *inputs, use_checkpointing=False):
    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)

        return custom_forward

    if use_checkpointing:
        raise NotImplementedError
    else:
        return module(*inputs)


@dataclass
class AutoencoderKLOutput(BaseOutput):
    """
    Output of AutoencoderKL encoding method.

    Args:
        latent_dist (`DiagonalGaussianDistribution`):
            Encoded outputs of `Encoder` represented as the mean and logvar of `DiagonalGaussianDistribution`.
            `DiagonalGaussianDistribution` allows for sampling latents from the distribution.
    """

    latent_dist: "DiagonalGaussianDistribution"  # noqa: F821


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters: ms.Tensor, deterministic: bool = False):
        if parameters.ndim == 3:
            dim = 2  # (B, L, C)
        elif parameters.ndim == 5 or parameters.ndim == 4:
            dim = 1  # (B, C, T, H ,W) / (B, C, H, W)
        else:
            raise NotImplementedError
        self.parameters = parameters
        self.mean, self.logvar = mint.chunk(parameters, 2, dim=dim)
        self.logvar = mint.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = mint.exp(0.5 * self.logvar)
        self.var = mint.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = mint.zeros_like(self.mean, dtype=self.parameters.dtype)

    def sample(self, generator: Optional[np.random.Generator] = None) -> ms.Tensor:
        # make sure sample is on the same device as the parameters and has same dtype
        sample = randn_tensor(
            self.mean.shape,
            generator=generator,
            dtype=self.parameters.dtype,
        )
        x = self.mean + self.std * sample
        return x

    def kl(self, other: "DiagonalGaussianDistribution" = None) -> ms.Tensor:
        if self.deterministic:
            return ms.Tensor([0.0])
        else:
            reduce_dim = list(range(1, self.mean.ndim))
            if other is None:
                return 0.5 * mint.sum(
                    mint.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=reduce_dim,
                )
            else:
                return 0.5 * mint.sum(
                    mint.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=reduce_dim,
                )

    def nll(self, sample: ms.Tensor, dims: Tuple[int, ...] = [1, 2, 3]) -> ms.Tensor:
        if self.deterministic:
            return ms.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * mint.sum(
            logtwopi + self.logvar + mint.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self) -> ms.Tensor:
        return self.mean


@dataclass
class DecoderOutput(BaseOutput):
    r"""
    Output of decoding method.

    Args:
        sample (`ms.tensor` of shape `(batch_size, num_channels, height, width)`):
            The decoded output sample from the last layer of the model.
    """

    sample: ms.tensor
    posterior: Optional[DiagonalGaussianDistribution] = None


class Conv3d(mint.nn.Conv3d):
    """
    Perform Conv3d on patches with numerical differences from mint.nn.Conv3d within 1e-5.
    Only symmetric padding is supported.
    """

    def construct(self, input):
        B, C, T, H, W = input.shape
        memory_count = (C * T * H * W) * 2 / 1024**3
        if memory_count > 2:
            n_split = math.ceil(memory_count / 2)
            assert n_split >= 2
            chunks = mint.chunk(input, chunks=n_split, dim=-3)
            padded_chunks = []
            for i in range(len(chunks)):
                if self.padding[0] > 0:
                    padded_chunk = F.pad(
                        chunks[i].float(),
                        (0, 0, 0, 0, self.padding[0], self.padding[0]),
                        mode="constant" if self.padding_mode == "zeros" else self.padding_mode,
                        value=0,
                    ).to(chunks[i].dtype)
                    if i > 0:
                        padded_chunk[:, :, : self.padding[0]] = chunks[i - 1][:, :, -self.padding[0] :]
                    if i < len(chunks) - 1:
                        padded_chunk[:, :, -self.padding[0] :] = chunks[i + 1][:, :, : self.padding[0]]
                else:
                    padded_chunk = chunks[i]
                padded_chunks.append(padded_chunk)
            padding_bak = self.padding
            self.padding = (0, self.padding[1], self.padding[2])
            outputs = []
            for i in range(len(padded_chunks)):
                outputs.append(super().construct(padded_chunks[i]))
            self.padding = padding_bak
            return mint.cat(outputs, dim=-3)
        else:
            return super().construct(input)


class AttnBlock(nn.Cell):
    r"""
    Self-attention with a single head.

    Args:
        in_channels (int): The number of channels in the input tensor.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        self.q = Conv3d(in_channels, in_channels, kernel_size=1)
        self.k = Conv3d(in_channels, in_channels, kernel_size=1)
        self.v = Conv3d(in_channels, in_channels, kernel_size=1)
        self.proj_out = Conv3d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: ms.Tensor) -> ms.Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, f, h, w = q.shape
        q = q.permute(0, 2, 3, 4, 1).reshape(b, 1, f * h * w, c).contiguous()
        k = k.permute(0, 2, 3, 4, 1).reshape(b, 1, f * h * w, c).contiguous()
        v = v.permute(0, 2, 3, 4, 1).reshape(b, 1, f * h * w, c).contiguous()

        # apply attention
        h_ = scaled_dot_product_attention(q, k, v)
        h_ = h_.reshape(b, f, h, w, c).permute(0, 4, 1, 2, 3)

        return h_

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Cell):
    r"""
    Residual block with two convolutions and optional channel change.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        non_linearity (str, optional): Type of non-linearity to use. Default is "swish".
    """

    def __init__(self, in_channels: int, out_channels: int, non_linearity: str = "swish") -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nonlinearity = get_activation(non_linearity)

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if in_channels != out_channels:
            self.conv_shortcut = Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.conv_shortcut = None

    def construct(self, x):
        # Apply shortcut connection
        residual = x

        # First normalization and activation
        x = self.norm1(x)
        x = self.nonlinearity(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.nonlinearity(x)
        x = self.conv2(x)

        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        # Add residual connection
        return x + residual


class Downsample(nn.Cell):
    """
    Downsampling block for spatial reduction.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels: int, add_temporal_downsample: bool = True):
        super().__init__()
        self.add_temporal_downsample = add_temporal_downsample
        stride = (2, 2, 2) if add_temporal_downsample else (1, 2, 2)  # THW
        # no asymmetric padding in torch conv, must do it ourselves
        self.conv = Conv3d(in_channels, in_channels, kernel_size=3, stride=stride, padding=0)

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        spatial_pad = (0, 1, 0, 1, 0, 0)  # WHT

        # TODO: bfloat16 is not supported in mint.nn.functional.pad
        x = F.pad(x.float(), spatial_pad, mode="constant", value=0).to(x.dtype)

        temporal_pad = (0, 0, 0, 0, 0, 1) if self.add_temporal_downsample else (0, 0, 0, 0, 1, 1)
        x = F.pad(x.float(), temporal_pad, mode="replicate").to(x.dtype)

        x = self.conv(x)
        return x


class DownsampleDCAE(nn.Cell):
    def __init__(self, in_channels: int, out_channels: int, add_temporal_downsample: bool = True):
        super().__init__()
        factor = 2 * 2 * 2 if add_temporal_downsample else 1 * 2 * 2
        assert out_channels % factor == 0
        self.conv = Conv3d(in_channels, out_channels // factor, kernel_size=3, stride=1, padding=1)

        self.add_temporal_downsample = add_temporal_downsample
        self.group_size = factor * in_channels // out_channels

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        r1 = 2 if self.add_temporal_downsample else 1
        h = self.conv(x)

        # h = rearrange(h, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
        B, C, frame, height, width = h.shape
        T = frame // r1
        H = height // 2
        W = width // 2

        h = h.reshape(B, C, T, r1, H, 2, W, 2)  # b, c, f, r1, h, r2, w, r3
        h = h.permute(0, 3, 5, 7, 1, 2, 4, 6)  # b, r1, r2, r3, c, f, h, w
        h = h.reshape(B, r1 * 2 * 2 * C, T, H, W)

        # shortcut = rearrange(x, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
        B, C, frame, height, width = x.shape
        T = frame // r1
        H = height // 2
        W = width // 2

        shortcut = x.reshape(B, C, T, r1, H, 2, W, 2)  # b, c, f, r1, h, r2, w, r3
        shortcut = shortcut.permute(0, 3, 5, 7, 1, 2, 4, 6)  # b, r1, r2, r3, c, f, h, w
        shortcut = shortcut.reshape(B, r1 * 2 * 2 * C, T, H, W)

        B, C, T, H, W = shortcut.shape
        shortcut = shortcut.view(B, h.shape[1], self.group_size, T, H, W).mean(dim=2)
        return h + shortcut


class Upsample(nn.Cell):
    """
    Upsampling block for spatial expansion.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels: int, add_temporal_upsample: bool = True):
        super().__init__()
        self.add_temporal_upsample = add_temporal_upsample
        self.scale_factor = (2, 2, 2) if add_temporal_upsample else (1, 2, 2)  # THW
        self.conv = Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: ms.Tensor):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        x = self.conv(x)
        return x


class UpsampleDCAE(nn.Cell):
    def __init__(self, in_channels: int, out_channels: int, add_temporal_upsample: bool = True):
        super().__init__()
        factor = 2 * 2 * 2 if add_temporal_upsample else 1 * 2 * 2
        self.conv = Conv3d(in_channels, out_channels * factor, kernel_size=3, stride=1, padding=1)

        self.add_temporal_upsample = add_temporal_upsample
        self.repeats = factor * out_channels // in_channels

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        r1 = 2 if self.add_temporal_upsample else 1
        h = self.conv(x)

        # h = rearrange(h, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)
        B, C_in, frame, H, W = h.shape
        C = C_in // (r1 * 2 * 2)

        h = h.reshape(B, r1, 2, 2, C, frame, H, W)  # b, r1, r2, r3, c, f, h, w
        h = h.permute(0, 4, 5, 1, 6, 2, 7, 3)  # b, c, f, r1, h, r2, w, r3
        h = h.reshape(B, C, frame * r1, H * 2, W * 2)

        shortcut = x.repeat_interleave(repeats=self.repeats, dim=1)
        # shortcut = rearrange(shortcut, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)
        B, C_in, frame, H, W = shortcut.shape
        C = C_in // (r1 * 2 * 2)

        shortcut = shortcut.reshape(B, r1, 2, 2, C, frame, H, W)  # b, r1, r2, r3, c, f, h, w
        shortcut = shortcut.permute(0, 4, 5, 1, 6, 2, 7, 3)  # b, c, f, r1, h, r2, w, r3
        shortcut = shortcut.reshape(B, C, frame * r1, H * 2, W * 2)

        return h + shortcut


class Encoder(nn.Cell):
    r"""
    Encoder network that compresses input to latent representation.

    Args:
        in_channels (int): Number of input channels.
        z_channels (int): Number of latent channels.
        block_out_channels (list of int): Output channels for each block.
        num_res_blocks (int): Number of residual blocks per block.
        ffactor_spatial (int): Spatial downsampling factor.
        ffactor_temporal (int): Temporal downsampling factor.
        non_linearity (str): Type of non-linearity to use. Default is "swish".
        downsample_match_channel (bool): Whether to match channels during downsampling.
    """

    def __init__(
        self,
        in_channels: int,
        z_channels: int,
        block_out_channels: Tuple[int, ...],
        num_res_blocks: int,
        ffactor_spatial: int,
        ffactor_temporal: int,
        non_linearity: str = "swish",
        downsample_match_channel: bool = True,
    ):
        super().__init__()
        if block_out_channels[-1] % (2 * z_channels) != 0:
            raise ValueError(
                "block_out_channels[-1 has to be divisible by 2 * out_channels, you have block_out_channels = "
                f"{block_out_channels[-1]} and out_channels = {z_channels}"
            )

        self.z_channels = z_channels
        self.block_out_channels = block_out_channels
        self.num_res_blocks = num_res_blocks

        self.group_size = block_out_channels[-1] // (2 * z_channels)
        self.nonlinearity = get_activation(non_linearity)

        # downsampling
        self.conv_in = Conv3d(in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1)

        self.down = nn.CellList()
        block_in = block_out_channels[0]
        for i_level, ch in enumerate(block_out_channels):
            block = nn.CellList()
            block_out = ch
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Cell()
            down.block = block

            add_spatial_downsample = bool(i_level < np.log2(ffactor_spatial))
            add_temporal_downsample = add_spatial_downsample and bool(
                i_level >= np.log2(ffactor_spatial // ffactor_temporal)
            )
            if add_spatial_downsample or add_temporal_downsample:
                assert i_level < len(block_out_channels) - 1
                block_out = block_out_channels[i_level + 1] if downsample_match_channel else block_in
                down.downsample = DownsampleDCAE(block_in, block_out, add_temporal_downsample)
                block_in = block_out
            self.down.append(down)

        # middle blocks
        self.mid = nn.CellList()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # output blocks / layers
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_out_channels[-1], eps=1e-6, affine=True)
        self.conv_out = Conv3d(block_out_channels[-1], 2 * z_channels, kernel_size=3, stride=1, padding=1)

        self.gradient_checkpointing = False

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        use_checkpointing = bool(self.training and self.gradient_checkpointing)

        # downsampling
        h = self.conv_in(x)
        for i_level in range(len(self.block_out_channels)):
            for i_block in range(self.num_res_blocks):
                h = forward_with_checkpointing(
                    self.down[i_level].block[i_block], h, use_checkpointing=use_checkpointing
                )
            if hasattr(self.down[i_level], "downsample"):
                h = forward_with_checkpointing(self.down[i_level].downsample, h, use_checkpointing=use_checkpointing)

        # middle
        h = forward_with_checkpointing(self.mid.block_1, h, use_checkpointing=use_checkpointing)
        h = forward_with_checkpointing(self.mid.attn_1, h, use_checkpointing=use_checkpointing)
        h = forward_with_checkpointing(self.mid.block_2, h, use_checkpointing=use_checkpointing)

        # output
        B, C, F, H, W = h.shape
        residual = h.view(B, C // self.group_size, self.group_size, F, H, W).mean(dim=2)

        h = self.norm_out(h)
        h = self.nonlinearity(h)
        h = self.conv_out(h)
        return h + residual


class Decoder(nn.Cell):
    r"""
    Decoder network that reconstructs output from latent representation.

    Args:
        z_channels (int): Number of latent channels.
        out_channels (int): Number of output channels.
        block_out_channels (list of int): Output channels for each block.
        num_res_blocks (int): Number of residual blocks per block.
        ffactor_spatial (int): Spatial upsampling factor.
        ffactor_temporal (int): Temporal upsampling factor.
        non_linearity (str): Type of non-linearity to use. Default is "swish".
        upsample_match_channel (bool): Whether to match channels during upsampling.
    """

    def __init__(
        self,
        z_channels: int,
        out_channels: int,
        block_out_channels: Tuple[int, ...],
        num_res_blocks: int,
        ffactor_spatial: int,
        ffactor_temporal: int,
        upsample_match_channel: bool = True,
        non_linearity: str = "swish",
    ):
        super().__init__()
        if block_out_channels[0] % z_channels != 0:
            raise ValueError(
                "block_out_channels[0] should be divisible by z_channels but has block_out_channels[0] = "
                f"{block_out_channels[0]} and z_channels = {z_channels}"
            )

        self.z_channels = z_channels
        self.block_out_channels = block_out_channels
        self.num_res_blocks = num_res_blocks

        self.repeats = block_out_channels[0] // z_channels
        self.nonlinearity = get_activation(non_linearity)

        # z to block_in
        block_in = block_out_channels[0]
        self.conv_in = Conv3d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Cell()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # upsampling
        self.up = nn.CellList()
        for i_level, ch in enumerate(block_out_channels):
            block = nn.CellList()
            block_out = ch
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Cell()
            up.block = block

            add_spatial_upsample = bool(i_level < np.log2(ffactor_spatial))
            add_temporal_upsample = bool(i_level < np.log2(ffactor_temporal))
            if add_spatial_upsample or add_temporal_upsample:
                assert i_level < len(block_out_channels) - 1
                block_out = block_out_channels[i_level + 1] if upsample_match_channel else block_in
                up.upsample = UpsampleDCAE(block_in, block_out, add_temporal_upsample)
                block_in = block_out
            self.up.append(up)

        # output
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_out_channels[-1], eps=1e-6, affine=True)
        self.conv_out = Conv3d(block_out_channels[-1], out_channels, kernel_size=3, stride=1, padding=1)

        self.gradient_checkpointing = False

    def construct(self, z: ms.Tensor) -> ms.Tensor:
        use_checkpointing = bool(self.training and self.gradient_checkpointing)

        # z to block_in
        h = self.conv_in(z) + z.repeat_interleave(repeats=self.repeats, dim=1)

        # middle
        h = forward_with_checkpointing(self.mid.block_1, h, use_checkpointing=use_checkpointing)
        h = forward_with_checkpointing(self.mid.attn_1, h, use_checkpointing=use_checkpointing)
        h = forward_with_checkpointing(self.mid.block_2, h, use_checkpointing=use_checkpointing)

        # upsampling
        for i_level in range(len(self.block_out_channels)):
            for i_block in range(self.num_res_blocks + 1):
                h = forward_with_checkpointing(self.up[i_level].block[i_block], h, use_checkpointing=use_checkpointing)
            if hasattr(self.up[i_level], "upsample"):
                h = forward_with_checkpointing(self.up[i_level].upsample, h, use_checkpointing=use_checkpointing)

        # output
        h = self.norm_out(h)
        h = self.nonlinearity(h)
        h = self.conv_out(h)
        return h


class AutoencoderKLConv3D(ModelMixin, ConfigMixin):
    r"""
    Autoencoder model with KL-regularized latent space based on 3D convolutions.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).
    """

    _supports_gradient_checkpointing = True

    # fmt: off
    @register_to_config
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_channels: int,
        block_out_channels: Tuple[int, ...],
        layers_per_block: int,
        ffactor_spatial: int,
        ffactor_temporal: int,
        sample_size: int,
        sample_tsize: int,
        scaling_factor: float = None,
        downsample_match_channel: bool = True,
        upsample_match_channel: bool = True,
        only_encoder: bool = False,     # only build encoder for saving memory
        only_decoder: bool = False,     # only build decoder for saving memory
    ):
        # fmt: on
        super().__init__()
        self.ffactor_spatial = ffactor_spatial
        self.ffactor_temporal = ffactor_temporal
        self.scaling_factor = scaling_factor

        # build model
        if not only_decoder:
            self.encoder = Encoder(
                in_channels=in_channels,
                z_channels=latent_channels,
                block_out_channels=block_out_channels,
                num_res_blocks=layers_per_block,
                ffactor_spatial=ffactor_spatial,
                ffactor_temporal=ffactor_temporal,
                downsample_match_channel=downsample_match_channel,
            )
        if not only_encoder:
            self.decoder = Decoder(
                z_channels=latent_channels,
                out_channels=out_channels,
                block_out_channels=list(reversed(block_out_channels)),
                num_res_blocks=layers_per_block,
                ffactor_spatial=ffactor_spatial,
                ffactor_temporal=ffactor_temporal,
                upsample_match_channel=upsample_match_channel,
            )

        # slicing and tiling related
        self.use_slicing = False
        self.slicing_bsz = 1
        self.use_spatial_tiling = False
        self.use_temporal_tiling = False
        self.use_tiling_during_training = False

        # only relevant if vae tiling is enabled
        self.tile_sample_min_size = sample_size
        self.tile_latent_min_size = sample_size // ffactor_spatial
        self.tile_sample_min_tsize = sample_tsize
        self.tile_latent_min_tsize = sample_tsize // ffactor_temporal
        self.tile_overlap_factor = 0.25

        # use torch.compile for faster encode speed ??
        self.use_compile = False

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (Encoder, Decoder)):
            module.gradient_checkpointing = value

    def enable_tiling_during_training(self, use_tiling: bool = True):
        self.use_tiling_during_training = use_tiling

    def disable_tiling_during_training(self):
        self.enable_tiling_during_training(False)

    def enable_temporal_tiling(self, use_tiling: bool = True):
        self.use_temporal_tiling = use_tiling

    def disable_temporal_tiling(self):
        self.enable_temporal_tiling(False)

    def enable_spatial_tiling(self, use_tiling: bool = True):
        self.use_spatial_tiling = use_tiling

    def disable_spatial_tiling(self):
        self.enable_spatial_tiling(False)

    def enable_tiling(
        self,
        tile_sample_min_size: Optional[int] = None,
        tile_sample_min_tsize: Optional[int] = None,
        tile_overlap_factor: Optional[float] = None,
    ):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.

        Args:
            tile_sample_min_size (`int`, *optional*):
                The minimum size required for a sample to be separated into tiles across the spatial dimension.
            tile_sample_min_size (`int`, *optional*):
                The minimum size required for a sample to be separated into tiles across the temporal dimension.
            tile_overlap_factor (`float`, *optional*):
                The overlap factor required for a latent to be separated into tiles across the dimensions.
        """
        self.use_tiling = True
        self.tile_sample_min_size = tile_sample_min_size or self.tile_sample_min_size
        self.tile_sample_min_tsize = tile_sample_min_tsize or self.tile_sample_min_tsize
        self.tile_overlap_factor = tile_overlap_factor or self.tile_overlap_factor
        self.tile_latent_min_size = self.tile_sample_min_size // self.config.ffactor_spatial
        self.tile_latent_min_tsize = self.tile_sample_min_tsize // self.config.ffactor_temporal  # config?

    def disable_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_tiling` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_tiling = False

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

    def blend_h(self, a: ms.Tensor, b: ms.Tensor, blend_extent: int) -> ms.Tensor:
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = \
                a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (x / blend_extent)
        return b

    def blend_v(self, a: ms.Tensor, b: ms.Tensor, blend_extent: int) -> ms.Tensor:
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = \
                a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (y / blend_extent)
        return b

    def blend_t(self, a: ms.Tensor, b: ms.Tensor, blend_extent: int) -> ms.Tensor:
        blend_extent = min(a.shape[-3], b.shape[-3], blend_extent)
        for x in range(blend_extent):
            b[:, :, x, :, :] = \
                a[:, :, -blend_extent + x, :, :] * (1 - x / blend_extent) + b[:, :, x, :, :] * (x / blend_extent)
        return b

    def spatial_tiled_encode(self, x: ms.Tensor):
        """ spatial tailing for frames """
        B, C, T, H, W = x.shape
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))  # 256 * (1 - 0.25) = 192
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)  # 8 * 0.25 = 2
        row_limit = self.tile_latent_min_size - blend_extent  # 8 - 2 = 6

        rows = []
        for i in range(0, H, overlap_size):
            row = []
            for j in range(0, W, overlap_size):
                tile = x[:, :, :, i: i + self.tile_sample_min_size, j: j + self.tile_sample_min_size]
                tile = self.encoder(tile)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(mint.cat(result_row, dim=-1))
        moments = mint.cat(result_rows, dim=-2)
        return moments

    def temporal_tiled_encode(self, x: ms.Tensor):
        """ temporal tailing for frames """
        B, C, T, H, W = x.shape
        overlap_size = int(self.tile_sample_min_tsize * (1 - self.tile_overlap_factor))  # 64 * (1 - 0.25) = 48
        blend_extent = int(self.tile_latent_min_tsize * self.tile_overlap_factor)  # 8 * 0.25 = 2
        t_limit = self.tile_latent_min_tsize - blend_extent  # 8 - 2 = 6

        row = []
        for i in range(0, T, overlap_size):
            tile = x[:, :, i: i + self.tile_sample_min_tsize, :, :]
            if self.use_spatial_tiling and (
                    tile.shape[-1] > self.tile_sample_min_size or tile.shape[-2] > self.tile_sample_min_size):
                tile = self.spatial_tiled_encode(tile)
            else:
                tile = self.encoder(tile)
            row.append(tile)
        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_extent)
            result_row.append(tile[:, :, :t_limit, :, :])
        moments = mint.cat(result_row, dim=-3)
        return moments

    def spatial_tiled_decode(self, z: ms.Tensor):
        """ spatial tailing for frames """
        B, C, T, H, W = z.shape
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))  # 8 * (1 - 0.25) = 6
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)  # 256 * 0.25 = 64
        row_limit = self.tile_sample_min_size - blend_extent  # 256 - 64 = 192

        rows = []
        for i in range(0, H, overlap_size):
            row = []
            for j in range(0, W, overlap_size):
                tile = z[:, :, :, i: i + self.tile_latent_min_size, j: j + self.tile_latent_min_size]
                decoded = self.decoder(tile)
                row.append(decoded)
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(mint.cat(result_row, dim=-1))
        dec = mint.cat(result_rows, dim=-2)
        return dec

    def temporal_tiled_decode(self, z: ms.Tensor):
        """ temporal tailing for frames """
        B, C, T, H, W = z.shape
        overlap_size = int(self.tile_latent_min_tsize * (1 - self.tile_overlap_factor))  # 8 * (1 - 0.25) = 6
        blend_extent = int(self.tile_sample_min_tsize * self.tile_overlap_factor)  # 64 * 0.25 = 16
        t_limit = self.tile_sample_min_tsize - blend_extent  # 64 - 16 = 48
        assert 0 < overlap_size < self.tile_latent_min_tsize

        row = []
        for i in range(0, T, overlap_size):
            tile = z[:, :, i: i + self.tile_latent_min_tsize, :, :]
            if self.use_spatial_tiling and (
                    tile.shape[-1] > self.tile_latent_min_size or tile.shape[-2] > self.tile_latent_min_size):
                decoded = self.spatial_tiled_decode(tile)
            else:
                decoded = self.decoder(tile)
            row.append(decoded)

        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_extent)
            result_row.append(tile[:, :, :t_limit, :, :])
        dec = mint.cat(result_row, dim=-3)
        return dec

    def _encode(self, x: ms.Tensor):

        batch_size, num_channels, num_frames, height, width = x.shape

        if self.use_temporal_tiling and num_frames > self.tile_sample_min_tsize:
            return self.temporal_tiled_encode(x)

        if self.use_spatial_tiling and (width > self.tile_sample_min_size or height > self.tile_sample_min_size):
            return self.spatial_tiled_encode(x)

        enc = self.encoder(x)

        return enc

    def encode(
        self, x: ms.Tensor, return_dict: bool = True
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        r"""
        Encodes the input by passing through the encoder network.
        Support slicing and tiling for memory efficiency.

        Args:
            x (`ms.Tensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
                The latent representations of the encoded videos. If `return_dict` is True, a
                [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain `tuple` is returned.
        """

        if len(x.shape) != 5:  # (B, C, T, H, W)
            x = x[:, :, None]

        if len(x.shape) != 5:  # (B, C, T, H, W)  this is original usage, seem redundant?
            raise ValueError(f"Input tensor must have 5 dimensions (B, C, T, H, W), but got shape {x.shape}")

        if x.shape[2] == 1:
            x = x.expand(-1, -1, self.ffactor_temporal, -1, -1)
        else:
            if x.shape[2] == self.ffactor_temporal or x.shape[2] % self.ffactor_temporal != 0:
                raise ValueError(
                    f"Temporal dimension must not equal ffactor_temporal and must be divisible by it. "
                    f"You have T = {x.shape[2]} and ffactor_temporal = {self.ffactor_temporal}"
                )

        if self.use_slicing and x.shape[0] > 1:
            if self.slicing_bsz == 1:
                encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
            else:
                sections = [self.slicing_bsz] * (x.shape[0] // self.slicing_bsz)
                if x.shape[0] % self.slicing_bsz != 0:
                    sections.append(x.shape[0] % self.slicing_bsz)
                encoded_slices = [self._encode(x_slice) for x_slice in x.split(sections)]
            h = mint.cat(encoded_slices)
        else:
            h = self._encode(x)
        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(self, z: ms.Tensor, return_dict: bool = True):

        batch_size, num_channels, num_frames, height, width = z.shape

        if self.use_temporal_tiling and num_frames > self.tile_latent_min_tsize:
            return self.temporal_tiled_decode(z)

        if self.use_spatial_tiling and (
                width > self.tile_latent_min_size or height > self.tile_latent_min_size):
            return self.spatial_tiled_decode(z)

        dec = self.decoder(z)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def decode(self, z: ms.Tensor, return_dict: bool = True) -> Union[DecoderOutput, ms.Tensor]:
        """
        Decodes the input by passing through the decoder network.
        Support slicing and tiling for memory efficiency.
        """
        # [B, C, H, W] -> [B, C, T, H, W]
        if len(z.shape) == 4 and hasattr(self, "ffactor_temporal"):
            z = z.unsqueeze(2)

        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            decoded = mint.cat(decoded_slices)
        else:
            decoded = self._decode(z).sample

        if z.shape[-3] == 1:
            decoded = decoded[:, :, -1:]

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def random_reset_tiling(self, x: ms.Tensor):
        if x.shape[-3] == 1:
            self.disable_spatial_tiling()
            self.disable_temporal_tiling()
            return

        # Use fixed shape here
        min_sample_size = int(1 / self.tile_overlap_factor) * self.ffactor_spatial
        min_sample_tsize = int(1 / self.tile_overlap_factor) * self.ffactor_temporal
        sample_size = random.choice([None, 1 * min_sample_size, 2 * min_sample_size, 3 * min_sample_size])
        if sample_size is None:
            self.disable_spatial_tiling()
        else:
            self.tile_sample_min_size = sample_size
            self.tile_latent_min_size = sample_size // self.ffactor_spatial
            self.enable_spatial_tiling()

        sample_tsize = random.choice([None, 1 * min_sample_tsize, 2 * min_sample_tsize, 3 * min_sample_tsize])
        if sample_tsize is None:
            self.disable_temporal_tiling()
        else:
            self.tile_sample_min_tsize = sample_tsize
            self.tile_latent_min_tsize = sample_tsize // self.ffactor_temporal

            self.enable_temporal_tiling()

    def construct(
        self,
        sample: ms.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True
    ):
        posterior = self.encode(sample).latent_dist
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()

        dec = self.decode(z, return_dict=return_dict).sample

        if not return_dict:
            return (dec, posterior)

        return DecoderOutput(sample=dec, posterior=posterior)
