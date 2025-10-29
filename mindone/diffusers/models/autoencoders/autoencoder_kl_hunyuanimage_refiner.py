# Copyright 2025 The Hunyuan Team and The HuggingFace Team. All rights reserved.
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

from typing import Optional, Tuple, Union

import numpy as np

import mindspore as ms
from mindspore import mint, nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import logging
from ..activations import get_activation
from ..layers_compat import scaled_dot_product_attention
from ..modeling_outputs import AutoencoderKLOutput
from ..modeling_utils import ModelMixin
from .vae import DecoderOutput, DiagonalGaussianDistribution

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class HunyuanImageRefinerCausalConv3d(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        bias: bool = True,
        pad_mode: str = "replicate",
    ) -> None:
        super().__init__()

        kernel_size = (kernel_size, kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size

        self.pad_mode = pad_mode
        self.time_causal_padding = (
            kernel_size[0] // 2,
            kernel_size[0] // 2,
            kernel_size[1] // 2,
            kernel_size[1] // 2,
            kernel_size[2] - 1,
            0,
        )

        self.conv = mint.nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        # mint.nn.functional.pad not support bf16
        hidden_states = mint.nn.functional.pad(hidden_states.float(), self.time_causal_padding, mode=self.pad_mode).to(
            hidden_states.dtype
        )
        return self.conv(hidden_states)


class HunyuanImageRefinerRMS_norm(nn.Cell):
    r"""
    A custom RMS normalization layer.

    Args:
        dim (int): The number of dimensions to normalize over.
        channel_first (bool, optional): Whether the input tensor has channels as the first dimension.
            Default is True.
        images (bool, optional): Whether the input represents image data. Default is True.
        bias (bool, optional): Whether to include a learnable bias term. Default is False.
    """

    def __init__(self, dim: int, channel_first: bool = True, images: bool = True, bias: bool = False) -> None:
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = ms.Parameter(mint.ones(shape))
        self.bias = ms.Parameter(mint.zeros(shape)) if bias else 0.0

    def construct(self, x):
        return (
            mint.nn.functional.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias
        )


class HunyuanImageRefinerAttnBlock(nn.Cell):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = HunyuanImageRefinerRMS_norm(in_channels, images=False)

        self.to_q = mint.nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.to_k = mint.nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.to_v = mint.nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.proj_out = mint.nn.Conv3d(in_channels, in_channels, kernel_size=1)

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        identity = x

        x = self.norm(x)

        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)

        batch_size, channels, frames, height, width = query.shape

        query = query.reshape(batch_size, channels, frames * height * width).permute(0, 2, 1).unsqueeze(1).contiguous()
        key = key.reshape(batch_size, channels, frames * height * width).permute(0, 2, 1).unsqueeze(1).contiguous()
        value = value.reshape(batch_size, channels, frames * height * width).permute(0, 2, 1).unsqueeze(1).contiguous()

        x = scaled_dot_product_attention(query, key, value, attn_mask=None)

        # batch_size, 1, frames * height * width, channels

        x = x.squeeze(1).reshape(batch_size, frames, height, width, channels).permute(0, 4, 1, 2, 3)
        x = self.proj_out(x)

        return x + identity


class HunyuanImageRefinerUpsampleDCAE(nn.Cell):
    def __init__(self, in_channels: int, out_channels: int, add_temporal_upsample: bool = True):
        super().__init__()
        factor = 2 * 2 * 2 if add_temporal_upsample else 1 * 2 * 2
        self.conv = HunyuanImageRefinerCausalConv3d(in_channels, out_channels * factor, kernel_size=3)

        self.add_temporal_upsample = add_temporal_upsample
        self.repeats = factor * out_channels // in_channels

    @staticmethod
    def _dcae_upsample_rearrange(tensor, r1=1, r2=2, r3=2):
        """
        Convert (b, r1*r2*r3*c, f, h, w) -> (b, c, r1*f, r2*h, r3*w)

        Args:
            tensor: Input tensor of shape (b, r1*r2*r3*c, f, h, w)
            r1: temporal upsampling factor
            r2: height upsampling factor
            r3: width upsampling factor
        """
        b, packed_c, f, h, w = tensor.shape
        factor = r1 * r2 * r3
        c = packed_c // factor

        tensor = tensor.view(b, r1, r2, r3, c, f, h, w)
        tensor = tensor.permute(0, 4, 5, 1, 6, 2, 7, 3)
        return tensor.reshape(b, c, f * r1, h * r2, w * r3)

    def construct(self, x: ms.Tensor):
        r1 = 2 if self.add_temporal_upsample else 1
        h = self.conv(x)
        if self.add_temporal_upsample:
            h = self._dcae_upsample_rearrange(h, r1=1, r2=2, r3=2)
            h = h[:, : h.shape[1] // 2]

            # shortcut computation
            shortcut = self._dcae_upsample_rearrange(x, r1=1, r2=2, r3=2)
            shortcut = shortcut.repeat_interleave(repeats=self.repeats // 2, dim=1)

        else:
            h = self._dcae_upsample_rearrange(h, r1=r1, r2=2, r3=2)
            shortcut = x.repeat_interleave(repeats=self.repeats, dim=1)
            shortcut = self._dcae_upsample_rearrange(shortcut, r1=r1, r2=2, r3=2)
        return h + shortcut


class HunyuanImageRefinerDownsampleDCAE(nn.Cell):
    def __init__(self, in_channels: int, out_channels: int, add_temporal_downsample: bool = True):
        super().__init__()
        factor = 2 * 2 * 2 if add_temporal_downsample else 1 * 2 * 2
        assert out_channels % factor == 0
        # self.conv = Conv3d(in_channels, out_channels // factor, kernel_size=3, stride=1, padding=1)
        self.conv = HunyuanImageRefinerCausalConv3d(in_channels, out_channels // factor, kernel_size=3)

        self.add_temporal_downsample = add_temporal_downsample
        self.group_size = factor * in_channels // out_channels

    @staticmethod
    def _dcae_downsample_rearrange(tensor, r1=1, r2=2, r3=2):
        """
        Convert (b, c, r1*f, r2*h, r3*w) -> (b, r1*r2*r3*c, f, h, w)

        This packs spatial/temporal dimensions into channels (opposite of upsample)
        """
        b, c, packed_f, packed_h, packed_w = tensor.shape
        f, h, w = packed_f // r1, packed_h // r2, packed_w // r3

        tensor = tensor.view(b, c, f, r1, h, r2, w, r3)
        tensor = tensor.permute(0, 3, 5, 7, 1, 2, 4, 6)
        return tensor.reshape(b, r1 * r2 * r3 * c, f, h, w)

    def construct(self, x: ms.Tensor):
        r1 = 2 if self.add_temporal_downsample else 1
        h = self.conv(x)
        if self.add_temporal_downsample:
            # h = rearrange(h, "b c f (h r2) (w r3) -> b (r2 r3 c) f h w", r2=2, r3=2)
            h = self._dcae_downsample_rearrange(h, r1=1, r2=2, r3=2)
            h = mint.cat([h, h], dim=1)
            # shortcut computation
            # shortcut = rearrange(x, "b c f (h r2) (w r3) -> b (r2 r3 c) f h w", r2=2, r3=2)
            shortcut = self._dcae_downsample_rearrange(x, r1=1, r2=2, r3=2)
            B, C, T, H, W = shortcut.shape
            shortcut = shortcut.view(B, h.shape[1], self.group_size // 2, T, H, W).mean(dim=2)
        else:
            # h = rearrange(h, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
            h = self._dcae_downsample_rearrange(h, r1=r1, r2=2, r3=2)
            # shortcut = rearrange(x, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
            shortcut = self._dcae_downsample_rearrange(x, r1=r1, r2=2, r3=2)
            B, C, T, H, W = shortcut.shape
            shortcut = shortcut.view(B, h.shape[1], self.group_size, T, H, W).mean(dim=2)

        return h + shortcut


class HunyuanImageRefinerResnetBlock(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        non_linearity: str = "swish",
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels

        self.nonlinearity = get_activation(non_linearity)

        self.norm1 = HunyuanImageRefinerRMS_norm(in_channels, images=False)
        self.conv1 = HunyuanImageRefinerCausalConv3d(in_channels, out_channels, kernel_size=3)

        self.norm2 = HunyuanImageRefinerRMS_norm(out_channels, images=False)
        self.conv2 = HunyuanImageRefinerCausalConv3d(out_channels, out_channels, kernel_size=3)

        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = mint.nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        residual = hidden_states

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)

        return hidden_states + residual


class HunyuanImageRefinerMidBlock(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        num_layers: int = 1,
        add_attention: bool = True,
    ) -> None:
        super().__init__()
        self.add_attention = add_attention

        # There is always at least one resnet
        resnets = [
            HunyuanImageRefinerResnetBlock(
                in_channels=in_channels,
                out_channels=in_channels,
            )
        ]
        attentions = []

        for _ in range(num_layers):
            if self.add_attention:
                attentions.append(HunyuanImageRefinerAttnBlock(in_channels))
            else:
                attentions.append(None)

            resnets.append(
                HunyuanImageRefinerResnetBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                )
            )

        self.attentions = nn.CellList(attentions)
        self.resnets = nn.CellList(resnets)

        self.gradient_checkpointing = False

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        hidden_states = self.resnets[0](hidden_states)

        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                hidden_states = attn(hidden_states)
            hidden_states = resnet(hidden_states)

        return hidden_states


class HunyuanImageRefinerDownBlock3D(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        downsample_out_channels: Optional[int] = None,
        add_temporal_downsample: int = True,
    ) -> None:
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                HunyuanImageRefinerResnetBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                )
            )

        self.resnets = nn.CellList(resnets)

        if downsample_out_channels is not None:
            self.downsamplers = nn.CellList(
                [
                    HunyuanImageRefinerDownsampleDCAE(
                        out_channels,
                        out_channels=downsample_out_channels,
                        add_temporal_downsample=add_temporal_downsample,
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

        return hidden_states


class HunyuanImageRefinerUpBlock3D(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        upsample_out_channels: Optional[int] = None,
        add_temporal_upsample: bool = True,
    ) -> None:
        super().__init__()
        resnets = []

        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            resnets.append(
                HunyuanImageRefinerResnetBlock(
                    in_channels=input_channels,
                    out_channels=out_channels,
                )
            )

        self.resnets = nn.CellList(resnets)

        if upsample_out_channels is not None:
            self.upsamplers = nn.CellList(
                [
                    HunyuanImageRefinerUpsampleDCAE(
                        out_channels,
                        out_channels=upsample_out_channels,
                        add_temporal_upsample=add_temporal_upsample,
                    )
                ]
            )
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


class HunyuanImageRefinerEncoder3D(nn.Cell):
    r"""
    3D vae encoder for HunyuanImageRefiner.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 1024, 1024),
        layers_per_block: int = 2,
        temporal_compression_ratio: int = 4,
        spatial_compression_ratio: int = 16,
        downsample_match_channel: bool = True,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.group_size = block_out_channels[-1] // self.out_channels

        self.conv_in = HunyuanImageRefinerCausalConv3d(in_channels, block_out_channels[0], kernel_size=3)
        self.mid_block = None
        self.down_blocks = nn.CellList([])

        input_channel = block_out_channels[0]
        for i in range(len(block_out_channels)):
            add_spatial_downsample = i < np.log2(spatial_compression_ratio)
            output_channel = block_out_channels[i]
            if not add_spatial_downsample:
                down_block = HunyuanImageRefinerDownBlock3D(
                    num_layers=layers_per_block,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    downsample_out_channels=None,
                    add_temporal_downsample=False,
                )
                input_channel = output_channel
            else:
                add_temporal_downsample = i >= np.log2(spatial_compression_ratio // temporal_compression_ratio)
                downsample_out_channels = block_out_channels[i + 1] if downsample_match_channel else output_channel
                down_block = HunyuanImageRefinerDownBlock3D(
                    num_layers=layers_per_block,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    downsample_out_channels=downsample_out_channels,
                    add_temporal_downsample=add_temporal_downsample,
                )
                input_channel = downsample_out_channels

            self.down_blocks.append(down_block)

        self.mid_block = HunyuanImageRefinerMidBlock(in_channels=block_out_channels[-1])

        self.norm_out = HunyuanImageRefinerRMS_norm(block_out_channels[-1], images=False)
        self.conv_act = mint.nn.SiLU()
        self.conv_out = HunyuanImageRefinerCausalConv3d(block_out_channels[-1], out_channels, kernel_size=3)

        self.gradient_checkpointing = False

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        hidden_states = self.conv_in(hidden_states)

        for down_block in self.down_blocks:
            hidden_states = down_block(hidden_states)

        hidden_states = self.mid_block(hidden_states)

        # short_cut = rearrange(hidden_states, "b (c r) f h w -> b c r f h w", r=self.group_size).mean(dim=2)
        batch_size, _, frame, height, width = hidden_states.shape
        short_cut = hidden_states.view(batch_size, -1, self.group_size, frame, height, width).mean(dim=2)

        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        hidden_states += short_cut

        return hidden_states


class HunyuanImageRefinerDecoder3D(nn.Cell):
    r"""
    Causal decoder for 3D video-like data used for HunyuanImage-2.1 Refiner.
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (1024, 1024, 512, 256, 128),
        layers_per_block: int = 2,
        spatial_compression_ratio: int = 16,
        temporal_compression_ratio: int = 4,
        upsample_match_channel: bool = True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.repeat = block_out_channels[0] // self.in_channels

        self.conv_in = HunyuanImageRefinerCausalConv3d(self.in_channels, block_out_channels[0], kernel_size=3)
        self.up_blocks = nn.CellList([])

        # mid
        self.mid_block = HunyuanImageRefinerMidBlock(in_channels=block_out_channels[0])

        # up
        input_channel = block_out_channels[0]
        for i in range(len(block_out_channels)):
            output_channel = block_out_channels[i]

            add_spatial_upsample = i < np.log2(spatial_compression_ratio)
            add_temporal_upsample = i < np.log2(temporal_compression_ratio)
            if add_spatial_upsample or add_temporal_upsample:
                upsample_out_channels = block_out_channels[i + 1] if upsample_match_channel else output_channel
                up_block = HunyuanImageRefinerUpBlock3D(
                    num_layers=self.layers_per_block + 1,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    upsample_out_channels=upsample_out_channels,
                    add_temporal_upsample=add_temporal_upsample,
                )
                input_channel = upsample_out_channels
            else:
                up_block = HunyuanImageRefinerUpBlock3D(
                    num_layers=self.layers_per_block + 1,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    upsample_out_channels=None,
                    add_temporal_upsample=False,
                )
                input_channel = output_channel

            self.up_blocks.append(up_block)

        # out
        self.norm_out = HunyuanImageRefinerRMS_norm(block_out_channels[-1], images=False)
        self.conv_act = mint.nn.SiLU()
        self.conv_out = HunyuanImageRefinerCausalConv3d(block_out_channels[-1], out_channels, kernel_size=3)

        self.gradient_checkpointing = False

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        hidden_states = self.conv_in(hidden_states) + hidden_states.repeat_interleave(repeats=self.repeat, dim=1)

        hidden_states = self.mid_block(hidden_states)

        for up_block in self.up_blocks:
            hidden_states = up_block(hidden_states)

        # post-process
        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        return hidden_states


class AutoencoderKLHunyuanImageRefiner(ModelMixin, ConfigMixin):
    r"""
    A VAE model with KL loss for encoding videos into latents and decoding latent representations into videos. Used for
    HunyuanImage-2.1 Refiner.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 32,
        block_out_channels: Tuple[int] = (128, 256, 512, 1024, 1024),
        layers_per_block: int = 2,
        spatial_compression_ratio: int = 16,
        temporal_compression_ratio: int = 4,
        downsample_match_channel: bool = True,
        upsample_match_channel: bool = True,
        scaling_factor: float = 1.03682,
    ) -> None:
        super().__init__()

        self.encoder = HunyuanImageRefinerEncoder3D(
            in_channels=in_channels,
            out_channels=latent_channels * 2,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            temporal_compression_ratio=temporal_compression_ratio,
            spatial_compression_ratio=spatial_compression_ratio,
            downsample_match_channel=downsample_match_channel,
        )

        self.decoder = HunyuanImageRefinerDecoder3D(
            in_channels=latent_channels,
            out_channels=out_channels,
            block_out_channels=list(reversed(block_out_channels)),
            layers_per_block=layers_per_block,
            temporal_compression_ratio=temporal_compression_ratio,
            spatial_compression_ratio=spatial_compression_ratio,
            upsample_match_channel=upsample_match_channel,
        )

        self.diag_gauss_dist = DiagonalGaussianDistribution()

        self.spatial_compression_ratio = spatial_compression_ratio
        self.temporal_compression_ratio = temporal_compression_ratio

        # When decoding a batch of video latents at a time, one can save memory by slicing across the batch dimension
        # to perform decoding of a single video latent at a time.
        self.use_slicing = False

        # When decoding spatially large video latents, the memory requirement is very high. By breaking the video latent
        # frames spatially into smaller tiles and performing multiple forward passes for decoding, and then blending the
        # intermediate tiles together, the memory requirement can be lowered.
        self.use_tiling = False

        # The minimal tile height and width for spatial tiling to be used
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256

        # The minimal distance between two spatial tiles
        self.tile_sample_stride_height = 192
        self.tile_sample_stride_width = 192

        self.tile_overlap_factor = 0.25

    def enable_tiling(
        self,
        tile_sample_min_height: Optional[int] = None,
        tile_sample_min_width: Optional[int] = None,
        tile_sample_stride_height: Optional[float] = None,
        tile_sample_stride_width: Optional[float] = None,
        tile_overlap_factor: Optional[float] = None,
    ) -> None:
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.

        Args:
            tile_sample_min_height (`int`, *optional*):
                The minimum height required for a sample to be separated into tiles across the height dimension.
            tile_sample_min_width (`int`, *optional*):
                The minimum width required for a sample to be separated into tiles across the width dimension.
            tile_sample_stride_height (`int`, *optional*):
                The minimum amount of overlap between two consecutive vertical tiles. This is to ensure that there are
                no tiling artifacts produced across the height dimension.
            tile_sample_stride_width (`int`, *optional*):
                The stride between two consecutive horizontal tiles. This is to ensure that there are no tiling
                artifacts produced across the width dimension.
        """
        self.use_tiling = True
        self.tile_sample_min_height = tile_sample_min_height or self.tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
        self.tile_sample_stride_height = tile_sample_stride_height or self.tile_sample_stride_height
        self.tile_sample_stride_width = tile_sample_stride_width or self.tile_sample_stride_width
        self.tile_overlap_factor = tile_overlap_factor or self.tile_overlap_factor

    def disable_tiling(self) -> None:
        r"""
        Disable tiled VAE decoding. If `enable_tiling` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_tiling = False

    def enable_slicing(self) -> None:
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.use_slicing = True

    def disable_slicing(self) -> None:
        r"""
        Disable sliced VAE decoding. If `enable_slicing` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_slicing = False

    def _encode(self, x: ms.Tensor) -> ms.Tensor:
        _, _, _, height, width = x.shape

        if self.use_tiling and (width > self.tile_sample_min_width or height > self.tile_sample_min_height):
            return self.tiled_encode(x)

        x = self.encoder(x)
        return x

    def encode(
        self, x: ms.Tensor, return_dict: bool = False
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        r"""
        Encode a batch of images into latents.

        Args:
            x (`ms.Tensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
                The latent representations of the encoded videos. If `return_dict` is True, a
                [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain `tuple` is returned.
        """
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
            h = mint.cat(encoded_slices)
        else:
            h = self._encode(x)

        # we cannot use class in graph mode, even for jit_class or subclass of Tensor. :-(
        # posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (h,)
        return AutoencoderKLOutput(latent_dist=h)

    def _decode(self, z: ms.Tensor) -> ms.Tensor:
        _, _, _, height, width = z.shape
        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio

        if self.use_tiling and (width > tile_latent_min_width or height > tile_latent_min_height):
            return self.tiled_decode(z)

        dec = self.decoder(z)

        return dec

    def decode(self, z: ms.Tensor, return_dict: bool = False) -> Union[DecoderOutput, ms.Tensor]:
        r"""
        Decode a batch of images.

        Args:
            z (`ms.Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.
        """
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice) for z_slice in z.split(1)]
            decoded = mint.cat(decoded_slices)
        else:
            decoded = self._decode(z)

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def blend_v(self, a: ms.Tensor, b: ms.Tensor, blend_extent: int) -> ms.Tensor:
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
                y / blend_extent
            )
        return b

    def blend_h(self, a: ms.Tensor, b: ms.Tensor, blend_extent: int) -> ms.Tensor:
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (
                x / blend_extent
            )
        return b

    def blend_t(self, a: ms.Tensor, b: ms.Tensor, blend_extent: int) -> ms.Tensor:
        blend_extent = min(a.shape[-3], b.shape[-3], blend_extent)
        for x in range(blend_extent):
            b[:, :, x, :, :] = a[:, :, -blend_extent + x, :, :] * (1 - x / blend_extent) + b[:, :, x, :, :] * (
                x / blend_extent
            )
        return b

    def tiled_encode(self, x: ms.Tensor) -> ms.Tensor:
        r"""Encode a batch of images using a tiled encoder.

        Args:
            x (`ms.Tensor`): Input batch of videos.

        Returns:
            `ms.Tensor`:
                The latent representation of the encoded videos.
        """
        _, _, _, height, width = x.shape

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        overlap_height = int(tile_latent_min_height * (1 - self.tile_overlap_factor))  # 256 * (1 - 0.25) = 192
        overlap_width = int(tile_latent_min_width * (1 - self.tile_overlap_factor))  # 256 * (1 - 0.25) = 192
        blend_height = int(tile_latent_min_height * self.tile_overlap_factor)  # 8 * 0.25 = 2
        blend_width = int(tile_latent_min_width * self.tile_overlap_factor)  # 8 * 0.25 = 2
        row_limit_height = tile_latent_min_height - blend_height  # 8 - 2 = 6
        row_limit_width = tile_latent_min_width - blend_width  # 8 - 2 = 6

        rows = []
        for i in range(0, height, overlap_height):
            row = []
            for j in range(0, width, overlap_width):
                tile = x[
                    :,
                    :,
                    :,
                    i : i + self.tile_sample_min_height,
                    j : j + self.tile_sample_min_width,
                ]
                tile = self.encoder(tile)
                row.append(tile)
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(tile[:, :, :, :row_limit_height, :row_limit_width])
            result_rows.append(mint.cat(result_row, dim=-1))
        moments = mint.cat(result_rows, dim=-2)

        return moments

    def tiled_decode(self, z: ms.Tensor) -> ms.Tensor:
        r"""
        Decode a batch of images using a tiled decoder.

        Args:
            z (`ms.Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.
        """

        _, _, _, height, width = z.shape
        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        overlap_height = int(tile_latent_min_height * (1 - self.tile_overlap_factor))  # 8 * (1 - 0.25) = 6
        overlap_width = int(tile_latent_min_width * (1 - self.tile_overlap_factor))  # 8 * (1 - 0.25) = 6
        blend_height = int(tile_latent_min_height * self.tile_overlap_factor)  # 256 * 0.25 = 64
        blend_width = int(tile_latent_min_width * self.tile_overlap_factor)  # 256 * 0.25 = 64
        row_limit_height = tile_latent_min_height - blend_height  # 256 - 64 = 192
        row_limit_width = tile_latent_min_width - blend_width  # 256 - 64 = 192

        rows = []
        for i in range(0, height, overlap_height):
            row = []
            for j in range(0, width, overlap_width):
                tile = z[
                    :,
                    :,
                    :,
                    i : i + tile_latent_min_height,
                    j : j + tile_latent_min_width,
                ]
                decoded = self.decoder(tile)
                row.append(decoded)
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(tile[:, :, :, :row_limit_height, :row_limit_width])
            result_rows.append(mint.cat(result_row, dim=-1))
        dec = mint.cat(result_rows, dim=-2)

        return dec

    def construct(
        self,
        sample: ms.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = False,
        generator: Optional[np.random.Generator] = None,
    ) -> Union[DecoderOutput, ms.Tensor]:
        r"""
        Args:
            sample (`ms.Tensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        x = sample
        posterior = self.encode(x)[0]
        if sample_posterior:
            z = self.diag_gauss_dist.sample(posterior, generator=generator)
        else:
            z = self.diag_gauss_dist.mode(posterior)
        dec = self.decode(z, return_dict=return_dict)
        return dec
