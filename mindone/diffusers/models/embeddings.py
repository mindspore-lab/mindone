# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# This code is adapted from https://github.com/huggingface/diffusers
# with modifications to run diffusers on mindspore.
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
from typing import List, Optional, Tuple, Union

import numpy as np

import mindspore as ms
import mindspore.common.initializer as init
from mindspore import mint, nn, ops

from ..utils import deprecate
from .activations import FP32SiLU, get_activation
from .attention_processor import Attention
from .layers_compat import GELU, pad, unflatten, view_as_complex


def get_timestep_embedding(
    timesteps: ms.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> ms.Tensor:
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    Args
        timesteps (ms.Tensor):
            a 1-D Tensor of N indices, one per batch element. These may be fractional.
        embedding_dim (int):
            the dimension of the output.
        flip_sin_to_cos (bool):
            Whether the embedding order should be `cos, sin` (if True) or `sin, cos` (if False)
        downscale_freq_shift (float):
            Controls the delta between frequencies between dimensions
        scale (float):
            Scaling factor applied to the embeddings.
        max_period (int):
            Controls the maximum frequency of the embeddings
    Returns
        ms.Tensor: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -mint.log(ms.tensor(max_period, dtype=ms.float32)) * mint.arange(start=0, end=half_dim, dtype=ms.float32)
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = mint.exp(exponent)
    # emb = timesteps[:, None].float() * emb[None, :]
    emb = timesteps.expand_dims(axis=1).float() * emb.expand_dims(axis=0)

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = mint.cat([mint.sin(emb), mint.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        # emb = ops.cat([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)
        split_emb = mint.split(emb, [half_dim, emb.shape[1] - half_dim], dim=1)
        emb = mint.cat([split_emb[1], split_emb[0]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = mint.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def get_3d_sincos_pos_embed(
    embed_dim: int,
    spatial_size: Union[int, Tuple[int, int]],
    temporal_size: int,
    spatial_interpolation_scale: float = 1.0,
    temporal_interpolation_scale: float = 1.0,
    output_type: str = "np",
) -> ms.Tensor:
    r"""
    Creates 3D sinusoidal positional embeddings.

    Args:
        embed_dim (`int`):
            The embedding dimension of inputs. It must be divisible by 16.
        spatial_size (`int` or `Tuple[int, int]`):
            The spatial dimension of positional embeddings. If an integer is provided, the same size is applied to both
            spatial dimensions (height and width).
        temporal_size (`int`):
            The temporal dimension of positional embeddings (number of frames).
        spatial_interpolation_scale (`float`, defaults to 1.0):
            Scale factor for spatial grid interpolation.
        temporal_interpolation_scale (`float`, defaults to 1.0):
            Scale factor for temporal grid interpolation.

    Returns:
        `ms.Tensor`:
            The 3D sinusoidal positional embeddings of shape `[temporal_size, spatial_size[0] * spatial_size[1],
            embed_dim]`.
    """
    if output_type == "np":
        return _get_3d_sincos_pos_embed_np(
            embed_dim=embed_dim,
            spatial_size=spatial_size,
            temporal_size=temporal_size,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
        )
    if embed_dim % 4 != 0:
        raise ValueError("`embed_dim` must be divisible by 4")
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)

    embed_dim_spatial = 3 * embed_dim // 4
    embed_dim_temporal = embed_dim // 4

    # 1. Spatial
    grid_h = mint.arange(spatial_size[1], dtype=ms.float32) / spatial_interpolation_scale
    grid_w = mint.arange(spatial_size[0], dtype=ms.float32) / spatial_interpolation_scale
    grid = mint.meshgrid(grid_w, grid_h, indexing="xy")  # here w goes first
    grid = mint.stack(grid, dim=0)

    grid = grid.reshape([2, 1, spatial_size[1], spatial_size[0]])
    pos_embed_spatial = get_2d_sincos_pos_embed_from_grid(embed_dim_spatial, grid, output_type="ms")

    # 2. Temporal
    grid_t = mint.arange(temporal_size, dtype=ms.float32) / temporal_interpolation_scale
    pos_embed_temporal = get_1d_sincos_pos_embed_from_grid(embed_dim_temporal, grid_t, output_type="ms")

    # 3. Concat
    pos_embed_spatial = pos_embed_spatial[None, :, :]
    if pos_embed_spatial.dtype == ms.float64:
        pos_embed_spatial = (
            pos_embed_spatial.to(ms.float32)
            .repeat_interleave(temporal_size, dim=0, output_size=pos_embed_spatial.shape[0] * temporal_size)
            .to(ms.float64)
        )  # [T, H*W, D // 4 * 3]
    else:
        pos_embed_spatial = pos_embed_spatial.repeat_interleave(
            temporal_size, dim=0, output_size=pos_embed_spatial.shape[0] * temporal_size
        )  # [T, H*W, D // 4 * 3]

    pos_embed_temporal = pos_embed_temporal[:, None, :]
    if pos_embed_spatial.dtype == ms.float64:
        pos_embed_temporal = (
            pos_embed_temporal.to(ms.float32).repeat_interleave(spatial_size[0] * spatial_size[1], dim=1).to(ms.float64)
        )  # [T, H*W, D // 4]
    else:
        pos_embed_temporal = pos_embed_temporal.repeat_interleave(
            spatial_size[0] * spatial_size[1], dim=1
        )  # [T, H*W, D // 4]

    pos_embed = mint.concat([pos_embed_temporal, pos_embed_spatial], dim=-1)  # [T, H*W, D]
    return pos_embed


def _get_3d_sincos_pos_embed_np(
    embed_dim: int,
    spatial_size: Union[int, Tuple[int, int]],
    temporal_size: int,
    spatial_interpolation_scale: float = 1.0,
    temporal_interpolation_scale: float = 1.0,
) -> np.ndarray:
    r"""
    Creates 3D sinusoidal positional embeddings.

    Args:
        embed_dim (`int`):
            The embedding dimension of inputs. It must be divisible by 16.
        spatial_size (`int` or `Tuple[int, int]`):
            The spatial dimension of positional embeddings. If an integer is provided, the same size is applied to both
            spatial dimensions (height and width).
        temporal_size (`int`):
            The temporal dimension of positional embeddings (number of frames).
        spatial_interpolation_scale (`float`, defaults to 1.0):
            Scale factor for spatial grid interpolation.
        temporal_interpolation_scale (`float`, defaults to 1.0):
            Scale factor for temporal grid interpolation.

    Returns:
        `np.ndarray`:
            The 3D sinusoidal positional embeddings of shape `[temporal_size, spatial_size[0] * spatial_size[1],
            embed_dim]`.
    """
    deprecation_message = (
        "`get_3d_sincos_pos_embed` uses `mindspore`."
        " `from_numpy` is no longer required."
        "  Pass `output_type='ms' to use the new version now."
    )
    deprecate("output_type=='np'", "0.33.0", deprecation_message, standard_warn=False)
    if embed_dim % 4 != 0:
        raise ValueError("`embed_dim` must be divisible by 4")
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)

    embed_dim_spatial = 3 * embed_dim // 4
    embed_dim_temporal = embed_dim // 4

    # 1. Spatial
    grid_h = np.arange(spatial_size[1], dtype=np.float32) / spatial_interpolation_scale
    grid_w = np.arange(spatial_size[0], dtype=np.float32) / spatial_interpolation_scale
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, spatial_size[1], spatial_size[0]])
    pos_embed_spatial = get_2d_sincos_pos_embed_from_grid(embed_dim_spatial, grid)

    # 2. Temporal
    grid_t = np.arange(temporal_size, dtype=np.float32) / temporal_interpolation_scale
    pos_embed_temporal = get_1d_sincos_pos_embed_from_grid(embed_dim_temporal, grid_t)

    # 3. Concat
    pos_embed_spatial = pos_embed_spatial[np.newaxis, :, :]
    pos_embed_spatial = np.repeat(pos_embed_spatial, temporal_size, axis=0)  # [T, H*W, D // 4 * 3]

    pos_embed_temporal = pos_embed_temporal[:, np.newaxis, :]
    pos_embed_temporal = np.repeat(pos_embed_temporal, spatial_size[0] * spatial_size[1], axis=1)  # [T, H*W, D // 4]

    pos_embed = np.concatenate([pos_embed_temporal, pos_embed_spatial], axis=-1)  # [T, H*W, D]
    return pos_embed


def get_2d_sincos_pos_embed(
    embed_dim,
    grid_size,
    cls_token=False,
    extra_tokens=0,
    interpolation_scale=1.0,
    base_size=16,
    output_type: str = "np",
):
    """
    Creates 2D sinusoidal positional embeddings.

    Args:
        embed_dim (`int`):
            The embedding dimension.
        grid_size (`int`):
            The size of the grid height and width.
        cls_token (`bool`, defaults to `False`):
            Whether or not to add a classification token.
        extra_tokens (`int`, defaults to `0`):
            The number of extra tokens to add.
        interpolation_scale (`float`, defaults to `1.0`):
            The scale of the interpolation.

    Returns:
        pos_embed (`ms.Tensor`):
            Shape is either `[grid_size * grid_size, embed_dim]` if not using cls_token, or `[1 + grid_size*grid_size,
            embed_dim]` if using cls_token
    """
    if output_type == "np":
        deprecation_message = (
            "`get_2d_sincos_pos_embed` uses `mindspore`."
            " `from_numpy` is no longer required."
            "  Pass `output_type='ms' to use the new version now."
        )
        deprecate("output_type=='np'", "0.33.0", deprecation_message, standard_warn=False)
        return get_2d_sincos_pos_embed_np(
            embed_dim=embed_dim,
            grid_size=grid_size,
            cls_token=cls_token,
            extra_tokens=extra_tokens,
            interpolation_scale=interpolation_scale,
            base_size=base_size,
        )
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)

    grid_h = mint.arange(grid_size[0], dtype=ms.float32) / (grid_size[0] / base_size) / interpolation_scale
    grid_w = mint.arange(grid_size[1], dtype=ms.float32) / (grid_size[1] / base_size) / interpolation_scale
    grid = mint.meshgrid(grid_w, grid_h, indexing="xy")  # here w goes first
    grid = mint.stack(grid, dim=0)

    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid, output_type=output_type)
    if cls_token and extra_tokens > 0:
        pos_embed = mint.concat([mint.zeros([extra_tokens, embed_dim]), pos_embed], dim=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid, output_type="np"):
    r"""
    This function generates 2D sinusoidal positional embeddings from a grid.

    Args:
        embed_dim (`int`): The embedding dimension.
        grid (`ms.Tensor`): Grid of positions with shape `(H * W,)`.

    Returns:
        `ms.Tensor`: The 2D sinusoidal positional embeddings with shape `(H * W, embed_dim)`
    """
    if output_type == "np":
        deprecation_message = (
            "`get_2d_sincos_pos_embed_from_grid` uses `mindspore`."
            " `from_numpy` is no longer required."
            "  Pass `output_type='ms' to use the new version now."
        )
        deprecate("output_type=='np'", "0.33.0", deprecation_message, standard_warn=False)
        return get_2d_sincos_pos_embed_from_grid_np(
            embed_dim=embed_dim,
            grid=grid,
        )
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0], output_type=output_type)  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1], output_type=output_type)  # (H*W, D/2)

    emb = mint.concat([emb_h, emb_w], dim=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos, output_type="np"):
    """
    This function generates 1D positional embeddings from a grid.

    Args:
        embed_dim (`int`): The embedding dimension `D`
        pos (`ms.Tensor`): 1D tensor of positions with shape `(M,)`

    Returns:
        `ms.Tensor`: Sinusoidal positional embeddings of shape `(M, D)`.
    """
    if output_type == "np":
        deprecation_message = (
            "`get_1d_sincos_pos_embed_from_grid` uses `mindspore`."
            " `from_numpy` is no longer required."
            "  Pass `output_type='ms' to use the new version now."
        )
        deprecate("output_type=='np'", "0.34.0", deprecation_message, standard_warn=False)
        return get_1d_sincos_pos_embed_from_grid_np(embed_dim=embed_dim, pos=pos)
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    omega = mint.arange(embed_dim // 2, dtype=ms.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = mint.reshape(pos, (-1,))  # (M,)
    out = mint.outer(pos, omega)  # (M, D/2), outer product

    emb_sin = mint.sin(out)  # (M, D/2)
    emb_cos = mint.cos(out)  # (M, D/2)

    emb = mint.concat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed_np(
    embed_dim, grid_size, cls_token=False, extra_tokens=0, interpolation_scale=1.0, base_size=16
):
    """
    Creates 2D sinusoidal positional embeddings.

    Args:
        embed_dim (`int`):
            The embedding dimension.
        grid_size (`int`):
            The size of the grid height and width.
        cls_token (`bool`, defaults to `False`):
            Whether or not to add a classification token.
        extra_tokens (`int`, defaults to `0`):
            The number of extra tokens to add.
        interpolation_scale (`float`, defaults to `1.0`):
            The scale of the interpolation.

    Returns:
        pos_embed (`np.ndarray`):
            Shape is either `[grid_size * grid_size, embed_dim]` if not using cls_token, or `[1 + grid_size*grid_size,
            embed_dim]` if using cls_token
    """
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)

    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0] / base_size) / interpolation_scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1] / base_size) / interpolation_scale
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid_np(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid_np(embed_dim, grid):
    r"""
    This function generates 2D sinusoidal positional embeddings from a grid.

    Args:
        embed_dim (`int`): The embedding dimension.
        grid (`np.ndarray`): Grid of positions with shape `(H * W,)`.

    Returns:
        `np.ndarray`: The 2D sinusoidal positional embeddings with shape `(H * W, embed_dim)`
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid_np(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid_np(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid_np(embed_dim, pos):
    """
    This function generates 1D positional embeddings from a grid.

    Args:
        embed_dim (`int`): The embedding dimension `D`
        pos (`numpy.ndarray`): 1D tensor of positions with shape `(M,)`

    Returns:
        `numpy.ndarray`: Sinusoidal positional embeddings of shape `(M, D)`.
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class PatchEmbed(nn.Cell):
    """
    2D Image to Patch Embedding with support for SD3 cropping.

    Args:
        height (`int`, defaults to `224`): The height of the image.
        width (`int`, defaults to `224`): The width of the image.
        patch_size (`int`, defaults to `16`): The size of the patches.
        in_channels (`int`, defaults to `3`): The number of input channels.
        embed_dim (`int`, defaults to `768`): The output dimension of the embedding.
        layer_norm (`bool`, defaults to `False`): Whether or not to use layer normalization.
        flatten (`bool`, defaults to `True`): Whether or not to flatten the output.
        bias (`bool`, defaults to `True`): Whether or not to use bias.
        interpolation_scale (`float`, defaults to `1`): The scale of the interpolation.
        pos_embed_type (`str`, defaults to `"sincos"`): The type of positional embedding.
        pos_embed_max_size (`int`, defaults to `None`): The maximum size of the positional embedding.
    """

    def __init__(
        self,
        height=224,
        width=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        layer_norm=False,
        flatten=True,
        bias=True,
        interpolation_scale=1,
        pos_embed_type="sincos",
        pos_embed_max_size=None,  # For SD3 cropping
        zero_module=False,  # For SD3 ControlNet
    ):
        super().__init__()
        num_patches = (height // patch_size) * (width // patch_size)
        self.flatten = flatten
        self.layer_norm = layer_norm
        self.pos_embed_max_size = pos_embed_max_size

        # weight_init_kwargs = {"weight_init": "zeros", "bias_init": "zeros"} if zero_module else {}
        self.proj = mint.nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=patch_size,
            bias=bias,
        )
        if zero_module:
            self.proj.weight.set_data(init.initializer("zeros", self.proj.weight.shape, self.proj.weight.dtype))
            self.proj.bias.set_data(init.initializer("zeros", self.proj.bias.shape, self.proj.bias.dtype))

        if layer_norm:
            self.norm = mint.nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm = None

        self.patch_size = patch_size
        self.height, self.width = height // patch_size, width // patch_size
        self.base_size = height // patch_size
        self.interpolation_scale = interpolation_scale

        # Calculate positional embeddings based on max size or default
        if pos_embed_max_size:
            grid_size = pos_embed_max_size
        else:
            grid_size = int(num_patches**0.5)

        if pos_embed_type is None:
            self.pos_embed = None
        elif pos_embed_type == "sincos":
            pos_embed = get_2d_sincos_pos_embed(
                embed_dim,
                grid_size,
                base_size=self.base_size,
                interpolation_scale=self.interpolation_scale,
                output_type="ms",
            )
            persistent = True if pos_embed_max_size else False
            if persistent:
                self.pos_embed = ms.Parameter(pos_embed.float().unsqueeze(0), name="pos_embed")
            else:
                self.pos_embed = pos_embed.float().unsqueeze(0)
        else:
            raise ValueError(f"Unsupported pos_embed_type: {pos_embed_type}")

    def cropped_pos_embed(self, height, width):
        """Crops positional embeddings for SD3 compatibility."""
        if self.pos_embed_max_size is None:
            raise ValueError("`pos_embed_max_size` must be set for cropping.")

        height = height // self.patch_size
        width = width // self.patch_size
        if height > self.pos_embed_max_size:
            raise ValueError(
                f"Height ({height}) cannot be greater than `pos_embed_max_size`: {self.pos_embed_max_size}."
            )
        if width > self.pos_embed_max_size:
            raise ValueError(f"Width ({width}) cannot be greater than `pos_embed_max_size`: {self.pos_embed_max_size}.")

        top = (self.pos_embed_max_size - height) // 2
        left = (self.pos_embed_max_size - width) // 2
        spatial_pos_embed = self.pos_embed.reshape(1, self.pos_embed_max_size, self.pos_embed_max_size, -1)
        spatial_pos_embed = spatial_pos_embed[:, top : top + height, left : left + width, :]
        spatial_pos_embed = spatial_pos_embed.reshape(1, -1, spatial_pos_embed.shape[-1])
        return spatial_pos_embed

    def construct(self, latent):
        if self.pos_embed_max_size is not None:
            height, width = latent.shape[-2:]
        else:
            height, width = latent.shape[-2] // self.patch_size, latent.shape[-1] // self.patch_size
        latent = self.proj(latent)
        if self.flatten:
            latent = latent.flatten(start_dim=2).swapaxes(1, 2)  # BCHW -> BNC
        if self.layer_norm:
            latent = self.norm(latent)
        if self.pos_embed is None:
            return latent.to(latent.dtype)
        # Interpolate or crop positional embeddings as needed
        if self.pos_embed_max_size:
            pos_embed = self.cropped_pos_embed(height, width)
        else:
            if self.height != height or self.width != width:
                pos_embed = get_2d_sincos_pos_embed(
                    embed_dim=self.pos_embed.shape[-1],
                    grid_size=(height, width),
                    base_size=self.base_size,
                    interpolation_scale=self.interpolation_scale,
                    output_type="ms",
                )
                pos_embed = pos_embed.float().unsqueeze(0)
            else:
                pos_embed = self.pos_embed

        return (latent + pos_embed).to(latent.dtype)


class LuminaPatchEmbed(nn.Cell):
    """
    2D Image to Patch Embedding with support for Lumina-T2X

    Args:
        patch_size (`int`, defaults to `2`): The size of the patches.
        in_channels (`int`, defaults to `4`): The number of input channels.
        embed_dim (`int`, defaults to `768`): The output dimension of the embedding.
        bias (`bool`, defaults to `True`): Whether or not to use bias.
    """

    def __init__(self, patch_size=2, in_channels=4, embed_dim=768, bias=True):
        super().__init__()
        self.patch_size = patch_size
        self.proj = mint.nn.Linear(patch_size * patch_size * in_channels, embed_dim, bias)

    def construct(self, x, freqs_cis):
        """
        Patchifies and embeds the input tensor(s).
        Args:
            x (List[ms.Tensor] | ms.Tensor): The input tensor(s) to be patchified and embedded.
        Returns:
            Tuple[ms.Tensor, ms.Tensor, List[Tuple[int, int]], ms.Tensor]: A tuple containing the patchified
            and embedded tensor(s), the mask indicating the valid patches, the original image size(s), and the
            frequency tensor(s).
        """
        patch_height = patch_width = self.patch_size
        batch_size, channel, height, width = x.shape
        height_tokens, width_tokens = height // patch_height, width // patch_width

        x = x.view(batch_size, channel, height_tokens, patch_height, width_tokens, patch_width).permute(
            0, 2, 4, 1, 3, 5
        )
        x = x.flatten(start_dim=3)
        x = self.proj(x)
        x = x.flatten(start_dim=1, end_dim=2)

        mask = mint.ones((x.shape[0], x.shape[1]), dtype=ms.int32)

        return (
            x,
            mask,
            [(height, width)] * batch_size,
            freqs_cis[:height_tokens, :width_tokens].flatten(start_dim=0, end_dim=1).unsqueeze(0),
        )


class CogVideoXPatchEmbed(nn.Cell):
    def __init__(
        self,
        patch_size: int = 2,
        patch_size_t: Optional[int] = None,
        in_channels: int = 16,
        embed_dim: int = 1920,
        text_embed_dim: int = 4096,
        bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_positional_embeddings: bool = True,
        use_learned_positional_embeddings: bool = True,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.embed_dim = embed_dim
        self.sample_height = sample_height
        self.sample_width = sample_width
        self.sample_frames = sample_frames
        self.temporal_compression_ratio = temporal_compression_ratio
        self.max_text_seq_length = max_text_seq_length
        self.spatial_interpolation_scale = spatial_interpolation_scale
        self.temporal_interpolation_scale = temporal_interpolation_scale
        self.use_positional_embeddings = use_positional_embeddings
        self.use_learned_positional_embeddings = use_learned_positional_embeddings

        if patch_size_t is None:
            # CogVideoX 1.0 checkpoints
            self.proj = mint.nn.Conv2d(
                in_channels,
                embed_dim,
                kernel_size=(patch_size, patch_size),
                stride=patch_size,
                bias=bias,
            )
        else:
            # CogVideoX 1.5 checkpoints
            self.proj = mint.nn.Linear(in_channels * patch_size * patch_size * patch_size_t, embed_dim)

        self.text_proj = mint.nn.Linear(text_embed_dim, embed_dim)

        if use_positional_embeddings or use_learned_positional_embeddings:
            persistent = use_learned_positional_embeddings
            pos_embedding = self._get_positional_embeddings(sample_height, sample_width, sample_frames)
            self.pos_embedding = (
                ms.Parameter(pos_embedding, name="pos_embedding", requires_grad=False) if persistent else pos_embedding
            )

    def _get_positional_embeddings(self, sample_height: int, sample_width: int, sample_frames: int) -> ms.Tensor:
        post_patch_height = sample_height // self.patch_size
        post_patch_width = sample_width // self.patch_size
        post_time_compression_frames = (sample_frames - 1) // self.temporal_compression_ratio + 1
        num_patches = post_patch_height * post_patch_width * post_time_compression_frames

        pos_embedding = get_3d_sincos_pos_embed(
            self.embed_dim,
            (post_patch_width, post_patch_height),
            post_time_compression_frames,
            self.spatial_interpolation_scale,
            self.temporal_interpolation_scale,
            output_type="ms",
        )
        pos_embedding = pos_embedding.flatten(start_dim=0, end_dim=1)
        joint_pos_embedding = mint.zeros(size=(1, self.max_text_seq_length + num_patches, self.embed_dim))
        joint_pos_embedding[:, self.max_text_seq_length :] += pos_embedding

        return joint_pos_embedding

    def construct(self, text_embeds: ms.Tensor, image_embeds: ms.Tensor):
        r"""
        Args:
            text_embeds (`ms.Tensor`):
                Input text embeddings. Expected shape: (batch_size, seq_length, embedding_dim).
            image_embeds (`ms.Tensor`):
                Input image embeddings. Expected shape: (batch_size, num_frames, channels, height, width).
        """
        text_embeds = self.text_proj(text_embeds)

        batch_size, num_frames, channels, height, width = image_embeds.shape

        if self.patch_size_t is None:
            image_embeds = image_embeds.reshape(-1, channels, height, width)
            image_embeds = self.proj(image_embeds)
            image_embeds = image_embeds.view(batch_size, num_frames, *image_embeds.shape[1:])
            image_embeds = image_embeds.flatten(start_dim=3).swapaxes(
                2, 3
            )  # [batch, num_frames, height x width, channels]
            image_embeds = image_embeds.flatten(
                start_dim=1, end_dim=2
            )  # [batch, num_frames x height x width, channels]
        else:
            p = self.patch_size
            p_t = self.patch_size_t

            image_embeds = image_embeds.permute(0, 1, 3, 4, 2)
            image_embeds = image_embeds.reshape(
                batch_size, num_frames // p_t, p_t, height // p, p, width // p, p, channels
            )
            image_embeds = (
                image_embeds.permute(0, 1, 3, 5, 7, 2, 4, 6)
                .flatten(start_dim=4, end_dim=7)
                .flatten(start_dim=1, end_dim=3)
            )
            image_embeds = self.proj(image_embeds)

        embeds = mint.cat(
            [text_embeds, image_embeds], dim=1
        ).contiguous()  # [batch, seq_length + num_frames x height x width, channels]

        if self.use_positional_embeddings or self.use_learned_positional_embeddings:
            if self.use_learned_positional_embeddings and (self.sample_width != width or self.sample_height != height):
                raise ValueError(
                    "It is currently not possible to generate videos at a different resolution that the defaults. This should only be the case with 'THUDM/CogVideoX-5b-I2V'."  # noqa: E501
                    "If you think this is incorrect, please open an issue at https://github.com/huggingface/diffusers/issues."
                )

            pre_time_compression_frames = (num_frames - 1) * self.temporal_compression_ratio + 1

            if (
                self.sample_height != height
                or self.sample_width != width
                or self.sample_frames != pre_time_compression_frames
            ):
                pos_embedding = self._get_positional_embeddings(height, width, pre_time_compression_frames)
            else:
                pos_embedding = self.pos_embedding

            pos_embedding = pos_embedding.to(dtype=embeds.dtype)
            embeds = embeds + pos_embedding

        return embeds


class CogView3PlusPatchEmbed(nn.Cell):
    def __init__(
        self,
        in_channels: int = 16,
        hidden_size: int = 2560,
        patch_size: int = 2,
        text_hidden_size: int = 4096,
        pos_embed_max_size: int = 128,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.text_hidden_size = text_hidden_size
        self.pos_embed_max_size = pos_embed_max_size
        # Linear projection for image patches
        self.proj = mint.nn.Linear(in_channels * patch_size**2, hidden_size)

        # Linear projection for text embeddings
        self.text_proj = mint.nn.Linear(text_hidden_size, hidden_size)

        pos_embed = get_2d_sincos_pos_embed(
            hidden_size, pos_embed_max_size, base_size=pos_embed_max_size, output_type="ms"
        )
        pos_embed = pos_embed.reshape(pos_embed_max_size, pos_embed_max_size, hidden_size)
        self.pos_embed = pos_embed.float()

    def construct(self, hidden_states: ms.Tensor, encoder_hidden_states: ms.Tensor) -> ms.Tensor:
        batch_size, channel, height, width = hidden_states.shape

        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError("Height and width must be divisible by patch size")

        height = height // self.patch_size
        width = width // self.patch_size
        hidden_states = hidden_states.view(batch_size, channel, height, self.patch_size, width, self.patch_size)
        hidden_states = hidden_states.permute(0, 2, 4, 1, 3, 5).contiguous()
        hidden_states = hidden_states.view(batch_size, height * width, channel * self.patch_size * self.patch_size)

        # Project the patches
        hidden_states = self.proj(hidden_states)
        encoder_hidden_states = self.text_proj(encoder_hidden_states)
        hidden_states = mint.cat([encoder_hidden_states, hidden_states], dim=1)

        # Calculate text_length
        text_length = encoder_hidden_states.shape[1]

        image_pos_embed = self.pos_embed[:height, :width].reshape(height * width, -1)
        text_pos_embed = mint.zeros((text_length, self.hidden_size), dtype=image_pos_embed.dtype)
        pos_embed = mint.cat([text_pos_embed, image_pos_embed], dim=0)[None, ...]

        return (hidden_states + pos_embed).to(hidden_states.dtype)


def get_3d_rotary_pos_embed(
    embed_dim,
    crops_coords,
    grid_size,
    temporal_size,
    theta: int = 10000,
    use_real: bool = True,
    grid_type: str = "linspace",
    max_size: Optional[Tuple[int, int]] = None,
) -> Union[ms.Tensor, Tuple[ms.Tensor, ms.Tensor]]:
    """
    RoPE for video tokens with 3D structure.

    Args:
    embed_dim: (`int`):
        The embedding dimension size, corresponding to hidden_size_head.
    crops_coords (`Tuple[int]`):
        The top-left and bottom-right coordinates of the crop.
    grid_size (`Tuple[int]`):
        The grid size of the spatial positional embedding (height, width).
    temporal_size (`int`):
        The size of the temporal dimension.
    theta (`float`):
        Scaling factor for frequency computation.
    grid_type (`str`):
        Whether to use "linspace" or "slice" to compute grids.

    Returns:
        `ms.Tensor`: positional embedding with shape `(temporal_size * grid_size[0] * grid_size[1], embed_dim/2)`.
    """
    if use_real is not True:
        raise ValueError(" `use_real = False` is not currently supported for get_3d_rotary_pos_embed")

    if grid_type == "linspace":
        start, stop = crops_coords
        grid_size_h, grid_size_w = grid_size
        grid_h = mint.linspace(start[0], stop[0] * (grid_size_h - 1) / grid_size_h, grid_size_h).to(ms.float32)
        grid_w = mint.linspace(start[1], stop[1] * (grid_size_w - 1) / grid_size_w, grid_size_w).to(ms.float32)
        grid_t = mint.arange(temporal_size, dtype=ms.float32)
        grid_t = mint.linspace(0, temporal_size * (temporal_size - 1) / temporal_size, temporal_size).to(ms.float32)
    elif grid_type == "slice":
        max_h, max_w = max_size
        grid_size_h, grid_size_w = grid_size
        grid_h = mint.arange(max_h, dtype=ms.float32)
        grid_w = mint.arange(max_w, dtype=ms.float32)
        grid_t = mint.arange(temporal_size, dtype=ms.float32)
    else:
        raise ValueError("Invalid value passed for `grid_type`.")

    # Compute dimensions for each axis
    dim_t = embed_dim // 4
    dim_h = embed_dim // 8 * 3
    dim_w = embed_dim // 8 * 3

    # Temporal frequencies
    freqs_t = get_1d_rotary_pos_embed(dim_t, grid_t, theta=theta, use_real=True)
    # Spatial frequencies for height and width
    freqs_h = get_1d_rotary_pos_embed(dim_h, grid_h, theta=theta, use_real=True)
    freqs_w = get_1d_rotary_pos_embed(dim_w, grid_w, theta=theta, use_real=True)

    # BroadCast and concatenate temporal and spaial frequencie (height and width) into a 3d tensor
    def combine_time_height_width(freqs_t, freqs_h, freqs_w):
        freqs_t = freqs_t[:, None, None, :].broadcast_to(
            (-1, grid_size_h, grid_size_w, -1)
        )  # temporal_size, grid_size_h, grid_size_w, dim_t
        freqs_h = freqs_h[None, :, None, :].broadcast_to(
            (temporal_size, -1, grid_size_w, -1)
        )  # temporal_size, grid_size_h, grid_size_2, dim_h
        freqs_w = freqs_w[None, None, :, :].broadcast_to(
            (temporal_size, grid_size_h, -1, -1)
        )  # temporal_size, grid_size_h, grid_size_2, dim_w

        freqs = mint.cat(
            [freqs_t, freqs_h, freqs_w], dim=-1
        )  # temporal_size, grid_size_h, grid_size_w, (dim_t + dim_h + dim_w)
        freqs = freqs.view(
            temporal_size * grid_size_h * grid_size_w, -1
        )  # (temporal_size * grid_size_h * grid_size_w), (dim_t + dim_h + dim_w)
        return freqs

    t_cos, t_sin = freqs_t  # both t_cos and t_sin has shape: temporal_size, dim_t
    h_cos, h_sin = freqs_h  # both h_cos and h_sin has shape: grid_size_h, dim_h
    w_cos, w_sin = freqs_w  # both w_cos and w_sin has shape: grid_size_w, dim_w

    if grid_type == "slice":
        t_cos, t_sin = t_cos[:temporal_size], t_sin[:temporal_size]
        h_cos, h_sin = h_cos[:grid_size_h], h_sin[:grid_size_h]
        w_cos, w_sin = w_cos[:grid_size_w], w_sin[:grid_size_w]

    cos = combine_time_height_width(t_cos, h_cos, w_cos)
    sin = combine_time_height_width(t_sin, h_sin, w_sin)
    return cos, sin


def get_3d_rotary_pos_embed_allegro(
    embed_dim,
    crops_coords,
    grid_size,
    temporal_size,
    interpolation_scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    theta: int = 10000,
) -> Union[ms.Tensor, Tuple[ms.Tensor, ms.Tensor]]:
    # TODO(aryan): docs
    start, stop = crops_coords
    grid_size_h, grid_size_w = grid_size
    interpolation_scale_t, interpolation_scale_h, interpolation_scale_w = interpolation_scale
    grid_t = mint.linspace(0, temporal_size * (temporal_size - 1) / temporal_size, temporal_size).float()
    grid_h = mint.linspace(start[0], stop[0] * (grid_size_h - 1) / grid_size_h, grid_size_h).float()
    grid_w = mint.linspace(start[1], stop[1] * (grid_size_w - 1) / grid_size_w, grid_size_w).float()

    # Compute dimensions for each axis
    dim_t = embed_dim // 3
    dim_h = embed_dim // 3
    dim_w = embed_dim // 3

    # Temporal frequencies
    freqs_t = get_1d_rotary_pos_embed(
        dim_t, grid_t / interpolation_scale_t, theta=theta, use_real=True, repeat_interleave_real=False
    )
    # Spatial frequencies for height and width
    freqs_h = get_1d_rotary_pos_embed(
        dim_h, grid_h / interpolation_scale_h, theta=theta, use_real=True, repeat_interleave_real=False
    )
    freqs_w = get_1d_rotary_pos_embed(
        dim_w, grid_w / interpolation_scale_w, theta=theta, use_real=True, repeat_interleave_real=False
    )

    return freqs_t, freqs_h, freqs_w, grid_t, grid_h, grid_w


def get_2d_rotary_pos_embed(embed_dim, crops_coords, grid_size, use_real=True, output_type: str = "np"):
    """
    RoPE for image tokens with 2d structure.

    Args:
    embed_dim: (`int`):
        The embedding dimension size
    crops_coords (`Tuple[int]`)
        The top-left and bottom-right coordinates of the crop.
    grid_size (`Tuple[int]`):
        The grid size of the positional embedding.
    use_real (`bool`):
        If True, return real part and imaginary part separately. Otherwise, return complex numbers.

    Returns:
        `ms.Tensor`: positional embedding with shape `( grid_size * grid_size, embed_dim/2)`.
    """
    if output_type == "np":
        deprecation_message = (
            "`get_2d_sincos_pos_embed` uses `mindspore`."
            " `from_numpy` is no longer required."
            "  Pass `output_type='pt' to use the new version now."
        )
        deprecate("output_type=='np'", "0.33.0", deprecation_message, standard_warn=False)
        return _get_2d_rotary_pos_embed_np(
            embed_dim=embed_dim,
            crops_coords=crops_coords,
            grid_size=grid_size,
            use_real=use_real,
        )
    start, stop = crops_coords
    # scale end by (steps−1)/steps matches np.linspace(..., endpoint=False)
    grid_h = mint.linspace(start[0], stop[0] * (grid_size[0] - 1) / grid_size[0], grid_size[0]).to(ms.float32)
    grid_w = mint.linspace(start[1], stop[1] * (grid_size[1] - 1) / grid_size[1], grid_size[1]).to(ms.float32)
    grid = mint.meshgrid(grid_w, grid_h, indexing="xy")
    grid = mint.stack(grid, dim=0)  # [2, W, H]

    grid = grid.reshape([2, 1, *grid.shape[1:]])
    pos_embed = get_2d_rotary_pos_embed_from_grid(embed_dim, grid, use_real=use_real)
    return pos_embed


def _get_2d_rotary_pos_embed_np(embed_dim, crops_coords, grid_size, use_real=True):
    """
    RoPE for image tokens with 2d structure.

    Args:
    embed_dim: (`int`):
        The embedding dimension size
    crops_coords (`Tuple[int]`)
        The top-left and bottom-right coordinates of the crop.
    grid_size (`Tuple[int]`):
        The grid size of the positional embedding.
    use_real (`bool`):
        If True, return real part and imaginary part separately. Otherwise, return complex numbers.

    Returns:
        `ms.Tensor`: positional embedding with shape `( grid_size * grid_size, embed_dim/2)`.
    """
    start, stop = crops_coords
    grid_h = np.linspace(start[0], stop[0], grid_size[0], endpoint=False, dtype=np.float32)
    grid_w = np.linspace(start[1], stop[1], grid_size[1], endpoint=False, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)  # [2, W, H]

    grid = grid.reshape([2, 1, *grid.shape[1:]])
    pos_embed = get_2d_rotary_pos_embed_from_grid(embed_dim, grid, use_real=use_real)
    return pos_embed


def get_2d_rotary_pos_embed_from_grid(embed_dim, grid, use_real=False):
    """
    Get 2D RoPE from grid.

    Args:
    embed_dim: (`int`):
        The embedding dimension size, corresponding to hidden_size_head.
    grid (`np.ndarray`):
        The grid of the positional embedding.
    use_real (`bool`):
        If True, return real part and imaginary part separately. Otherwise, return complex numbers.

    Returns:
        `ms.Tensor`: positional embedding with shape `( grid_size * grid_size, embed_dim/2)`.
    """
    assert embed_dim % 4 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_rotary_pos_embed(
        embed_dim // 2, grid[0].reshape(-1), use_real=use_real
    )  # (H*W, D/2) if use_real else (H*W, D/4)
    emb_w = get_1d_rotary_pos_embed(
        embed_dim // 2, grid[1].reshape(-1), use_real=use_real
    )  # (H*W, D/2) if use_real else (H*W, D/4)

    if use_real:
        cos = mint.cat([emb_h[0], emb_w[0]], dim=1)  # (H*W, D)
        sin = mint.cat([emb_h[1], emb_w[1]], dim=1)  # (H*W, D)
        return cos, sin
    else:
        emb = mint.cat([emb_h, emb_w], dim=1)  # (H*W, D/2)
        return emb


def get_2d_rotary_pos_embed_lumina(embed_dim, len_h, len_w, linear_factor=1.0, ntk_factor=1.0):
    """
    Get 2D RoPE from grid.

    Args:
    embed_dim: (`int`):
        The embedding dimension size, corresponding to hidden_size_head.
    grid (`np.ndarray`):
        The grid of the positional embedding.
    linear_factor (`float`):
        The linear factor of the positional embedding, which is used to scale the positional embedding in the linear
        layer.
    ntk_factor (`float`):
        The ntk factor of the positional embedding, which is used to scale the positional embedding in the ntk layer.

    Returns:
        `ms.Tensor`: positional embedding with shape `( grid_size * grid_size, embed_dim/2)`.
    """
    assert embed_dim % 4 == 0

    emb_h = get_1d_rotary_pos_embed(
        embed_dim // 2, len_h, linear_factor=linear_factor, ntk_factor=ntk_factor
    )  # (H, D/4)
    emb_w = get_1d_rotary_pos_embed(
        embed_dim // 2, len_w, linear_factor=linear_factor, ntk_factor=ntk_factor
    )  # (W, D/4)
    emb_h = emb_h.view(len_h, 1, embed_dim // 4, 1).tile((1, len_w, 1, 1))  # (H, W, D/4, 1)
    emb_w = emb_w.view(1, len_w, embed_dim // 4, 1).tile((len_h, 1, 1, 1))  # (H, W, D/4, 1)

    emb = mint.cat([emb_h, emb_w], dim=-1).flatten(start_dim=2)  # (H, W, D/2)
    return emb


def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[np.ndarray, int],
    theta: float = 10000.0,
    use_real=False,
    linear_factor=1.0,
    ntk_factor=1.0,
    repeat_interleave_real=True,
    freqs_dtype=ms.float32,  # ms.float32, ms.float64 (flux)
):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
    index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
    data type.

    Args:
        dim (`int`): Dimension of the frequency tensor.
        pos (`np.ndarray` or `int`): Position indices for the frequency tensor. [S] or scalar
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (`bool`, *optional*):
            If True, return real part and imaginary part separately. Otherwise, return complex numbers.
        linear_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the context extrapolation. Defaults to 1.0.
        ntk_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the NTK-Aware RoPE. Defaults to 1.0.
        repeat_interleave_real (`bool`, *optional*, defaults to `True`):
            If `True` and `use_real`, real part and imaginary part are each interleaved with themselves to reach `dim`.
            Otherwise, they are concateanted with themselves.
        freqs_dtype (`ms.float32` or `ms.float64`, *optional*, defaults to `ms.float32`):
            the dtype of the frequency tensor.
    Returns:
        `ms.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
    """
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = mint.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = ms.Tensor.from_numpy(pos)  # type: ignore  # [S]

    theta = theta * ntk_factor
    freqs = 1.0 / (theta ** (mint.arange(0, dim, 2, dtype=freqs_dtype) / dim)) / linear_factor  # [D/2]
    freqs = mint.outer(pos, freqs)  # type: ignore   # [S, D/2]
    freqs = freqs.float()
    if use_real and repeat_interleave_real:
        # flux, hunyuan-dit, cogvideox
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()  # [S, D]
        return freqs_cos, freqs_sin
    elif use_real:
        # stable audio, allegro
        freqs_cos = mint.cat([freqs.cos(), freqs.cos()], dim=-1).float()  # [S, D]
        freqs_sin = mint.cat([freqs.sin(), freqs.sin()], dim=-1).float()  # [S, D]
        return freqs_cos, freqs_sin
    else:
        # lumina
        freqs_cis = mint.polar(mint.ones_like(freqs), freqs)  # complex64     # [S, D/2]
        return freqs_cis


def apply_rotary_emb(
    x: ms.Tensor,
    freqs_cis: Union[ms.Tensor, Tuple[ms.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> Tuple[ms.Tensor, ms.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`ms.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (ms.Tensor): Key tensor to apply
        freqs_cis (`Tuple[ms.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[ms.Tensor, ms.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        # Support concatenated `freqs_cis` since MindSpore recompute doesn't support calculate tensors' gradient from tuple.
        # todo: unavailable mint interface
        if ops.is_tensor(freqs_cis):
            cos, sin = freqs_cis.chunk(2)  # [1, S, D]
            # cos = cos[None]
            # sin = sin[None]
            cos = cos.expand_dims(axis=0)
            sin = sin.expand_dims(axis=0)
        else:
            cos, sin = freqs_cis  # [S, D]
            # cos = cos[None, None]
            # sin = sin[None, None]
            cos = cos.expand_dims(axis=0).expand_dims(axis=0)
            sin = sin.expand_dims(axis=0).expand_dims(axis=0)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, H, S, D//2]
            x_rotated = mint.stack([-x_imag, x_real], dim=-1).flatten(start_dim=3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio, OmniGen, CogView4 and Cosmos
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, H, S, D//2]
            x_rotated = mint.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        # used for lumina
        x_rotated = view_as_complex(x.float().reshape(x.shape[:-1] + (-1, 2)))
        freqs_cis = freqs_cis.unsqueeze(2)
        # todo: unavailable mint interface
        x_out = ops.view_as_real(x_rotated * freqs_cis).flatten(start_dim=3)

        return x_out.type_as(x)


def apply_rotary_emb_allegro(x: ms.Tensor, freqs_cis, positions):
    # TODO(aryan): rewrite
    def apply_1d_rope(tokens, pos, cos, sin):
        # cos = ops.embedding(pos, ms.Parameter(cos))[:, None, :, :]
        # sin = ops.embedding(pos, ms.Parameter(sin))[:, None, :, :]
        # In `ops.embedding`, weight should be a Parameter, but we do not support `parameter` in graph mode.
        cos = cos[pos][:, None, :, :]
        sin = sin[pos][:, None, :, :]
        x1, x2 = tokens[..., : tokens.shape[-1] // 2], tokens[..., tokens.shape[-1] // 2 :]
        tokens_rotated = mint.cat((-x2, x1), dim=-1)
        return (tokens.float() * cos + tokens_rotated.float() * sin).to(tokens.dtype)

    (t_cos, t_sin), (h_cos, h_sin), (w_cos, w_sin) = freqs_cis
    t, h, w = x.chunk(3, dim=-1)
    t = apply_1d_rope(t, positions[0], t_cos, t_sin)
    h = apply_1d_rope(h, positions[1], h_cos, h_sin)
    w = apply_1d_rope(w, positions[2], w_cos, w_sin)
    x = mint.cat([t, h, w], dim=-1)
    return x


class FluxPosEmbed(nn.Cell):
    # modified from https://github.com/black-forest-labs/flux/blob/c00d7c60b085fce8058b9df845e036090873f2ce/src/flux/modules/layers.py#L11
    def __init__(self, theta: int, axes_dim: List[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def construct(self, ids: ms.Tensor) -> ms.Tensor:
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.float()
        freqs_dtype = ms.float32
        for i in range(n_axes):
            cos, sin = get_1d_rotary_pos_embed(
                self.axes_dim[i],
                mint.split(pos, 1, dim=1)[i].squeeze(axis=1),
                theta=self.theta,
                repeat_interleave_real=True,
                use_real=True,
                freqs_dtype=freqs_dtype,
            )
            cos_out.append(cos)
            sin_out.append(sin)
        freqs_cos = mint.cat(cos_out, dim=-1)
        freqs_sin = mint.cat(sin_out, dim=-1)
        return freqs_cos, freqs_sin


class TimestepEmbedding(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
        sample_proj_bias=True,
    ):
        super().__init__()

        self.linear_1 = mint.nn.Linear(in_channels, time_embed_dim, bias=sample_proj_bias)

        if cond_proj_dim is not None:
            self.cond_proj = mint.nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        self.act = get_activation(act_fn)

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = mint.nn.Linear(time_embed_dim, time_embed_dim_out, bias=sample_proj_bias)

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = get_activation(post_act_fn)

    def construct(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


class Timesteps(nn.Cell):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def construct(self, timesteps: ms.Tensor) -> ms.Tensor:
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )
        return t_emb


class GaussianFourierProjection(nn.Cell):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(
        self, embedding_size: int = 256, scale: float = 1.0, set_W_to_weight=True, log=True, flip_sin_to_cos=False
    ):
        super().__init__()
        self.weight = ms.Parameter(mint.randn(embedding_size) * scale, requires_grad=False, name="weight")
        self.log = log
        self.flip_sin_to_cos = flip_sin_to_cos

        if set_W_to_weight:
            # to delete later
            # FIXME: what is the logic here ???
            del self.weight
            self.W = ms.Parameter(mint.randn(embedding_size) * scale, requires_grad=False, name="weight")
            self.weight = self.W
            del self.W

    def construct(self, x):
        if self.log:
            x = mint.log(x)

        x_proj = x[:, None] * self.weight[None, :] * 2 * ms.numpy.pi

        if self.flip_sin_to_cos:
            out = mint.cat([mint.cos(x_proj), mint.sin(x_proj)], dim=-1)
        else:
            out = mint.cat([mint.sin(x_proj), mint.cos(x_proj)], dim=-1)
        return out


class SinusoidalPositionalEmbedding(nn.Cell):
    """Apply positional information to a sequence of embeddings.

    Takes in a sequence of embeddings with shape (batch_size, seq_length, embed_dim) and adds positional embeddings to
    them

    Args:
        embed_dim: (int): Dimension of the positional embedding.
        max_seq_length: Maximum sequence length to apply positional embeddings

    """

    def __init__(self, embed_dim: int, max_seq_length: int = 32):
        super().__init__()
        position = mint.arange(max_seq_length).unsqueeze(1)
        div_term = mint.exp(mint.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = mint.zeros((1, max_seq_length, embed_dim))
        pe[0, :, 0::2] = mint.sin(position * div_term)
        pe[0, :, 1::2] = mint.cos(position * div_term)
        self.register_buffer("pe", pe)

    def construct(self, x):
        _, seq_length, _ = x.shape
        # In PyTorch, register_buffer allows automatic dtype alignment when using `.to()`.
        # However, in MindSpore, the `.to()` method only applies to parameters.
        # Therefore, we need to manually align the dtype of buffers here.
        x = x + self.pe[:, :seq_length].to(x.dtype)
        return x


class ImagePositionalEmbeddings(nn.Cell):
    """
    Converts latent image classes into vector embeddings. Sums the vector embeddings with positional embeddings for the
    height and width of the latent space.

    For more details, see figure 10 of the dall-e paper: https://huggingface.co/papers/2102.12092

    For VQ-diffusion:

    Output vector embeddings are used as input for the transformer.

    Note that the vector embeddings for the transformer are different than the vector embeddings from the VQVAE.

    Args:
        num_embed (`int`):
            Number of embeddings for the latent pixels embeddings.
        height (`int`):
            Height of the latent image i.e. the number of height embeddings.
        width (`int`):
            Width of the latent image i.e. the number of width embeddings.
        embed_dim (`int`):
            Dimension of the produced vector embeddings. Used for the latent pixel, height, and width embeddings.
    """

    def __init__(
        self,
        num_embed: int,
        height: int,
        width: int,
        embed_dim: int,
    ):
        super().__init__()

        self.height = height
        self.width = width
        self.num_embed = num_embed
        self.embed_dim = embed_dim

        self.emb = mint.nn.Embedding(self.num_embed, embed_dim)
        self.height_emb = mint.nn.Embedding(self.height, embed_dim)
        self.width_emb = mint.nn.Embedding(self.width, embed_dim)

    def construct(self, index):
        emb = self.emb(index)

        height_emb = self.height_emb(mint.arange(self.height).view(1, self.height))

        # 1 x H x D -> 1 x H x 1 x D
        height_emb = height_emb.unsqueeze(2)

        width_emb = self.width_emb(mint.arange(self.width).view(1, self.width))

        # 1 x W x D -> 1 x 1 x W x D
        width_emb = width_emb.unsqueeze(1)

        pos_emb = height_emb + width_emb

        # 1 x H x W x D -> 1 x L xD
        pos_emb = pos_emb.view(1, self.height * self.width, -1)

        emb = emb + pos_emb[:, : emb.shape[1], :]

        return emb


class LabelEmbedding(nn.Cell):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.

    Args:
        num_classes (`int`): The number of classes.
        hidden_size (`int`): The size of the vector embeddings.
        dropout_prob (`float`): The probability of dropping a label.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = mint.nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = mint.rand(labels.shape[0]) < self.dropout_prob
        else:
            drop_ids = ms.tensor(force_drop_ids == 1)
        labels = mint.where(drop_ids, self.num_classes, labels)
        return labels

    def construct(self, labels: ms.Tensor, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (self.training and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class TextImageProjection(nn.Cell):
    def __init__(
        self,
        text_embed_dim: int = 1024,
        image_embed_dim: int = 768,
        cross_attention_dim: int = 768,
        num_image_text_embeds: int = 10,
    ):
        super().__init__()

        self.num_image_text_embeds = num_image_text_embeds
        self.image_embeds = mint.nn.Linear(image_embed_dim, self.num_image_text_embeds * cross_attention_dim)
        self.text_proj = mint.nn.Linear(text_embed_dim, cross_attention_dim)

    def construct(self, text_embeds: ms.Tensor, image_embeds: ms.Tensor):
        batch_size = text_embeds.shape[0]

        # image
        image_text_embeds = self.image_embeds(image_embeds)
        image_text_embeds = image_text_embeds.reshape(batch_size, self.num_image_text_embeds, -1)

        # text
        text_embeds = self.text_proj(text_embeds)

        return mint.cat([image_text_embeds, text_embeds], dim=1)


class ImageProjection(nn.Cell):
    def __init__(
        self,
        image_embed_dim: int = 768,
        cross_attention_dim: int = 768,
        num_image_text_embeds: int = 32,
    ):
        super().__init__()
        self.num_image_text_embeds = num_image_text_embeds
        self.image_embeds = mint.nn.Linear(image_embed_dim, self.num_image_text_embeds * cross_attention_dim)
        self.norm = mint.nn.LayerNorm(cross_attention_dim)

    def construct(self, image_embeds: ms.Tensor):
        batch_size = image_embeds.shape[0]

        # image
        image_embeds = self.image_embeds(image_embeds.to(self.image_embeds.weight.dtype))
        image_embeds = image_embeds.reshape(batch_size, self.num_image_text_embeds, -1)
        image_embeds = self.norm(image_embeds)
        return image_embeds


class IPAdapterFullImageProjection(nn.Cell):
    def __init__(self, image_embed_dim=1024, cross_attention_dim=1024):
        super().__init__()
        from .attention import FeedForward

        self.ff = FeedForward(image_embed_dim, cross_attention_dim, mult=1, activation_fn="gelu")
        self.norm = mint.nn.LayerNorm(cross_attention_dim)

    def construct(self, image_embeds: ms.Tensor):
        return self.norm(self.ff(image_embeds))


class IPAdapterFaceIDImageProjection(nn.Cell):
    def __init__(self, image_embed_dim=1024, cross_attention_dim=1024, mult=1, num_tokens=1):
        super().__init__()
        from .attention import FeedForward

        self.num_tokens = num_tokens
        self.cross_attention_dim = cross_attention_dim
        self.ff = FeedForward(image_embed_dim, cross_attention_dim * num_tokens, mult=mult, activation_fn="gelu")
        self.norm = mint.nn.LayerNorm(cross_attention_dim)

    def construct(self, image_embeds: ms.Tensor):
        x = self.ff(image_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        return self.norm(x)


class CombinedTimestepLabelEmbeddings(nn.Cell):
    def __init__(self, num_classes, embedding_dim, class_dropout_prob=0.1):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.class_embedder = LabelEmbedding(num_classes, embedding_dim, class_dropout_prob)

    def construct(self, timestep, class_labels, hidden_dtype=None):
        timesteps_proj = self.time_proj(timestep)
        hidden_dtype = hidden_dtype or timesteps_proj.dtype  # mindspore does't support tensor.to(None)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (N, D)

        class_labels = self.class_embedder(class_labels)  # (N, D)

        conditioning = timesteps_emb + class_labels  # (N, D)

        return conditioning


class CombinedTimestepTextProjEmbeddings(nn.Cell):
    def __init__(self, embedding_dim, pooled_projection_dim):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.text_embedder = PixArtAlphaTextProjection(pooled_projection_dim, embedding_dim, act_fn="silu")

    def construct(self, timestep, pooled_projection):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))  # (N, D)

        pooled_projections = self.text_embedder(pooled_projection)

        conditioning = timesteps_emb + pooled_projections

        return conditioning


class CombinedTimestepGuidanceTextProjEmbeddings(nn.Cell):
    def __init__(self, embedding_dim, pooled_projection_dim):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.guidance_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.text_embedder = PixArtAlphaTextProjection(pooled_projection_dim, embedding_dim, act_fn="silu")

    def construct(self, timestep, guidance, pooled_projection):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))  # (N, D)

        guidance_proj = self.time_proj(guidance)
        guidance_emb = self.guidance_embedder(guidance_proj.to(dtype=pooled_projection.dtype))  # (N, D)

        time_guidance_emb = timesteps_emb + guidance_emb

        pooled_projections = self.text_embedder(pooled_projection)
        conditioning = time_guidance_emb + pooled_projections

        return conditioning


class CogView3CombinedTimestepSizeEmbeddings(nn.Cell):
    def __init__(self, embedding_dim: int, condition_dim: int, pooled_projection_dim: int, timesteps_dim: int = 256):
        super().__init__()

        self.time_proj = Timesteps(num_channels=timesteps_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.condition_proj = Timesteps(num_channels=condition_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=timesteps_dim, time_embed_dim=embedding_dim)
        self.condition_embedder = PixArtAlphaTextProjection(pooled_projection_dim, embedding_dim, act_fn="silu")

    def construct(
        self,
        timestep: ms.Tensor,
        original_size: ms.Tensor,
        target_size: ms.Tensor,
        crop_coords: ms.Tensor,
        hidden_dtype: ms.Type,
    ) -> ms.Tensor:
        timesteps_proj = self.time_proj(timestep)

        original_size_proj = self.condition_proj(original_size.flatten()).view(original_size.shape[0], -1)
        crop_coords_proj = self.condition_proj(crop_coords.flatten()).view(crop_coords.shape[0], -1)
        target_size_proj = self.condition_proj(target_size.flatten()).view(target_size.shape[0], -1)

        # (B, 3 * condition_dim)
        condition_proj = mint.cat([original_size_proj, crop_coords_proj, target_size_proj], dim=1)

        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (B, embedding_dim)
        condition_emb = self.condition_embedder(condition_proj.to(dtype=hidden_dtype))  # (B, embedding_dim)

        conditioning = timesteps_emb + condition_emb
        return conditioning


class HunyuanDiTAttentionPool(nn.Cell):
    # Copied from https://github.com/Tencent/HunyuanDiT/blob/cb709308d92e6c7e8d59d0dff41b74d35088db6a/hydit/modules/poolers.py#L6

    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = ms.Parameter(
            mint.randn(spacial_dim + 1, embed_dim) / embed_dim**0.5, name="positional_embedding"
        )
        self.k_proj = mint.nn.Linear(embed_dim, embed_dim)
        self.q_proj = mint.nn.Linear(embed_dim, embed_dim)
        self.v_proj = mint.nn.Linear(embed_dim, embed_dim)
        self.c_proj = mint.nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def construct(self, x: ms.Tensor):
        x = x.permute(1, 0, 2)  # NLC -> LNC
        x = mint.cat([mint.mean(x, dim=0, keepdim=True), x], dim=0)  # (L+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (L+1)NC
        # todo: unavailable mint interface
        x, _ = ops.function.nn_func.multi_head_attention_forward(
            query=x[:1],
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=mint.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            dtype=x.dtype,  # mindspore must specify argument dtype, otherwise fp32 will be used
        )
        return x.squeeze(0)


class HunyuanCombinedTimestepTextSizeStyleEmbedding(nn.Cell):
    def __init__(
        self,
        embedding_dim,
        pooled_projection_dim=1024,
        seq_len=256,
        cross_attention_dim=2048,
        use_style_cond_and_image_meta_size=True,
    ):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

        self.size_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)

        self.pooler = HunyuanDiTAttentionPool(
            seq_len, cross_attention_dim, num_heads=8, output_dim=pooled_projection_dim
        )

        # Here we use a default learned embedder layer for future extension.
        self.use_style_cond_and_image_meta_size = use_style_cond_and_image_meta_size
        if use_style_cond_and_image_meta_size:
            self.style_embedder = mint.nn.Embedding(1, embedding_dim)
            extra_in_dim = 256 * 6 + embedding_dim + pooled_projection_dim
        else:
            extra_in_dim = pooled_projection_dim

        self.extra_embedder = PixArtAlphaTextProjection(
            in_features=extra_in_dim,
            hidden_size=embedding_dim * 4,
            out_features=embedding_dim,
            act_fn="silu_fp32",
        )

    def construct(self, timestep, encoder_hidden_states, image_meta_size, style, hidden_dtype=None):
        hidden_dtype = hidden_dtype or encoder_hidden_states.dtype  # tensor.to(None) is invalid
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (N, 256)

        # extra condition1: text
        pooled_projections = self.pooler(encoder_hidden_states)  # (N, 1024)

        if self.use_style_cond_and_image_meta_size:
            # extra condition2: image meta size embdding
            image_meta_size = self.size_proj(image_meta_size.view(-1))
            image_meta_size = image_meta_size.to(dtype=hidden_dtype)
            image_meta_size = image_meta_size.view(-1, 6 * 256)  # (N, 1536)

            # extra condition3: style embedding
            style_embedding = self.style_embedder(style)  # (N, embedding_dim)

            # Concatenate all extra vectors
            extra_cond = mint.cat([pooled_projections, image_meta_size, style_embedding], dim=1)
        else:
            extra_cond = mint.cat([pooled_projections], dim=1)

        conditioning = timesteps_emb + self.extra_embedder(extra_cond)  # [B, D]

        return conditioning


class LuminaCombinedTimestepCaptionEmbedding(nn.Cell):
    def __init__(self, hidden_size=4096, cross_attention_dim=2048, frequency_embedding_size=256):
        super().__init__()
        self.time_proj = Timesteps(
            num_channels=frequency_embedding_size, flip_sin_to_cos=True, downscale_freq_shift=0.0
        )

        self.timestep_embedder = TimestepEmbedding(in_channels=frequency_embedding_size, time_embed_dim=hidden_size)

        self.caption_embedder = nn.SequentialCell(
            mint.nn.LayerNorm(cross_attention_dim),
            mint.nn.Linear(
                cross_attention_dim,
                hidden_size,
                bias=True,
            ),
        )

    def construct(self, timestep, caption_feat, caption_mask):
        # timestep embedding:
        time_freq = self.time_proj(timestep)
        time_embed = self.timestep_embedder(time_freq.to(dtype=caption_feat.dtype))

        # caption condition embedding:
        caption_mask_float = caption_mask.float().unsqueeze(-1)
        caption_feats_pool = (caption_feat * caption_mask_float).sum(dim=1) / caption_mask_float.sum(dim=1)
        caption_feats_pool = caption_feats_pool.to(caption_feat.dtype)
        caption_embed = self.caption_embedder(caption_feats_pool)

        conditioning = time_embed + caption_embed

        return conditioning


class MochiCombinedTimestepCaptionEmbedding(nn.Cell):
    def __init__(
        self,
        embedding_dim: int,
        pooled_projection_dim: int,
        text_embed_dim: int,
        time_embed_dim: int = 256,
        num_attention_heads: int = 8,
    ) -> None:
        super().__init__()

        self.time_proj = Timesteps(num_channels=time_embed_dim, flip_sin_to_cos=True, downscale_freq_shift=0.0)
        self.timestep_embedder = TimestepEmbedding(in_channels=time_embed_dim, time_embed_dim=embedding_dim)
        self.pooler = MochiAttentionPool(
            num_attention_heads=num_attention_heads, embed_dim=text_embed_dim, output_dim=embedding_dim
        )
        self.caption_proj = mint.nn.Linear(text_embed_dim, pooled_projection_dim)

    def construct(
        self,
        timestep: ms.Tensor,
        encoder_hidden_states: ms.Tensor,
        encoder_attention_mask: ms.Tensor,
        hidden_dtype: Optional[ms.Type] = None,
    ):
        time_proj = self.time_proj(timestep)
        time_emb = self.timestep_embedder(time_proj.to(dtype=hidden_dtype))

        pooled_projections = self.pooler(encoder_hidden_states, encoder_attention_mask)
        caption_proj = self.caption_proj(encoder_hidden_states)

        conditioning = time_emb + pooled_projections
        return conditioning, caption_proj


class TextTimeEmbedding(nn.Cell):
    def __init__(self, encoder_dim: int, time_embed_dim: int, num_heads: int = 64):
        super().__init__()
        self.norm1 = mint.nn.LayerNorm(encoder_dim)
        self.pool = AttentionPooling(num_heads, encoder_dim)
        self.proj = mint.nn.Linear(encoder_dim, time_embed_dim)
        self.norm2 = mint.nn.LayerNorm(time_embed_dim)

    def construct(self, hidden_states):
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.pool(hidden_states)
        hidden_states = self.proj(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return hidden_states


class TextImageTimeEmbedding(nn.Cell):
    def __init__(self, text_embed_dim: int = 768, image_embed_dim: int = 768, time_embed_dim: int = 1536):
        super().__init__()
        self.text_proj = mint.nn.Linear(text_embed_dim, time_embed_dim)
        self.text_norm = mint.nn.LayerNorm(time_embed_dim)
        self.image_proj = mint.nn.Linear(image_embed_dim, time_embed_dim)

    def construct(self, text_embeds: ms.Tensor, image_embeds: ms.Tensor):
        # text
        time_text_embeds = self.text_proj(text_embeds)
        time_text_embeds = self.text_norm(time_text_embeds)

        # image
        time_image_embeds = self.image_proj(image_embeds)

        return time_image_embeds + time_text_embeds


class ImageTimeEmbedding(nn.Cell):
    def __init__(self, image_embed_dim: int = 768, time_embed_dim: int = 1536):
        super().__init__()
        self.image_proj = mint.nn.Linear(image_embed_dim, time_embed_dim)
        self.image_norm = mint.nn.LayerNorm(time_embed_dim)

    def construct(self, image_embeds: ms.Tensor):
        # image
        time_image_embeds = self.image_proj(image_embeds)
        time_image_embeds = self.image_norm(time_image_embeds)
        return time_image_embeds


class ImageHintTimeEmbedding(nn.Cell):
    def __init__(self, image_embed_dim: int = 768, time_embed_dim: int = 1536):
        super().__init__()
        self.image_proj = mint.nn.Linear(image_embed_dim, time_embed_dim)
        self.image_norm = mint.nn.LayerNorm(time_embed_dim)
        self.input_hint_block = nn.SequentialCell(
            mint.nn.Conv2d(3, 16, 3, padding=1),
            mint.nn.SiLU(),
            mint.nn.Conv2d(16, 16, 3, padding=1),
            mint.nn.SiLU(),
            mint.nn.Conv2d(16, 32, 3, padding=1, stride=2),
            mint.nn.SiLU(),
            mint.nn.Conv2d(32, 32, 3, padding=1),
            mint.nn.SiLU(),
            mint.nn.Conv2d(32, 96, 3, padding=1, stride=2),
            mint.nn.SiLU(),
            mint.nn.Conv2d(96, 96, 3, padding=1),
            mint.nn.SiLU(),
            mint.nn.Conv2d(96, 256, 3, padding=1, stride=2),
            mint.nn.SiLU(),
            mint.nn.Conv2d(256, 4, 3, padding=1),
        )

    def construct(self, image_embeds: ms.Tensor, hint: ms.Tensor):
        # image
        time_image_embeds = self.image_proj(image_embeds)
        time_image_embeds = self.image_norm(time_image_embeds)
        hint = self.input_hint_block(hint)
        return time_image_embeds, hint


class AttentionPooling(nn.Cell):
    # Copied from:
    # https://github.com/deep-floyd/IF/blob/2f91391f27dd3c468bf174be5805b4cc92980c0b/deepfloyd_if/model/nn.py#L54

    def __init__(self, num_heads, embed_dim, dtype=None):
        super().__init__()
        self.dtype = dtype if dtype else ms.float32
        self.positional_embedding = ms.Parameter(
            mint.randn(1, embed_dim) / embed_dim**0.5, name="positional_embedding"
        )
        self.k_proj = mint.nn.Linear(embed_dim, embed_dim, dtype=self.dtype)
        self.q_proj = mint.nn.Linear(embed_dim, embed_dim, dtype=self.dtype)
        self.v_proj = mint.nn.Linear(embed_dim, embed_dim, dtype=self.dtype)
        self.num_heads = num_heads
        self.dim_per_head = embed_dim // self.num_heads

    def construct(self, x: ms.Tensor):
        bs, length, width = x.shape

        def shape(x):
            # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
            x = x.view(bs, -1, self.num_heads, self.dim_per_head)
            # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
            x = x.swapaxes(1, 2)
            # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
            x = x.reshape(bs * self.num_heads, -1, self.dim_per_head)
            # (bs*n_heads, length, dim_per_head) --> (bs*n_heads, dim_per_head, length)
            x = x.swapaxes(1, 2)
            return x

        class_token = x.mean(dim=1, keepdim=True) + self.positional_embedding.to(x.dtype)
        x = mint.cat([class_token, x], dim=1)  # (bs, length+1, width)

        # (bs*n_heads, class_token_length, dim_per_head)
        q = shape(self.q_proj(class_token))
        # (bs*n_heads, length+class_token_length, dim_per_head)
        k = shape(self.k_proj(x))
        v = shape(self.v_proj(x))

        # (bs*n_heads, class_token_length, length+class_token_length):
        scale = float(1 / math.sqrt(math.sqrt(self.dim_per_head)))
        weight = mint.bmm(q.swapaxes(-1, -2) * scale, k * scale)  # More stable with f16 than dividing afterwards
        weight = mint.nn.functional.softmax(weight.float(), dim=-1).type(weight.dtype)

        # (bs*n_heads, dim_per_head, class_token_length)
        a = mint.bmm(v, weight.swapaxes(-1, -2))

        # (bs, length+1, width)
        a = a.reshape(bs, -1, 1).swapaxes(1, 2)

        return a[:, 0, :]  # cls_token


class MochiAttentionPool(nn.Cell):
    def __init__(
        self,
        num_attention_heads: int,
        embed_dim: int,
        output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.output_dim = output_dim or embed_dim
        self.num_attention_heads = num_attention_heads

        self.to_kv = mint.nn.Linear(embed_dim, 2 * embed_dim)
        self.to_q = mint.nn.Linear(embed_dim, embed_dim)
        self.to_out = mint.nn.Linear(embed_dim, self.output_dim)

    @staticmethod
    def pool_tokens(x: ms.Tensor, mask: ms.Tensor, *, keepdim=False) -> ms.Tensor:
        """
        Pool tokens in x using mask.

        NOTE: We assume x does not require gradients.

        Args:
            x: (B, L, D) tensor of tokens.
            mask: (B, L) boolean tensor indicating which tokens are not padding.

        Returns:
            pooled: (B, D) tensor of pooled tokens.
        """
        assert x.shape[1] == mask.shape[1]  # Expected mask to have same length as tokens.
        assert x.shape[0] == mask.shape[0]  # Expected mask to have same batch size as tokens.
        mask = mask[:, :, None].to(dtype=x.dtype)
        mask = mask / mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = (x * mask).sum(dim=1, keepdim=keepdim)
        return pooled

    def construct(self, x: ms.Tensor, mask: ms.Tensor) -> ms.Tensor:
        r"""
        Args:
            x (`ms.Tensor`):
                Tensor of shape `(B, S, D)` of input tokens.
            mask (`ms.Tensor`):
                Boolean ensor of shape `(B, S)` indicating which tokens are not padding.

        Returns:
            `ms.Tensor`:
                `(B, D)` tensor of pooled tokens.
        """
        D = x.shape[2]

        # Construct attention mask, shape: (B, 1, num_queries=1, num_keys=1+L).
        attn_mask = mask[:, None, None, :].bool()  # (B, 1, 1, L).
        attn_mask = pad(attn_mask, (1, 0), value=True)  # (B, 1, 1, 1+L).

        # Average non-padding token features. These will be used as the query.
        x_pool = self.pool_tokens(x, mask, keepdim=True)  # (B, 1, D)

        # Concat pooled features to input sequence.
        x = mint.cat([x_pool, x], dim=1)  # (B, L+1, D)

        # Compute queries, keys, values. Only the mean token is used to create a query.
        kv = self.to_kv(x)  # (B, L+1, 2 * D)
        q = self.to_q(x[:, 0])  # (B, D)

        # Extract heads.
        head_dim = D // self.num_attention_heads
        kv = unflatten(kv, 2, (2, self.num_attention_heads, head_dim))  # (B, 1+L, 2, H, head_dim)
        kv = kv.swapaxes(1, 3)  # (B, H, 2, 1+L, head_dim)
        k, v = kv.unbind(2)  # (B, H, 1+L, head_dim)
        q = unflatten(q, 1, (self.num_attention_heads, head_dim))  # (B, H, head_dim)
        q = q.unsqueeze(2)  # (B, H, 1, head_dim)

        # Compute attention.
        if attn_mask is not None:
            attn_mask = mint.logical_not(attn_mask) if attn_mask.dtype == ms.bool_ else attn_mask.bool()
            attn_mask = mint.broadcast_to(
                attn_mask, (attn_mask.shape[0], attn_mask.shape[1], q.shape[-2], k.shape[-2])
            )[:, :1, :, :]

        scale = head_dim**-0.5
        if q.dtype in (ms.float16, ms.bfloat16):
            x = ops.operations.nn_ops.FlashAttentionScore(
                head_num=self.num_attention_heads, keep_prob=1.0, scale_value=scale, input_layout="BNSD"
            )(q, k, v, None, None, None, attn_mask)[3]

        else:
            x = ops.operations.nn_ops.FlashAttentionScore(
                head_num=self.num_attention_heads, keep_prob=1.0, scale_value=scale, input_layout="BNSD"
            )(q.to(ms.float16), k.to(ms.float16), v.to(ms.float16), None, None, None, attn_mask)[3]
            x = x.to(q.dtype)

        # Concatenate heads and run output.
        x = x.squeeze(2).flatten(start_dim=1, end_dim=2)  # (B, D = H * head_dim)
        x = self.to_out(x)
        return x


def get_fourier_embeds_from_boundingbox(embed_dim, box):
    """
    Args:
        embed_dim: int
        box: a 3-D tensor [B x N x 4] representing the bounding boxes for GLIGEN pipeline
    Returns:
        [B x N x embed_dim] tensor of positional embeddings
    """

    batch_size, num_boxes = box.shape[:2]

    emb = 100 ** (mint.arange(embed_dim).to(dtype=box.dtype) / embed_dim)
    emb = emb[None, None, None].to(dtype=box.dtype)
    emb = emb * box.unsqueeze(-1)

    emb = mint.stack((emb.sin(), emb.cos()), dim=-1)
    emb = emb.permute(0, 1, 3, 4, 2).reshape(batch_size, num_boxes, embed_dim * 2 * 4)

    return emb


class GLIGENTextBoundingboxProjection(nn.Cell):
    def __init__(self, positive_len, out_dim, feature_type="text-only", fourier_freqs=8):
        super().__init__()
        self.positive_len = positive_len
        self.out_dim = out_dim

        self.fourier_embedder_dim = fourier_freqs
        self.position_dim = fourier_freqs * 2 * 4  # 2: sin/cos, 4: xyxy

        if isinstance(out_dim, tuple):
            out_dim = out_dim[0]

        if feature_type == "text-only":
            self.linears = nn.SequentialCell(
                mint.nn.Linear(self.positive_len + self.position_dim, 512),
                mint.nn.SiLU(),
                mint.nn.Linear(512, 512),
                mint.nn.SiLU(),
                mint.nn.Linear(512, out_dim),
            )
            self.null_positive_feature = ms.Parameter(mint.zeros([self.positive_len]), name="null_positive_feature")

        elif feature_type == "text-image":
            self.linears_text = nn.SequentialCell(
                mint.nn.Linear(self.positive_len + self.position_dim, 512),
                mint.nn.SiLU(),
                mint.nn.Linear(512, 512),
                mint.nn.SiLU(),
                mint.nn.Linear(512, out_dim),
            )
            self.linears_image = nn.SequentialCell(
                mint.nn.Linear(self.positive_len + self.position_dim, 512),
                mint.nn.SiLU(),
                mint.nn.Linear(512, 512),
                mint.nn.SiLU(),
                mint.nn.Linear(512, out_dim),
            )
            self.null_text_feature = ms.Parameter(mint.zeros([self.positive_len]), name="null_text_feature")
            self.null_image_feature = ms.Parameter(mint.zeros([self.positive_len]), name="null_image_feature")

        self.null_position_feature = ms.Parameter(mint.zeros([self.position_dim]), name="null_position_feature")

    def construct(
        self,
        boxes,
        masks,
        positive_embeddings=None,
        phrases_masks=None,
        image_masks=None,
        phrases_embeddings=None,
        image_embeddings=None,
    ):
        masks = masks.unsqueeze(-1)

        # embedding position (it may includes padding as placeholder)
        xyxy_embedding = get_fourier_embeds_from_boundingbox(self.fourier_embedder_dim, boxes)  # B*N*4 -> B*N*C

        # learnable null embedding
        xyxy_null = self.null_position_feature.view(1, 1, -1)

        # replace padding with learnable null embedding
        xyxy_embedding = xyxy_embedding * masks + (1 - masks) * xyxy_null

        # positionet with text only information
        if positive_embeddings is not None:
            # learnable null embedding
            positive_null = self.null_positive_feature.view(1, 1, -1)

            # replace padding with learnable null embedding
            positive_embeddings = positive_embeddings * masks + (1 - masks) * positive_null

            objs = self.linears(mint.cat([positive_embeddings, xyxy_embedding], dim=-1))

        # positionet with text and image information
        else:
            phrases_masks = phrases_masks.unsqueeze(-1)
            image_masks = image_masks.unsqueeze(-1)

            # learnable null embedding
            text_null = self.null_text_feature.view(1, 1, -1)
            image_null = self.null_image_feature.view(1, 1, -1)

            # replace padding with learnable null embedding
            phrases_embeddings = phrases_embeddings * phrases_masks + (1 - phrases_masks) * text_null
            image_embeddings = image_embeddings * image_masks + (1 - image_masks) * image_null

            objs_text = self.linears_text(mint.cat([phrases_embeddings, xyxy_embedding], dim=-1))
            objs_image = self.linears_image(mint.cat([image_embeddings, xyxy_embedding], dim=-1))
            objs = mint.cat([objs_text, objs_image], dim=1)

        return objs


class PixArtAlphaCombinedTimestepSizeEmbeddings(nn.Cell):
    """
    For PixArt-Alpha.
    Reference:
    https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L164C9-L168C29
    """

    def __init__(self, embedding_dim, size_emb_dim, use_additional_conditions: bool = False):
        super().__init__()

        self.outdim = size_emb_dim
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

        self.use_additional_conditions = use_additional_conditions
        if use_additional_conditions:
            self.additional_condition_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
            self.resolution_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=size_emb_dim)
            self.aspect_ratio_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=size_emb_dim)

    def construct(self, timestep, resolution, aspect_ratio, batch_size, hidden_dtype):
        timesteps_proj = self.time_proj(timestep)
        hidden_dtype = hidden_dtype or timesteps_proj.dtype
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (N, D)

        if self.use_additional_conditions:
            resolution_emb = self.additional_condition_proj(resolution.flatten()).to(hidden_dtype)
            resolution_emb = self.resolution_embedder(resolution_emb).reshape(batch_size, -1)
            aspect_ratio_emb = self.additional_condition_proj(aspect_ratio.flatten()).to(hidden_dtype)
            aspect_ratio_emb = self.aspect_ratio_embedder(aspect_ratio_emb).reshape(batch_size, -1)
            conditioning = timesteps_emb + mint.cat([resolution_emb, aspect_ratio_emb], dim=1)
        else:
            conditioning = timesteps_emb

        return conditioning


class PixArtAlphaTextProjection(nn.Cell):
    """
    Projects caption embeddings. Also handles dropout for classifier-free guidance.
    Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
    """

    def __init__(self, in_features, hidden_size, out_features=None, act_fn="gelu_tanh"):
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear_1 = mint.nn.Linear(in_features, hidden_size, bias=True)
        if act_fn == "gelu_tanh":
            self.act_1 = GELU(approximate="tanh")
        elif act_fn == "silu":
            self.act_1 = mint.nn.SiLU()
        elif act_fn == "silu_fp32":
            self.act_1 = FP32SiLU()
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")
        self.linear_2 = mint.nn.Linear(hidden_size, out_features, bias=True)

    def construct(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class IPAdapterPlusImageProjectionBlock(nn.Cell):
    def __init__(
        self,
        embed_dims: int = 768,
        dim_head: int = 64,
        heads: int = 16,
        ffn_ratio: float = 4,
    ) -> None:
        super().__init__()
        from .attention import FeedForward

        self.ln0 = mint.nn.LayerNorm(embed_dims)
        self.ln1 = mint.nn.LayerNorm(embed_dims)
        self.attn = Attention(
            query_dim=embed_dims,
            dim_head=dim_head,
            heads=heads,
            out_bias=False,
        )
        self.ff = nn.SequentialCell(
            mint.nn.LayerNorm(embed_dims),
            FeedForward(embed_dims, embed_dims, activation_fn="gelu", mult=ffn_ratio, bias=False),
        )

    def construct(self, x, latents, residual):
        encoder_hidden_states = self.ln0(x)
        latents = self.ln1(latents)
        encoder_hidden_states = mint.cat([encoder_hidden_states, latents], dim=-2)
        latents = self.attn(latents, encoder_hidden_states) + residual
        latents = self.ff(latents) + latents
        return latents


class IPAdapterPlusImageProjection(nn.Cell):
    """Resampler of IP-Adapter Plus.

    Args:
        embed_dims (int): The feature dimension. Defaults to 768. output_dims (int): The number of output channels,
        that is the same
            number of the channels in the `unet.config.cross_attention_dim`. Defaults to 1024.
        hidden_dims (int):
            The number of hidden channels. Defaults to 1280. depth (int): The number of blocks. Defaults
        to 8. dim_head (int): The number of head channels. Defaults to 64. heads (int): Parallel attention heads.
        Defaults to 16. num_queries (int):
            The number of queries. Defaults to 8. ffn_ratio (float): The expansion ratio
        of feedforward network hidden
            layer channels. Defaults to 4.
    """

    def __init__(
        self,
        embed_dims: int = 768,
        output_dims: int = 1024,
        hidden_dims: int = 1280,
        depth: int = 4,
        dim_head: int = 64,
        heads: int = 16,
        num_queries: int = 8,
        ffn_ratio: float = 4,
    ) -> None:
        super().__init__()
        self.latents = ms.Parameter(mint.randn(1, num_queries, hidden_dims) / hidden_dims**0.5, name="latents")

        self.proj_in = mint.nn.Linear(embed_dims, hidden_dims)

        self.proj_out = mint.nn.Linear(hidden_dims, output_dims)
        self.norm_out = mint.nn.LayerNorm(output_dims)

        self.layers = nn.CellList(
            [IPAdapterPlusImageProjectionBlock(hidden_dims, dim_head, heads, ffn_ratio) for _ in range(depth)]
        )

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        """Forward pass.

        Args:
            x (ms.Tensor): Input Tensor.
        Returns:
            ms.Tensor: Output Tensor.
        """
        latents = self.latents.tile((x.shape[0], 1, 1))

        x = self.proj_in(x)

        for block in self.layers:
            residual = latents
            latents = block(x, latents, residual)

        latents = self.proj_out(latents)
        return self.norm_out(latents)


class IPAdapterFaceIDPlusImageProjection(nn.Cell):
    """FacePerceiverResampler of IP-Adapter Plus.

    Args:
        embed_dims (int): The feature dimension. Defaults to 768. output_dims (int): The number of output channels,
        that is the same
            number of the channels in the `unet.config.cross_attention_dim`. Defaults to 1024.
        hidden_dims (int):
            The number of hidden channels. Defaults to 1280. depth (int): The number of blocks. Defaults
        to 8. dim_head (int): The number of head channels. Defaults to 64. heads (int): Parallel attention heads.
        Defaults to 16. num_tokens (int): Number of tokens num_queries (int): The number of queries. Defaults to 8.
        ffn_ratio (float): The expansion ratio of feedforward network hidden
            layer channels. Defaults to 4.
        ffproj_ratio (float): The expansion ratio of feedforward network hidden
            layer channels (for ID embeddings). Defaults to 4.
    """

    def __init__(
        self,
        embed_dims: int = 768,
        output_dims: int = 768,
        hidden_dims: int = 1280,
        id_embeddings_dim: int = 512,
        depth: int = 4,
        dim_head: int = 64,
        heads: int = 16,
        num_tokens: int = 4,
        num_queries: int = 8,
        ffn_ratio: float = 4,
        ffproj_ratio: int = 2,
    ) -> None:
        super().__init__()
        from .attention import FeedForward

        self.num_tokens = num_tokens
        self.embed_dim = embed_dims
        self.clip_embeds = None
        self.shortcut = False
        self.shortcut_scale = 1.0

        self.proj = FeedForward(id_embeddings_dim, embed_dims * num_tokens, activation_fn="gelu", mult=ffproj_ratio)
        self.norm = mint.nn.LayerNorm(embed_dims)

        self.proj_in = mint.nn.Linear(hidden_dims, embed_dims)

        self.proj_out = mint.nn.Linear(embed_dims, output_dims)
        self.norm_out = mint.nn.LayerNorm(output_dims)

        self.layers = nn.CellList(
            [IPAdapterPlusImageProjectionBlock(embed_dims, dim_head, heads, ffn_ratio) for _ in range(depth)]
        )

    def construct(self, id_embeds: ms.Tensor) -> ms.Tensor:
        """Forward pass.

        Args:
            id_embeds (ms.Tensor): Input Tensor (ID embeds).
        Returns:
            ms.Tensor: Output Tensor.
        """
        id_embeds = id_embeds.to(self.clip_embeds.dtype)
        id_embeds = self.proj(id_embeds)
        id_embeds = id_embeds.reshape(-1, self.num_tokens, self.embed_dim)
        id_embeds = self.norm(id_embeds)
        latents = id_embeds

        clip_embeds = self.proj_in(self.clip_embeds)
        x = clip_embeds.reshape(-1, clip_embeds.shape[2], clip_embeds.shape[3])

        for block in self.layers:
            residual = latents
            latents = block(x, latents, residual)

        latents = self.proj_out(latents)
        out = self.norm_out(latents)
        if self.shortcut:
            out = id_embeds + self.shortcut_scale * out
        return out


class IPAdapterTimeImageProjectionBlock(nn.Cell):
    """Block for IPAdapterTimeImageProjection.

    Args:
        hidden_dim (`int`, defaults to 1280):
            The number of hidden channels.
        dim_head (`int`, defaults to 64):
            The number of head channels.
        heads (`int`, defaults to 20):
            Parallel attention heads.
        ffn_ratio (`int`, defaults to 4):
            The expansion ratio of feedforward network hidden layer channels.
    """

    def __init__(
        self,
        hidden_dim: int = 1280,
        dim_head: int = 64,
        heads: int = 20,
        ffn_ratio: int = 4,
    ) -> None:
        super().__init__()
        from .attention import FeedForward

        self.ln0 = mint.nn.LayerNorm(hidden_dim)
        self.ln1 = mint.nn.LayerNorm(hidden_dim)
        self.attn = Attention(
            query_dim=hidden_dim,
            cross_attention_dim=hidden_dim,
            dim_head=dim_head,
            heads=heads,
            bias=False,
            out_bias=False,
        )
        self.ff = FeedForward(hidden_dim, hidden_dim, activation_fn="gelu", mult=ffn_ratio, bias=False)

        # AdaLayerNorm
        self.adaln_silu = mint.nn.SiLU()
        self.adaln_proj = mint.nn.Linear(hidden_dim, 4 * hidden_dim)
        self.adaln_norm = mint.nn.LayerNorm(hidden_dim)

        # Set attention scale and fuse KV
        self.attn.scale = 1 / math.sqrt(math.sqrt(dim_head))
        self.attn.fuse_projections()
        self.attn.to_k = None
        self.attn.to_v = None

    def construct(self, x: ms.Tensor, latents: ms.Tensor, timestep_emb: ms.Tensor) -> ms.Tensor:
        """Forward pass.

        Args:
            x (`ms.Tensor`):
                Image features.
            latents (`ms.Tensor`):
                Latent features.
            timestep_emb (`ms.Tensor`):
                Timestep embedding.

        Returns:
            `ms.Tensor`: Output latent features.
        """

        # Shift and scale for AdaLayerNorm
        emb = self.adaln_proj(self.adaln_silu(timestep_emb))
        shift_msa, scale_msa, shift_mlp, scale_mlp = emb.chunk(4, dim=1)

        # Fused Attention
        residual = latents
        x = self.ln0(x)
        latents = self.ln1(latents) * (1 + scale_msa[:, None]) + shift_msa[:, None]

        batch_size = latents.shape[0]

        query = self.attn.to_q(latents)
        kv_input = mint.cat((x, latents), dim=-2)
        key, value = self.attn.to_kv(kv_input).chunk(2, dim=-1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.attn.heads

        query = query.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)

        weight = mint.matmul(query * self.attn.scale, key * self.attn.scale).transpose(-2, -1)
        weight = mint.softmax(weight.float(), dim=-1).to(weight.dtype)
        latents = mint.matmul(weight, value)

        latents = latents.transpose(1, 2).reshape(batch_size, -1, self.attn.heads * head_dim)
        latents = self.attn.to_out[0](latents)
        latents = self.attn.to_out[1](latents)
        latents = latents + residual

        # FeedForward
        residual = latents
        latents = self.adaln_norm(latents) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        return self.ff(latents) + residual


# Modified from https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py
class IPAdapterTimeImageProjection(nn.Cell):
    """Resampler of SD3 IP-Adapter with timestep embedding.

    Args:
        embed_dim (`int`, defaults to 1152):
            The feature dimension.
        output_dim (`int`, defaults to 2432):
            The number of output channels.
        hidden_dim (`int`, defaults to 1280):
            The number of hidden channels.
        depth (`int`, defaults to 4):
            The number of blocks.
        dim_head (`int`, defaults to 64):
            The number of head channels.
        heads (`int`, defaults to 20):
            Parallel attention heads.
        num_queries (`int`, defaults to 64):
            The number of queries.
        ffn_ratio (`int`, defaults to 4):
            The expansion ratio of feedforward network hidden layer channels.
        timestep_in_dim (`int`, defaults to 320):
            The number of input channels for timestep embedding.
        timestep_flip_sin_to_cos (`bool`, defaults to True):
            Flip the timestep embedding order to `cos, sin` (if True) or `sin, cos` (if False).
        timestep_freq_shift (`int`, defaults to 0):
            Controls the timestep delta between frequencies between dimensions.
    """

    def __init__(
        self,
        embed_dim: int = 1152,
        output_dim: int = 2432,
        hidden_dim: int = 1280,
        depth: int = 4,
        dim_head: int = 64,
        heads: int = 20,
        num_queries: int = 64,
        ffn_ratio: int = 4,
        timestep_in_dim: int = 320,
        timestep_flip_sin_to_cos: bool = True,
        timestep_freq_shift: int = 0,
    ) -> None:
        super().__init__()
        self.latents = ms.Parameter(mint.randn(1, num_queries, hidden_dim) / hidden_dim**0.5, name="latents")
        self.proj_in = mint.nn.Linear(embed_dim, hidden_dim)
        self.proj_out = mint.nn.Linear(hidden_dim, output_dim)
        self.norm_out = mint.nn.LayerNorm(output_dim)
        self.layers = nn.CellList(
            [IPAdapterTimeImageProjectionBlock(hidden_dim, dim_head, heads, ffn_ratio) for _ in range(depth)]
        )
        self.time_proj = Timesteps(timestep_in_dim, timestep_flip_sin_to_cos, timestep_freq_shift)
        self.time_embedding = TimestepEmbedding(timestep_in_dim, hidden_dim, act_fn="silu")

    def construct(self, x: ms.Tensor, timestep: ms.Tensor) -> Tuple[ms.Tensor, ms.Tensor]:
        """Forward pass.

        Args:
            x (`ms.Tensor`):
                Image features.
            timestep (`ms.Tensor`):
                Timestep in denoising process.
        Returns:
            `Tuple`[`ms.Tensor`, `ms.Tensor`]: The pair (latents, timestep_emb).
        """
        timestep_emb = self.time_proj(timestep).to(dtype=x.dtype)
        timestep_emb = self.time_embedding(timestep_emb)

        latents = self.latents.tile((x.shape[0], 1, 1))

        x = self.proj_in(x)
        x = x + timestep_emb[:, None]

        for block in self.layers:
            latents = block(x, latents, timestep_emb)

        latents = self.proj_out(latents)
        latents = self.norm_out(latents)

        return latents, timestep_emb


class MultiIPAdapterImageProjection(nn.Cell):
    def __init__(self, IPAdapterImageProjectionLayers: Union[List[nn.Cell], Tuple[nn.Cell]]):
        super().__init__()
        self.image_projection_layers = nn.CellList(IPAdapterImageProjectionLayers)

    @property
    def num_ip_adapters(self) -> int:
        """Number of IP-Adapters loaded."""
        return len(self.image_projection_layers)

    def construct(self, image_embeds: List[ms.Tensor]):
        projected_image_embeds = []

        # currently, we accept `image_embeds` as 1. a tensor (deprecated) with shape [batch_size, embed_dim] or [
        # batch_size, sequence_length, embed_dim] 2. list of `n` tensors where `n` is number of ip-adapters,
        # each tensor can hae shape [batch_size, num_images, embed_dim] or [batch_size, num_images, sequence_length,
        # embed_dim]
        if not isinstance(image_embeds, list):
            image_embeds = [image_embeds.unsqueeze(1)]

        assert len(image_embeds) == len(self.image_projection_layers), (
            f"image_embeds must have the same length as "
            f"image_projection_layers, "
            f"got {len(image_embeds)} and "
            f"{len(self.image_projection_layers)}"
        )

        for image_embed, image_projection_layer in zip(image_embeds, self.image_projection_layers):
            batch_size, num_images = image_embed.shape[0], image_embed.shape[1]
            image_embed = image_embed.reshape((batch_size * num_images,) + image_embed.shape[2:])
            image_embed = image_projection_layer(image_embed)
            image_embed = image_embed.reshape((batch_size, num_images) + image_embed.shape[1:])

            projected_image_embeds.append(image_embed)

        return projected_image_embeds


class _GELU(nn.Cell):
    def __init__(self, approximate: str = "none") -> None:
        super().__init__()
        self.approximate = approximate

    def construct(self, input: ms.Tensor) -> ms.Tensor:
        return mint.nn.functional.gelu(input, approximate=self.approximate)
