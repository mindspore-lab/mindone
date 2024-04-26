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
from typing import Optional

import numpy as np

import mindspore as ms
from mindspore import nn, ops

from .activations import get_activation
from .normalization import LayerNorm


def get_timestep_embedding(
    timesteps: ms.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -ops.log(ms.Tensor(max_period, dtype=ms.float32)) * ops.arange(start=0, end=half_dim, dtype=ms.float32)
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = ops.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = ops.cat([ops.sin(emb), ops.cos(emb)], axis=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = ops.cat([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = ops.pad(emb, (0, 1, 0, 0))
    return emb


def get_2d_sincos_pos_embed(
    embed_dim, grid_size, cls_token=False, extra_tokens=0, interpolation_scale=1.0, base_size=16
):
    """
    grid_size: int of the grid height and width return: pos_embed: [grid_size*grid_size, embed_dim] or
    [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)

    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0] / base_size) / interpolation_scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1] / base_size) / interpolation_scale
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
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
        linear_cls = nn.Dense

        self.linear_1 = linear_cls(in_channels, time_embed_dim, has_bias=sample_proj_bias)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Dense(cond_proj_dim, in_channels, has_bias=False)
        else:
            self.cond_proj = None

        self.act = get_activation(act_fn)()

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = linear_cls(time_embed_dim, time_embed_dim_out, has_bias=sample_proj_bias)

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = get_activation(post_act_fn)()

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
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def construct(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )
        return t_emb


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
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = ops.rand(labels.shape[0]) < self.dropout_prob
        else:
            drop_ids = ms.tensor(force_drop_ids == 1)
        labels = ops.where(drop_ids, self.num_classes, labels)
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
        self.image_embeds = nn.Dense(image_embed_dim, self.num_image_text_embeds * cross_attention_dim)
        self.text_proj = nn.Dense(text_embed_dim, cross_attention_dim)

    def construct(self, text_embeds: ms.Tensor, image_embeds: ms.Tensor):
        batch_size = text_embeds.shape[0]

        # image
        image_text_embeds = self.image_embeds(image_embeds)
        image_text_embeds = image_text_embeds.reshape(batch_size, self.num_image_text_embeds, -1)

        # text
        text_embeds = self.text_proj(text_embeds)

        return ops.cat([image_text_embeds, text_embeds], axis=1)


class ImageProjection(nn.Cell):
    def __init__(
        self,
        image_embed_dim: int = 768,
        cross_attention_dim: int = 768,
        num_image_text_embeds: int = 32,
    ):
        super().__init__()

        self.num_image_text_embeds = num_image_text_embeds
        self.image_embeds = nn.Dense(image_embed_dim, self.num_image_text_embeds * cross_attention_dim)
        self.norm = LayerNorm(cross_attention_dim)

    def construct(self, image_embeds: ms.Tensor):
        batch_size = image_embeds.shape[0]

        # image
        image_embeds = self.image_embeds(image_embeds)
        image_embeds = image_embeds.reshape(batch_size, self.num_image_text_embeds, -1)
        image_embeds = self.norm(image_embeds)
        return image_embeds


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
        position = ops.arange(max_seq_length).unsqueeze(1)
        div_term = ops.exp(ops.arange(0, embed_dim, 2) * (-ops.log(ms.Tensor(10000.0)) / embed_dim))
        pe = ops.zeros(1, max_seq_length, embed_dim)
        pe[0, :, 0::2] = ops.sin(position * div_term)
        pe[0, :, 1::2] = ops.cos(position * div_term)
        self.pe = nn.Parameter(ms.Tensor(pe), requires_grad=False)

    def construct(self, x):
        _, seq_length, _ = x.shape
        x = x + self.pe[:, :seq_length]
        return x


class ImagePositionalEmbeddings(nn.Cell):
    """
    Converts latent image classes into vector embeddings. Sums the vector embeddings with positional embeddings for the
    height and width of the latent space.
    For more details, see figure 10 of the dall-e paper: https://arxiv.org/abs/2102.12092
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

        self.emb = nn.Embedding(self.num_embed, embed_dim)
        self.height_emb = nn.Embedding(self.height, embed_dim)
        self.width_emb = nn.Embedding(self.width, embed_dim)

    def forward(self, index):
        emb = self.emb(index)

        height_emb = self.height_emb(ops.arange(self.height).view(1, self.height))

        # 1 x H x D -> 1 x H x 1 x D
        height_emb = height_emb.unsqueeze(2)

        width_emb = self.width_emb(ops.arange(self.width).view(1, self.width))

        # 1 x W x D -> 1 x 1 x W x D
        width_emb = width_emb.unsqueeze(1)

        pos_emb = height_emb + width_emb

        # 1 x H x W x D -> 1 x L xD
        pos_emb = pos_emb.view(1, self.height * self.width, -1)

        emb = emb + pos_emb[:, : emb.shape[1], :]

        return emb


class CombinedTimestepLabelEmbeddings(nn.Cell):
    def __init__(self, num_classes, embedding_dim, class_dropout_prob=0.1):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.class_embedder = LabelEmbedding(num_classes, embedding_dim, class_dropout_prob)

    def construct(self, timestep, class_labels, hidden_dtype=None):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (N, D)

        class_labels = self.class_embedder(class_labels)  # (N, D)

        conditioning = timesteps_emb + class_labels  # (N, D)

        return conditioning


class PixArtAlphaCombinedTimestepSizeEmbeddings(nn.Cell):
    """
    For PixArt-Alpha.
    Reference:
    https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion\
        /model/nets/PixArtMS.py#L164C9-L168C29
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
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (N, D)

        if self.use_additional_conditions:
            resolution_emb = self.additional_condition_proj(resolution.flatten()).to(hidden_dtype)
            resolution_emb = self.resolution_embedder(resolution_emb).reshape(batch_size, -1)
            aspect_ratio_emb = self.additional_condition_proj(aspect_ratio.flatten()).to(hidden_dtype)
            aspect_ratio_emb = self.aspect_ratio_embedder(aspect_ratio_emb).reshape(batch_size, -1)
            conditioning = timesteps_emb + ops.cat([resolution_emb, aspect_ratio_emb], axis=1)
        else:
            conditioning = timesteps_emb

        return conditioning


class PatchEmbed(nn.Cell):
    """2D Image to Patch Embedding"""

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
    ):
        super().__init__()

        num_patches = (height // patch_size) * (width // patch_size)
        self.flatten = flatten
        self.layer_norm = layer_norm

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, has_bias=bias)
        if layer_norm:
            self.norm = LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm = None

        self.patch_size = patch_size
        # See:
        # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L161
        self.height, self.width = height // patch_size, width // patch_size
        self.base_size = height // patch_size
        self.interpolation_scale = interpolation_scale
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim, int(num_patches**0.5), base_size=self.base_size, interpolation_scale=self.interpolation_scale
        )
        self.pos_embed = ms.Parameter(ms.Tensor(pos_embed).float().unsqueeze(0), requires_grad=False)

    def construct(self, latent):
        height, width = latent.shape[-2] // self.patch_size, latent.shape[-1] // self.patch_size

        latent = self.proj(latent)
        if self.flatten:
            latent = latent.flatten(start_dim=2).permute(0, 2, 1)  # BCHW -> BNC
        if self.layer_norm:
            latent = self.norm(latent)

        # Interpolate positional embeddings if needed.
        # (For PixArt-Alpha: https://github.com/PixArt-alpha/PixArt-alpha/\
        # blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L162C151-L162C160)
        if self.height != height or self.width != width:
            pos_embed = get_2d_sincos_pos_embed(
                embed_dim=self.pos_embed.shape[-1],
                grid_size=(height, width),
                base_size=self.base_size,
                interpolation_scale=self.interpolation_scale,
            )
            pos_embed = ms.Tensor(pos_embed)
            pos_embed = pos_embed.float().unsqueeze(0)
        else:
            pos_embed = self.pos_embed

        return (latent + pos_embed).to(latent.dtype)
