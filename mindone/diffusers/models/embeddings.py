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
from typing import List, Optional, Tuple, Union

import numpy as np

import mindspore as ms
from mindspore import nn, ops

from .activations import FP32SiLU, get_activation
from .attention_processor import Attention


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


class PatchEmbed(nn.Cell):
    """2D Image to Patch Embedding with support for SD3 cropping."""

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
    ):
        super().__init__()
        from .normalization import LayerNorm

        num_patches = (height // patch_size) * (width // patch_size)
        self.flatten = flatten
        self.layer_norm = layer_norm
        self.pos_embed_max_size = pos_embed_max_size

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=patch_size,
            pad_mode="pad",
            has_bias=bias,
        )
        if layer_norm:
            self.norm = LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
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
                embed_dim, grid_size, base_size=self.base_size, interpolation_scale=self.interpolation_scale
            )
            pos_embed = ms.Tensor.from_numpy(pos_embed).float().unsqueeze(0)
            persistent = True if pos_embed_max_size else False
            if persistent:
                self.pos_embed = ms.Parameter(pos_embed, name="pos_embed")
            else:
                self.pos_embed = pos_embed
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
                raise NotImplementedError(
                    "A MindSpore version 'get_2d_sincos_pos_embed' method is needed and not implemented so far."
                )
            else:
                pos_embed = self.pos_embed

        return (latent + pos_embed).to(latent.dtype)


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


class GaussianFourierProjection(nn.Cell):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(
        self, embedding_size: int = 256, scale: float = 1.0, set_W_to_weight=True, log=True, flip_sin_to_cos=False
    ):
        super().__init__()
        self.weight = ms.Parameter(ops.randn(embedding_size) * scale, requires_grad=False, name="weight")
        self.log = log
        self.flip_sin_to_cos = flip_sin_to_cos

        if set_W_to_weight:
            # to delete later
            self.W = ms.Parameter(ops.randn(embedding_size) * scale, requires_grad=False, name="W")

            self.weight = self.W

    def construct(self, x):
        if self.log:
            x = ops.log(x)

        x_proj = x[:, None] * self.weight[None, :] * 2 * ms.numpy.pi

        if self.flip_sin_to_cos:
            out = ops.cat([ops.cos(x_proj), ops.sin(x_proj)], axis=-1)
        else:
            out = ops.cat([ops.sin(x_proj), ops.cos(x_proj)], axis=-1)
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
        position = np.expand_dims(np.arange(max_seq_length), axis=1)
        div_term = np.exp(np.arange(0, embed_dim, 2) * (-np.log(10000.0) / embed_dim))
        pe = np.zeros(shape=(1, max_seq_length, embed_dim))
        pe[0, :, 0::2] = np.sin(position * div_term)
        pe[0, :, 1::2] = np.cos(position * div_term)
        self.pe = ms.Tensor.from_numpy(pe).float()

    def construct(self, x):
        _, seq_length, _ = x.shape
        x = x + self.pe[:, :seq_length].to(x.dtype)
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

    def construct(self, index):
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
        from .normalization import LayerNorm

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


class IPAdapterFullImageProjection(nn.Cell):
    def __init__(self, image_embed_dim=1024, cross_attention_dim=1024):
        super().__init__()
        from .attention import FeedForward
        from .normalization import LayerNorm

        self.ff = FeedForward(image_embed_dim, cross_attention_dim, mult=1, activation_fn="gelu")
        self.norm = LayerNorm(cross_attention_dim)

    def construct(self, image_embeds: ms.Tensor):
        return self.norm(self.ff(image_embeds))


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


class TextTimeEmbedding(nn.Cell):
    def __init__(self, encoder_dim: int, time_embed_dim: int, num_heads: int = 64):
        super().__init__()
        from .normalization import LayerNorm

        self.norm1 = LayerNorm(encoder_dim)
        self.pool = AttentionPooling(num_heads, encoder_dim)
        self.proj = nn.Dense(encoder_dim, time_embed_dim)
        self.norm2 = LayerNorm(time_embed_dim)

    def construct(self, hidden_states):
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.pool(hidden_states)
        hidden_states = self.proj(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return hidden_states


class TextImageTimeEmbedding(nn.Cell):
    def __init__(self, text_embed_dim: int = 768, image_embed_dim: int = 768, time_embed_dim: int = 1536):
        super().__init__()
        from .normalization import LayerNorm

        self.text_proj = nn.Dense(text_embed_dim, time_embed_dim)
        self.text_norm = LayerNorm(time_embed_dim)
        self.image_proj = nn.Dense(image_embed_dim, time_embed_dim)

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
        from .normalization import LayerNorm

        self.image_proj = nn.Dense(image_embed_dim, time_embed_dim)
        self.image_norm = LayerNorm(time_embed_dim)

    def construct(self, image_embeds: ms.Tensor):
        # image
        time_image_embeds = self.image_proj(image_embeds)
        time_image_embeds = self.image_norm(time_image_embeds)
        return time_image_embeds


class ImageHintTimeEmbedding(nn.Cell):
    def __init__(self, image_embed_dim: int = 768, time_embed_dim: int = 1536):
        super().__init__()
        from .normalization import LayerNorm

        self.image_proj = nn.Dense(image_embed_dim, time_embed_dim)
        self.image_norm = LayerNorm(time_embed_dim)
        self.input_hint_block = nn.SequentialCell(
            nn.Conv2d(3, 16, 3, pad_mode="pad", padding=1, has_bias=True),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, pad_mode="pad", padding=1, has_bias=True),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, pad_mode="pad", padding=1, stride=2, has_bias=True),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, pad_mode="pad", padding=1, has_bias=True),
            nn.SiLU(),
            nn.Conv2d(32, 96, 3, pad_mode="pad", padding=1, stride=2, has_bias=True),
            nn.SiLU(),
            nn.Conv2d(96, 96, 3, pad_mode="pad", padding=1, has_bias=True),
            nn.SiLU(),
            nn.Conv2d(96, 256, 3, pad_mode="pad", padding=1, stride=2, has_bias=True),
            nn.SiLU(),
            nn.Conv2d(256, 4, 3, pad_mode="pad", padding=1, has_bias=True),
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
            ops.randn(1, embed_dim) / embed_dim**0.5, name="positional_embedding"
        )
        self.k_proj = nn.Dense(embed_dim, embed_dim, dtype=self.dtype)
        self.q_proj = nn.Dense(embed_dim, embed_dim, dtype=self.dtype)
        self.v_proj = nn.Dense(embed_dim, embed_dim, dtype=self.dtype)
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

        class_token = x.mean(axis=1, keep_dims=True) + self.positional_embedding.to(x.dtype)
        x = ops.cat([class_token, x], axis=1)  # (bs, length+1, width)

        # (bs*n_heads, class_token_length, dim_per_head)
        q = shape(self.q_proj(class_token))
        # (bs*n_heads, length+class_token_length, dim_per_head)
        k = shape(self.k_proj(x))
        v = shape(self.v_proj(x))

        # (bs*n_heads, class_token_length, length+class_token_length):
        scale = float(1 / math.sqrt(math.sqrt(self.dim_per_head)))
        weight = ops.bmm(q.swapaxes(-1, -2) * scale, k * scale)  # More stable with f16 than dividing afterwards
        weight = ops.softmax(weight.float(), axis=-1).type(weight.dtype)

        # (bs*n_heads, dim_per_head, class_token_length)
        a = ops.bmm(v, weight.swapaxes(-1, -2))

        # (bs, length+1, width)
        a = a.reshape(bs, -1, 1).swapaxes(1, 2)

        return a[:, 0, :]  # cls_token


def get_fourier_embeds_from_boundingbox(embed_dim, box):
    """
    Args:
        embed_dim: int
        box: a 3-D tensor [B x N x 4] representing the bounding boxes for GLIGEN pipeline
    Returns:
        [B x N x embed_dim] tensor of positional embeddings
    """

    batch_size, num_boxes = box.shape[:2]

    emb = 100 ** (ops.arange(embed_dim).to(dtype=box.dtype) / embed_dim)
    emb = emb[None, None, None].to(dtype=box.dtype)
    emb = emb * box.unsqueeze(-1)

    emb = ops.stack((emb.sin(), emb.cos()), axis=-1)
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
                nn.Dense(self.positive_len + self.position_dim, 512),
                nn.SiLU(),
                nn.Dense(512, 512),
                nn.SiLU(),
                nn.Dense(512, out_dim),
            )
            self.null_positive_feature = ms.Parameter(ops.zeros([self.positive_len]), name="null_positive_feature")

        elif feature_type == "text-image":
            self.linears_text = nn.SequentialCell(
                nn.Dense(self.positive_len + self.position_dim, 512),
                nn.SiLU(),
                nn.Dense(512, 512),
                nn.SiLU(),
                nn.Dense(512, out_dim),
            )
            self.linears_image = nn.SequentialCell(
                nn.Dense(self.positive_len + self.position_dim, 512),
                nn.SiLU(),
                nn.Dense(512, 512),
                nn.SiLU(),
                nn.Dense(512, out_dim),
            )
            self.null_text_feature = ms.Parameter(ops.zeros([self.positive_len]), name="null_text_feature")
            self.null_image_feature = ms.Parameter(ops.zeros([self.positive_len]), name="null_image_feature")

        self.null_position_feature = ms.Parameter(ops.zeros([self.position_dim]), name="null_position_feature")

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

            objs = self.linears(ops.cat([positive_embeddings, xyxy_embedding], axis=-1))

        # positionet with text and image infomation
        else:
            phrases_masks = phrases_masks.unsqueeze(-1)
            image_masks = image_masks.unsqueeze(-1)

            # learnable null embedding
            text_null = self.null_text_feature.view(1, 1, -1)
            image_null = self.null_image_feature.view(1, 1, -1)

            # replace padding with learnable null embedding
            phrases_embeddings = phrases_embeddings * phrases_masks + (1 - phrases_masks) * text_null
            image_embeddings = image_embeddings * image_masks + (1 - image_masks) * image_null

            objs_text = self.linears_text(ops.cat([phrases_embeddings, xyxy_embedding], axis=-1))
            objs_image = self.linears_image(ops.cat([image_embeddings, xyxy_embedding], axis=-1))
            objs = ops.cat([objs_text, objs_image], axis=1)

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
            conditioning = timesteps_emb + ops.cat([resolution_emb, aspect_ratio_emb], axis=1)
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
        self.linear_1 = nn.Dense(in_channels=in_features, out_channels=hidden_size, has_bias=True)
        if act_fn == "gelu_tanh":
            self.act_1 = nn.GELU(approximate=True)
        elif act_fn == "silu":
            self.act_1 = nn.SiLU()
        elif act_fn == "silu_fp32":
            self.act_1 = FP32SiLU()
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")
        self.linear_2 = nn.Dense(in_channels=hidden_size, out_channels=hidden_size, has_bias=True)

    def construct(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class IPAdapterPlusImageProjection(nn.Cell):
    """Resampler of IP-Adapter Plus.

    Args:
    ----
        embed_dims (int): The feature dimension. Defaults to 768.
        output_dims (int): The number of output channels, that is the same
            number of the channels in the
            `unet.config.cross_attention_dim`. Defaults to 1024.
        hidden_dims (int): The number of hidden channels. Defaults to 1280.
        depth (int): The number of blocks. Defaults to 8.
        dim_head (int): The number of head channels. Defaults to 64.
        heads (int): Parallel attention heads. Defaults to 16.
        num_queries (int): The number of queries. Defaults to 8.
        ffn_ratio (float): The expansion ratio of feedforward network hidden
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
        from .attention import FeedForward
        from .normalization import LayerNorm  # Lazy import to avoid circular import

        self.latents = ms.Parameter(ops.randn(1, num_queries, hidden_dims) / hidden_dims**0.5, name="latents")

        self.proj_in = nn.Dense(embed_dims, hidden_dims)

        self.proj_out = nn.Dense(hidden_dims, output_dims)
        self.norm_out = LayerNorm(output_dims)

        layers = []
        for _ in range(depth):
            layers.append(
                nn.CellList(
                    [
                        LayerNorm(hidden_dims),
                        LayerNorm(hidden_dims),
                        Attention(
                            query_dim=hidden_dims,
                            dim_head=dim_head,
                            heads=heads,
                            out_bias=False,
                        ),
                        nn.SequentialCell(
                            LayerNorm(hidden_dims),
                            FeedForward(hidden_dims, hidden_dims, activation_fn="gelu", mult=ffn_ratio, bias=False),
                        ),
                    ]
                )
            )
        self.layers = nn.CellList(layers)

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        """Forward pass.

        Args:
        ----
            x (ms.Tensor): Input Tensor.

        Returns:
        -------
            ms.Tensor: Output Tensor.
        """
        latents = self.latents.tile((x.size(0), 1, 1))

        x = self.proj_in(x)

        for ln0, ln1, attn, ff in self.layers:
            residual = latents

            encoder_hidden_states = ln0(x)
            latents = ln1(latents)
            encoder_hidden_states = ops.cat([encoder_hidden_states, latents], axis=-2)
            latents = attn(latents, encoder_hidden_states) + residual
            latents = ff(latents) + latents

        latents = self.proj_out(latents)
        return self.norm_out(latents)


class MultiIPAdapterImageProjection(nn.Cell):
    def __init__(self, IPAdapterImageProjectionLayers: Union[List[nn.Cell], Tuple[nn.Cell]]):
        super().__init__()
        self.image_projection_layers = nn.CellList(IPAdapterImageProjectionLayers)

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
