# Adapted from https://github.com/VectorSpaceLab/OmniGen/blob/main/OmniGen/model.py

import math
import numbers

import numpy as np
from omnigen.transformer import Phi3Transformer

import mindspore as ms
from mindspore import Parameter, nn, ops
from mindspore.common.initializer import initializer

from mindone.diffusers.loaders import PeftAdapterMixin


class SiLU(nn.Cell):
    def __init__(self, upcast=True):
        super(SiLU, self).__init__()
        self.sigmoid = ops.Sigmoid()
        self.upcast = upcast

    def construct(self, x):
        if self.upcast:
            # force sigmoid to use fp32
            return x * self.sigmoid(x.astype(ms.float32)).astype(x.dtype)
        else:
            return x * self.sigmoid(x)


class LayerNorm(nn.Cell):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine: bool = True, dtype: ms.dtype = ms.float32):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.gamma = Parameter(initializer("ones", normalized_shape, dtype=dtype))
            self.beta = Parameter(initializer("zeros", normalized_shape, dtype=dtype))
        else:
            self.gamma = ops.ones(normalized_shape, dtype=dtype)
            self.beta = ops.zeros(normalized_shape, dtype=dtype)
        self.layer_norm = ops.LayerNorm(-1, -1, epsilon=eps)

    def construct(self, x: ms.Tensor):
        oridtype = x.dtype
        x, _, _ = self.layer_norm(x.to(ms.float32), self.gamma.to(ms.float32), self.beta.to(ms.float32))
        return x.to(oridtype)


def modulate(x, shift, scale):
    """Apply modulation to features"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Cell):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.SequentialCell(
            [
                nn.Dense(frequency_embedding_size, hidden_size, has_bias=True),
                SiLU(),
                nn.Dense(hidden_size, hidden_size, has_bias=True),
            ]
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = ops.exp(-math.log(max_period) * ops.arange(0, half, dtype=ms.float32) / half)

        args = t[:, None].float() * freqs[None]
        embedding = ops.cat([ops.cos(args), ops.sin(args)], axis=-1)
        if dim % 2:
            embedding = ops.cat([embedding, ops.zeros_like(embedding[:, :1])], axis=-1)
        return embedding

    def construct(self, t, dtype=ms.float32):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).astype(dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class FinalLayer(nn.Cell):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Dense(hidden_size, patch_size * patch_size * out_channels, has_bias=True)
        self.adaLN_modulation = nn.SequentialCell(SiLU(), nn.Dense(hidden_size, 2 * hidden_size, has_bias=True))

    def construct(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, axis=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


def get_2d_sincos_pos_embed(
    embed_dim, grid_size, cls_token=False, extra_tokens=0, interpolation_scale=1.0, base_size=1
):
    """Generate 2D positional embeddings"""
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)

    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0] / base_size) / interpolation_scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1] / base_size) / interpolation_scale
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


class PatchEmbedMR(nn.Cell):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        patch_size: int = 2,
        in_chans: int = 4,
        embed_dim: int = 768,
        has_bias: bool = True,
    ):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, has_bias=has_bias)

    def construct(self, x):
        x = self.proj(x)
        x = x.flatten(start_dim=2)  # NCHW -> NLC
        x = ops.swapaxes(x, 1, 2)

        return x


class OmniGen(nn.Cell, PeftAdapterMixin):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        transformer_config,
        patch_size=2,
        in_channels=4,
        pe_interpolation: float = 1.0,
        pos_embed_max_size: int = 192,
    ):
        super().__init__(transformer_config)
        self.transformer_config = transformer_config
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.pos_embed_max_size = pos_embed_max_size

        hidden_size = transformer_config.hidden_size

        self.x_embedder = PatchEmbedMR(patch_size, in_channels, hidden_size, has_bias=True)
        self.input_x_embedder = PatchEmbedMR(patch_size, in_channels, hidden_size, has_bias=True)

        self.time_token = TimestepEmbedder(hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)

        self.pe_interpolation = pe_interpolation
        pos_embed = get_2d_sincos_pos_embed(
            hidden_size, pos_embed_max_size, interpolation_scale=pe_interpolation, base_size=64
        )

        self.pos_embed = ms.Parameter(ms.Tensor(pos_embed[None, ...], dtype=ms.float32), requires_grad=False)
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.llm = Phi3Transformer(transformer_config)
        self.llm.config.use_cache = False

    def unpatchify(self, x, h, w):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        x = ops.reshape(
            x, (x.shape[0], h // self.patch_size, w // self.patch_size, self.patch_size, self.patch_size, c)
        )
        x = ops.transpose(x, (0, 5, 1, 3, 2, 4))
        imgs = ops.reshape(x, (x.shape[0], c, h, w))
        return imgs

    def cropped_pos_embed(self, height, width):
        """Crops positional embeddings for compatibility"""
        if self.pos_embed_max_size is None:
            raise ValueError("`pos_embed_max_size` must be set for cropping.")

        height = height // self.patch_size
        width = width // self.patch_size
        if height > self.pos_embed_max_size:
            raise ValueError(
                f"Height ({height}) cannot be greater than 'pos_embed_max_size': {self.pos_embed_max_size}."
            )
        if width > self.pos_embed_max_size:
            raise ValueError(f"Width ({width}) cannot be greater than 'pos_embed_max_size': {self.pos_embed_max_size}.")

        top = (self.pos_embed_max_size - height) // 2
        left = (self.pos_embed_max_size - width) // 2
        spatial_pos_embed = ops.reshape(self.pos_embed, (1, self.pos_embed_max_size, self.pos_embed_max_size, -1))
        spatial_pos_embed = spatial_pos_embed[:, top : top + height, left : left + width, :]
        spatial_pos_embed = ops.reshape(spatial_pos_embed, (1, -1, spatial_pos_embed.shape[-1]))
        return spatial_pos_embed

    def patch_multiple_resolutions(self, latents, padding_latent=None, is_input_images: bool = False):
        """Process inputs at multiple resolutions"""
        if isinstance(latents, (list, tuple)):
            # Handle list of inputs
            return_list = padding_latent is None
            if padding_latent is None:
                padding_latent = [None] * len(latents)
                return_list = True

            patched_latents, num_tokens, shapes = [], [], []
            for latent, padding in zip(latents, padding_latent):
                height, width = latent.shape[-2:]
                if is_input_images:
                    latent = self.input_x_embedder(latent)
                else:
                    latent = self.x_embedder(latent)
                pos_embed = self.cropped_pos_embed(height, width)
                latent = latent + pos_embed

                if padding is not None:
                    latent = ops.concat([latent, padding], axis=-2)

                patched_latents.append(latent)
                num_tokens.append(pos_embed.shape[1])
                shapes.append([height, width])

            if not return_list:
                latents = ops.concat(patched_latents, axis=0)
            else:
                latents = patched_latents

        else:
            # Handle single input
            # Handle single input (continued)
            height, width = latents.shape[-2:]

            if is_input_images:
                latents = self.input_x_embedder(latents)
            else:
                latents = self.x_embedder(latents)

            pos_embed = self.cropped_pos_embed(height, width).to(latents.dtype)
            latents = latents + pos_embed
            num_tokens = pos_embed.shape[1]
            shapes = [height, width]

        return latents, num_tokens, shapes

    def construct(
        self,
        x,
        timestep,
        input_ids,
        attention_mask,
        position_ids,
        input_img_latents=[],
        input_image_sizes={},
        padding_latents=[None],
        past_key_values=None,
        return_past_key_values=False,
    ):
        """Forward pass of the model"""
        input_is_list = isinstance(x, (list, tuple))

        x, num_tokens, shapes = self.patch_multiple_resolutions(x, padding_latents)
        time_token = self.time_token(timestep, dtype=x[0].dtype).unsqueeze(1)
        # if input_img_latents is not None:
        input_latents, _, _ = self.patch_multiple_resolutions(input_img_latents, is_input_images=True)
        if input_ids is not None:
            condition_embeds = self.llm.embed_tokens(input_ids)
            input_img_inx = 0
            for b_inx in input_image_sizes.keys():
                for start_inx, end_inx in input_image_sizes[b_inx]:
                    latent = input_latents[input_img_inx]
                    latent_reshaped = latent.reshape((end_inx - start_inx, latent.shape[-1]))
                    condition_embeds[b_inx, start_inx:end_inx] = latent_reshaped

                    input_img_inx += 1

            if input_img_latents is not None:
                assert input_img_inx == len(input_latents)

            inputs_embs = ops.cat((condition_embeds, time_token, x), axis=1)
        else:
            inputs_embs = ops.cat((time_token, x), axis=1)

        output_llm = self.llm(
            inputs_embeds=inputs_embs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

        # output, past_key_values = output.last_hidden_state, output.past_key_values
        output, past_key_values = output_llm[0], output_llm[1]
        if input_is_list:
            image_embedding = output[:, -max(num_tokens) :]
            time_emb = self.t_embedder(timestep, dtype=x.dtype)
            x = self.final_layer(image_embedding, time_emb)

            latents = []
            for i in range(x.shape[0]):
                latent = x[i : i + 1, : num_tokens[i]]
                latent = self.unpatchify(latent, shapes[i][0], shapes[i][1])
                latents.append(latent)
        else:
            image_embedding = output[:, -num_tokens:]
            time_emb = self.t_embedder(timestep, dtype=x.dtype)
            x = self.final_layer(image_embedding, time_emb)
            latents = self.unpatchify(x, shapes[0], shapes[1])

        if return_past_key_values:
            return latents, past_key_values
        return latents

    def forward_with_cfg(
        self,
        x,
        timestep,
        input_ids,
        attention_mask,
        position_ids,
        input_img_latents,
        input_image_sizes,
        cfg_scale,
        use_img_cfg,
        img_cfg_scale,
        past_key_values,
        use_kv_cache,
    ):
        """Forward pass with classifier-free guidance"""
        self.llm.config.use_cache = use_kv_cache
        model_out, past_key_values = self.construct(
            x,
            timestep,
            input_ids,
            attention_mask,
            position_ids,
            input_img_latents,
            input_image_sizes,
            past_key_values=past_key_values,
            return_past_key_values=True,
        )

        if use_img_cfg:
            # Split for conditional, unconditional, and image-conditional
            cond, uncond, img_cond = ops.split(model_out, len(model_out) // 3, axis=0)
            cond = uncond + img_cfg_scale * (img_cond - uncond) + cfg_scale * (cond - img_cond)
            model_out = [cond, cond, cond]
        else:
            # Split for conditional and unconditional
            cond, uncond = ops.split(model_out, len(model_out) // 2, axis=0)
            cond = uncond + cfg_scale * (cond - uncond)
            model_out = [cond, cond]

        return ops.concat(model_out, axis=0), past_key_values

    def forward_with_separate_cfg(
        self,
        x,
        timestep,
        input_ids,
        attention_mask,
        position_ids,
        input_img_latents,
        input_image_sizes,
        cfg_scale,
        use_img_cfg,
        img_cfg_scale,
        past_key_values,
        use_kv_cache,
    ):
        """Forward pass with separate classifier-free guidance computation"""
        self.llm.config.use_cache = use_kv_cache
        if past_key_values is None:
            past_key_values = [None] * len(attention_mask)

        x = ops.split(x, len(x) // len(attention_mask), axis=0)
        timestep = timestep.astype(x[0].dtype)
        timestep = ops.split(timestep, len(timestep) // len(input_ids), axis=0)

        model_out, past_key_values_list = [], []
        for i in range(len(input_ids)):
            temp_out, temp_past_key_values = self.construct(
                x[i],
                timestep[i],
                input_ids[i],
                attention_mask[i],
                position_ids[i],
                input_img_latents[i],
                input_image_sizes[i],
                past_key_values=past_key_values[i],
                return_past_key_values=True,
            )
            model_out.append(temp_out)
            past_key_values_list.append(temp_past_key_values)

        if len(model_out) == 3:
            cond, uncond, img_cond = model_out
            cond = uncond + img_cfg_scale * (img_cond - uncond) + cfg_scale * (cond - img_cond)
            model_out = [cond, cond, cond]
        elif len(model_out) == 2:
            cond, uncond = model_out
            cond = uncond + cfg_scale * (cond - uncond)
            model_out = [cond, cond]
        else:
            return model_out[0]

        return ops.concat(model_out, axis=0), past_key_values_list
