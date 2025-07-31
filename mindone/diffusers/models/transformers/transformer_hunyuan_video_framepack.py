# Copyright 2025 The Framepack Team, The Hunyuan Team and The HuggingFace Team. All rights reserved.
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

from typing import Any, Dict, List, Optional, Tuple

import mindspore as ms
from mindspore import mint, nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ...utils import get_logger
from ..embeddings import get_1d_rotary_pos_embed
from ..layers_compat import unflatten
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import AdaLayerNormContinuous
from .transformer_hunyuan_video import (
    HunyuanVideoConditionEmbedding,
    HunyuanVideoPatchEmbed,
    HunyuanVideoSingleTransformerBlock,
    HunyuanVideoTokenRefiner,
    HunyuanVideoTransformerBlock,
)

logger = get_logger(__name__)  # pylint: disable=invalid-name


class HunyuanVideoFramepackRotaryPosEmbed(nn.Cell):
    def __init__(self, patch_size: int, patch_size_t: int, rope_dim: List[int], theta: float = 256.0) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.rope_dim = rope_dim
        self.theta = theta

    def construct(self, frame_indices: ms.Tensor, height: int, width: int):
        height = height // self.patch_size
        width = width // self.patch_size
        grid = mint.meshgrid(
            frame_indices.to(dtype=ms.float32),
            mint.arange(0, height, dtype=ms.float32),
            mint.arange(0, width, dtype=ms.float32),
            indexing="ij",
        )  # 3 * [W, H, T]
        grid = mint.stack(grid, dim=0)  # [3, W, H, T]

        freqs = []
        for i in range(3):
            freq = get_1d_rotary_pos_embed(self.rope_dim[i], grid[i].reshape(-1), self.theta, use_real=True)
            freqs.append(freq)

        freqs_cos = mint.cat([f[0] for f in freqs], dim=1)  # (W * H * T, D / 2)
        freqs_sin = mint.cat([f[1] for f in freqs], dim=1)  # (W * H * T, D / 2)

        return freqs_cos, freqs_sin


class FramepackClipVisionProjection(nn.Cell):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = mint.nn.Linear(in_channels, out_channels * 3)
        self.down = mint.nn.Linear(out_channels * 3, out_channels)

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        hidden_states = self.up(hidden_states)
        hidden_states = mint.nn.functional.silu(hidden_states)
        hidden_states = self.down(hidden_states)
        return hidden_states


class HunyuanVideoHistoryPatchEmbed(nn.Cell):
    def __init__(self, in_channels: int, inner_dim: int):
        super().__init__()
        self.proj = mint.nn.Conv3d(in_channels, inner_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.proj_2x = mint.nn.Conv3d(in_channels, inner_dim, kernel_size=(2, 4, 4), stride=(2, 4, 4))
        self.proj_4x = mint.nn.Conv3d(in_channels, inner_dim, kernel_size=(4, 8, 8), stride=(4, 8, 8))

    def construct(
        self,
        latents_clean: Optional[ms.Tensor] = None,
        latents_clean_2x: Optional[ms.Tensor] = None,
        latents_clean_4x: Optional[ms.Tensor] = None,
    ):
        if latents_clean is not None:
            latents_clean = self.proj(latents_clean)
            latents_clean = latents_clean.flatten(2).swapaxes(1, 2)
        if latents_clean_2x is not None:
            latents_clean_2x = _pad_for_3d_conv(latents_clean_2x, (2, 4, 4))
            latents_clean_2x = self.proj_2x(latents_clean_2x)
            latents_clean_2x = latents_clean_2x.flatten(2).swapaxes(1, 2)
        if latents_clean_4x is not None:
            latents_clean_4x = _pad_for_3d_conv(latents_clean_4x, (4, 8, 8))
            latents_clean_4x = self.proj_4x(latents_clean_4x)
            latents_clean_4x = latents_clean_4x.flatten(2).swapaxes(1, 2)
        return latents_clean, latents_clean_2x, latents_clean_4x


class HunyuanVideoFramepackTransformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["x_embedder", "context_embedder", "norm"]
    _no_split_modules = [
        "HunyuanVideoTransformerBlock",
        "HunyuanVideoSingleTransformerBlock",
        "HunyuanVideoHistoryPatchEmbed",
        "HunyuanVideoTokenRefiner",
    ]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        num_layers: int = 20,
        num_single_layers: int = 40,
        num_refiner_layers: int = 2,
        mlp_ratio: float = 4.0,
        patch_size: int = 2,
        patch_size_t: int = 1,
        qk_norm: str = "rms_norm",
        guidance_embeds: bool = True,
        text_embed_dim: int = 4096,
        pooled_projection_dim: int = 768,
        rope_theta: float = 256.0,
        rope_axes_dim: Tuple[int] = (16, 56, 56),
        image_condition_type: Optional[str] = None,
        has_image_proj: int = False,
        image_proj_dim: int = 1152,
        has_clean_x_embedder: int = False,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Latent and condition embedders
        self.x_embedder = HunyuanVideoPatchEmbed((patch_size_t, patch_size, patch_size), in_channels, inner_dim)

        # Framepack history projection embedder
        self.clean_x_embedder = None
        if has_clean_x_embedder:
            self.clean_x_embedder = HunyuanVideoHistoryPatchEmbed(in_channels, inner_dim)

        self.context_embedder = HunyuanVideoTokenRefiner(
            text_embed_dim, num_attention_heads, attention_head_dim, num_layers=num_refiner_layers
        )

        # Framepack image-conditioning embedder
        self.image_projection = FramepackClipVisionProjection(image_proj_dim, inner_dim) if has_image_proj else None

        self.time_text_embed = HunyuanVideoConditionEmbedding(
            inner_dim, pooled_projection_dim, guidance_embeds, image_condition_type
        )

        # 2. RoPE
        self.rope = HunyuanVideoFramepackRotaryPosEmbed(patch_size, patch_size_t, rope_axes_dim, rope_theta)

        # 3. Dual stream transformer blocks
        self.transformer_blocks = nn.CellList(
            [
                HunyuanVideoTransformerBlock(
                    num_attention_heads, attention_head_dim, mlp_ratio=mlp_ratio, qk_norm=qk_norm
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Single stream transformer blocks
        self.single_transformer_blocks = nn.CellList(
            [
                HunyuanVideoSingleTransformerBlock(
                    num_attention_heads, attention_head_dim, mlp_ratio=mlp_ratio, qk_norm=qk_norm
                )
                for _ in range(num_single_layers)
            ]
        )

        # 5. Output projection
        self.norm_out = AdaLayerNormContinuous(inner_dim, inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = mint.nn.Linear(inner_dim, patch_size_t * patch_size * patch_size * out_channels)

        self.gradient_checkpointing = False

        self.config_patch_size = self.config.patch_size
        self.config_patch_size_t = self.config.patch_size_t

    def construct(
        self,
        hidden_states: ms.Tensor,
        timestep: ms.Tensor,
        encoder_hidden_states: ms.Tensor,
        encoder_attention_mask: ms.Tensor,
        pooled_projections: ms.Tensor,
        image_embeds: ms.Tensor,
        indices_latents: ms.Tensor,
        guidance: Optional[ms.Tensor] = None,
        latents_clean: Optional[ms.Tensor] = None,
        indices_latents_clean: Optional[ms.Tensor] = None,
        latents_history_2x: Optional[ms.Tensor] = None,
        indices_latents_history_2x: Optional[ms.Tensor] = None,
        latents_history_4x: Optional[ms.Tensor] = None,
        indices_latents_history_4x: Optional[ms.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = False,
    ):
        if attention_kwargs is not None and "scale" in attention_kwargs:
            # weight the lora layers by setting `lora_scale` for each PEFT layer here
            # and remove `lora_scale` from each PEFT layer at the end.
            # scale_lora_layers & unscale_lora_layers maybe contains some operation forbidden in graph mode
            raise RuntimeError(
                f"You are trying to set scaling of lora layer by passing {attention_kwargs['scale']=}. "
                f"However it's not allowed in on-the-fly model forwarding. "
                f"Please manually call `scale_lora_layers(model, lora_scale)` before model forwarding and "
                f"`unscale_lora_layers(model, lora_scale)` after model forwarding. "
                f"For example, it can be done in a pipeline call like `StableDiffusionPipeline.__call__`."
            )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p, p_t = self.config_patch_size, self.config_patch_size_t
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p
        post_patch_width = width // p
        original_context_length = post_patch_num_frames * post_patch_height * post_patch_width

        if indices_latents is None:
            indices_latents = mint.arange(0, num_frames).unsqueeze(0).broadcast_to((batch_size, -1))

        hidden_states = self.x_embedder(hidden_states)
        image_rotary_emb = self.rope(frame_indices=indices_latents, height=height, width=width)

        latents_clean, latents_history_2x, latents_history_4x = self.clean_x_embedder(
            latents_clean, latents_history_2x, latents_history_4x
        )

        if latents_clean is not None and indices_latents_clean is not None:
            image_rotary_emb_clean = self.rope(frame_indices=indices_latents_clean, height=height, width=width)
        else:
            image_rotary_emb_clean = None
        if latents_history_2x is not None and indices_latents_history_2x is not None:
            image_rotary_emb_history_2x = self.rope(
                frame_indices=indices_latents_history_2x, height=height, width=width
            )
        else:
            image_rotary_emb_history_2x = None
        if latents_history_4x is not None and indices_latents_history_4x is not None:
            image_rotary_emb_history_4x = self.rope(
                frame_indices=indices_latents_history_4x, height=height, width=width
            )
        else:
            image_rotary_emb_history_4x = None

        hidden_states, image_rotary_emb = self._pack_history_states(
            hidden_states,
            latents_clean,
            latents_history_2x,
            latents_history_4x,
            image_rotary_emb,
            image_rotary_emb_clean,
            image_rotary_emb_history_2x,
            image_rotary_emb_history_4x,
            post_patch_height,
            post_patch_width,
        )

        temb, _ = self.time_text_embed(timestep, pooled_projections, guidance)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states, timestep, encoder_attention_mask)

        encoder_hidden_states_image = self.image_projection(image_embeds)
        attention_mask_image = encoder_attention_mask.new_ones((batch_size, encoder_hidden_states_image.shape[1]))

        # must cat before (not after) encoder_hidden_states, due to attn masking
        encoder_hidden_states = mint.cat([encoder_hidden_states_image, encoder_hidden_states], dim=1)
        encoder_attention_mask = mint.cat([attention_mask_image, encoder_attention_mask], dim=1)

        latent_sequence_length = hidden_states.shape[1]
        condition_sequence_length = encoder_hidden_states.shape[1]
        sequence_length = latent_sequence_length + condition_sequence_length
        attention_mask = mint.zeros((batch_size, sequence_length), dtype=ms.bool_)  # [B, N]
        effective_condition_sequence_length = encoder_attention_mask.sum(dim=1, dtype=ms.int_)  # [B,]
        effective_sequence_length = latent_sequence_length + effective_condition_sequence_length

        if batch_size == 1:
            encoder_hidden_states = encoder_hidden_states[:, : effective_condition_sequence_length[0]]
            attention_mask = None
        else:
            for i in range(batch_size):
                attention_mask[i, : effective_sequence_length[i]] = True
            # [B, 1, 1, N], for broadcasting across attention heads
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)

        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states, encoder_hidden_states, temb, attention_mask, image_rotary_emb
            )

        for block in self.single_transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states, encoder_hidden_states, temb, attention_mask, image_rotary_emb
            )

        hidden_states = hidden_states[:, -original_context_length:]
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1, p_t, p, p
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (hidden_states,)
        return Transformer2DModelOutput(sample=hidden_states)

    def _pack_history_states(
        self,
        hidden_states: ms.Tensor,
        latents_clean: Optional[ms.Tensor] = None,
        latents_history_2x: Optional[ms.Tensor] = None,
        latents_history_4x: Optional[ms.Tensor] = None,
        image_rotary_emb: Tuple[ms.Tensor, ms.Tensor] = None,
        image_rotary_emb_clean: Optional[Tuple[ms.Tensor, ms.Tensor]] = None,
        image_rotary_emb_history_2x: Optional[Tuple[ms.Tensor, ms.Tensor]] = None,
        image_rotary_emb_history_4x: Optional[Tuple[ms.Tensor, ms.Tensor]] = None,
        height: int = None,
        width: int = None,
    ):
        image_rotary_emb = list(image_rotary_emb)  # convert tuple to list for in-place modification

        if latents_clean is not None and image_rotary_emb_clean is not None:
            hidden_states = mint.cat([latents_clean, hidden_states], dim=1)
            image_rotary_emb[0] = mint.cat([image_rotary_emb_clean[0], image_rotary_emb[0]], dim=0)
            image_rotary_emb[1] = mint.cat([image_rotary_emb_clean[1], image_rotary_emb[1]], dim=0)

        if latents_history_2x is not None and image_rotary_emb_history_2x is not None:
            hidden_states = mint.cat([latents_history_2x, hidden_states], dim=1)
            image_rotary_emb_history_2x = self._pad_rotary_emb(image_rotary_emb_history_2x, height, width, (2, 2, 2))
            image_rotary_emb[0] = mint.cat([image_rotary_emb_history_2x[0], image_rotary_emb[0]], dim=0)
            image_rotary_emb[1] = mint.cat([image_rotary_emb_history_2x[1], image_rotary_emb[1]], dim=0)

        if latents_history_4x is not None and image_rotary_emb_history_4x is not None:
            hidden_states = mint.cat([latents_history_4x, hidden_states], dim=1)
            image_rotary_emb_history_4x = self._pad_rotary_emb(image_rotary_emb_history_4x, height, width, (4, 4, 4))
            image_rotary_emb[0] = mint.cat([image_rotary_emb_history_4x[0], image_rotary_emb[0]], dim=0)
            image_rotary_emb[1] = mint.cat([image_rotary_emb_history_4x[1], image_rotary_emb[1]], dim=0)

        return hidden_states, tuple(image_rotary_emb)

    def _pad_rotary_emb(
        self,
        image_rotary_emb: Tuple[ms.Tensor],
        height: int,
        width: int,
        kernel_size: Tuple[int, int, int],
    ):
        # freqs_cos, freqs_sin have shape [W * H * T, D / 2], where D is attention head dim
        freqs_cos, freqs_sin = image_rotary_emb
        freqs_cos = unflatten(freqs_cos.unsqueeze(0).permute(0, 2, 1), 2, (-1, height, width))
        freqs_sin = unflatten(freqs_sin.unsqueeze(0).permute(0, 2, 1), 2, (-1, height, width))
        freqs_cos = _pad_for_3d_conv(freqs_cos, kernel_size)
        freqs_sin = _pad_for_3d_conv(freqs_sin, kernel_size)
        freqs_cos = _center_down_sample_3d(freqs_cos, kernel_size)
        freqs_sin = _center_down_sample_3d(freqs_sin, kernel_size)
        freqs_cos = freqs_cos.flatten(2).permute(0, 2, 1).squeeze(0)
        freqs_sin = freqs_sin.flatten(2).permute(0, 2, 1).squeeze(0)
        return freqs_cos, freqs_sin


def _pad_for_3d_conv(x, kernel_size):
    if isinstance(x, (tuple, list)):
        return tuple(_pad_for_3d_conv(i, kernel_size) for i in x)
    b, c, t, h, w = x.shape
    pt, ph, pw = kernel_size
    pad_t = (pt - (t % pt)) % pt
    pad_h = (ph - (h % ph)) % ph
    pad_w = (pw - (w % pw)) % pw
    # TODO: bfloat16 is not supported in mint.nn.functional.pad
    return mint.nn.functional.pad(x.float(), (0, pad_w, 0, pad_h, 0, pad_t), mode="replicate").to(x.dtype)


def _center_down_sample_3d(x, kernel_size):
    if isinstance(x, (tuple, list)):
        return tuple(_center_down_sample_3d(i, kernel_size) for i in x)
    return mint.nn.functional.avg_pool3d(x, kernel_size, stride=kernel_size)
