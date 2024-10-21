from typing import Literal, Optional, Tuple

import numpy as np

import mindspore as ms
import mindspore.mint as mint
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, Tensor, load_checkpoint

from mindone.models.utils import normal_, zeros_

from ..activation import ACT2FN
from .layer import (
    CaptionEmbedder,
    LinearPatchEmbed3D,
    LlamaAttention,
    LlamaFlashAttention,
    LlamaMLP,
    LlamaRMSNorm,
    PatchEmbed3D,
    TimestepEmbedder,
)

__all__ = ["LlamaModel", "llama3_1B", "llama3_5B", "llama3_30B"]

Llama_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attention": LlamaFlashAttention,
}


def t2i_modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


class LlamaDecoderLayer(nn.Cell):
    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        rms_norm_eps: float = 1e-5,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
        hidden_act: str = "silu",
        attn_implementation: Literal["eager", "flash_attention"] = "eager",
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__()

        self.self_attn = Llama_ATTENTION_CLASSES[attn_implementation](
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            attention_dropout=attention_dropout,
            attention_bias=attention_bias,
            dtype=dtype,
        )

        self.cross_attn = Llama_ATTENTION_CLASSES[attn_implementation](
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            attention_dropout=attention_dropout,
            attention_bias=attention_bias,
            dtype=dtype,
        )

        self.mlp = LlamaMLP(
            intermediate_size=intermediate_size, hidden_size=hidden_size, hidden_act=hidden_act, dtype=dtype
        )

        self.scale_shift_table = Parameter(Tensor(np.random.randn(6, hidden_size), dtype=dtype) / hidden_size**0.5)

        self.input_layernorm = LlamaRMSNorm(hidden_size, eps=rms_norm_eps, dtype=dtype)
        self.post_attention_layernorm = LlamaRMSNorm(hidden_size, eps=rms_norm_eps, dtype=dtype)

    def construct(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        modulation_parameters: Tensor,
        position_embedding: Tensor,
    ) -> Tensor:
        B = hidden_states.shape[0]

        # 3.1.3 Positional Embedding
        hidden_states = hidden_states + position_embedding.to(hidden_states.dtype)

        # 3.1.3 Adaptive Layer Norm
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ops.chunk(
            ops.unsqueeze(self.scale_shift_table, 0) + modulation_parameters.reshape(B, 6, -1), 6, axis=1
        )

        # Self Attention (Bi-Directional Attention)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = t2i_modulate(hidden_states, shift_msa, scale_msa)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = gate_msa * hidden_states
        hidden_states = residual + hidden_states

        # 3.1.3 Cross Attention
        residual = hidden_states
        hidden_states = self.cross_attn(hidden_states, encoder_hidden_states=encoder_hidden_states)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = t2i_modulate(hidden_states, shift_mlp, scale_mlp)
        hidden_states = self.mlp(hidden_states)
        hidden_states = gate_mlp * hidden_states
        hidden_states = residual + hidden_states

        return hidden_states


class LlamaFinalLayer(nn.Cell):
    def __init__(
        self,
        hidden_size: int = 4096,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        out_channels: int = 8,
        rms_norm_eps: float = 1e-5,
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__()
        self.input_layernorm = LlamaRMSNorm(hidden_size, eps=rms_norm_eps, dtype=dtype)
        self.proj = nn.Dense(
            hidden_size, patch_size[0] * patch_size[1] * patch_size[2] * out_channels, has_bias=False, dtype=dtype
        )
        self.scale_shift_table = Parameter(Tensor(np.random.randn(2, hidden_size), dtype=dtype) / hidden_size**0.5)

    def construct(self, hidden_states: Tensor, timestep_embedding: Tensor):
        shift, scale = ops.chunk(
            ops.unsqueeze(self.scale_shift_table, 0) + ops.unsqueeze(timestep_embedding, 1), 2, axis=1
        )
        hidden_states = t2i_modulate(self.input_layernorm(hidden_states), shift, scale)
        hidden_states = self.proj(hidden_states)
        return hidden_states


class LlamaModel(nn.Cell):
    def __init__(
        self,
        in_channels: int = 8,
        out_channels: Optional[int] = None,
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        num_attention_heads: int = 32,
        num_hidden_layers: int = 32,
        num_key_value_heads: int = 8,
        rms_norm_eps: float = 1e-5,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
        hidden_act: str = "silu",
        initializer_range: float = 0.02,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        max_length: Tuple[int, int, int] = (128, 64, 64),
        caption_channels: int = 4096,
        attn_implementation: Literal["eager", "flash_attention"] = "eager",
        gradient_checkpointing: bool = False,
        use_linear_patch_embedder: bool = True,
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_length = max_length

        self.layers = nn.CellList(
            [
                LlamaDecoderLayer(
                    hidden_size=self.hidden_size,
                    intermediate_size=intermediate_size,
                    num_attention_heads=self.num_attention_heads,
                    num_key_value_heads=self.num_key_value_heads,
                    rms_norm_eps=rms_norm_eps,
                    attention_dropout=attention_dropout,
                    attention_bias=attention_bias,
                    hidden_act=hidden_act,
                    attn_implementation=attn_implementation,
                    dtype=dtype,
                )
                for _ in range(num_hidden_layers)
            ]
        )
        self.final_layer = LlamaFinalLayer(
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            out_channels=self.out_channels,
            rms_norm_eps=rms_norm_eps,
            dtype=dtype,
        )

        self.pos_embedding_table_t = nn.Embedding(max_length[0], self.hidden_size, dtype=dtype)
        self.pos_embedding_table_h = nn.Embedding(max_length[1], self.hidden_size, dtype=dtype)
        self.pos_embedding_table_w = nn.Embedding(max_length[2], self.hidden_size, dtype=dtype)

        if use_linear_patch_embedder:
            self.latent_embedder = LinearPatchEmbed3D(self.patch_size, self.in_channels, self.hidden_size, dtype=dtype)
        else:
            self.latent_embedder = PatchEmbed3D(self.patch_size, self.in_channels, self.hidden_size, dtype=dtype)

        self.timestep_embedder = TimestepEmbedder(self.hidden_size, dtype=dtype)
        self.adaLN_modulation = nn.SequentialCell(
            ACT2FN[hidden_act], mint.nn.Linear(self.hidden_size, 6 * self.hidden_size, bias=False, dtype=dtype)
        )
        self.caption_embedder = CaptionEmbedder(caption_channels, self.hidden_size, eps=rms_norm_eps, dtype=dtype)

        # post-init
        self.initializer_range = initializer_range
        self.init_weights()

        # recompute
        if gradient_checkpointing:
            self.layers.recompute()

    def init_weights(self):
        std = self.initializer_range

        def _init_weights(module):
            if isinstance(module, mint.nn.Linear):
                normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    zeros_(module.weight)
            elif isinstance(module, nn.Embedding):
                normal_(module.embedding_table, mean=0.0, std=std)

        self.apply(_init_weights)

        # Initialize patch_embed like nn.Dense (instead of nn.Conv3d):
        normal_(self.latent_embedder.proj.weight, mean=0.0, std=std)
        if self.latent_embedder.proj.bias is not None:
            zeros_(self.latent_embedder.proj.bias)

        # Zero-out adaLN modulation layer:
        zeros_(self.adaLN_modulation[-1].weight)
        if self.adaLN_modulation[-1].bias is not None:
            zeros_(self.adaLN_modulation[-1].bias)

        # Zero-out final layer as DiT does
        zeros_(self.final_layer.proj.weight)
        if self.final_layer.proj.bias is not None:
            zeros_(self.final_layer.proj.bias)

    def learnable_position_embedding(self, latent_embedding: Tensor) -> Tensor:
        # 3.1.3
        _, t, _, h, w = latent_embedding.shape
        p0, p1, p2 = self.patch_size[0], self.patch_size[1], self.patch_size[2]
        nt, nh, nw = t // p0, h // p1, w // p2

        assert nt < self.max_length[0]
        assert nh < self.max_length[1]
        assert nw < self.max_length[2]

        t_inds = mint.arange(nt, dtype=ms.int64)
        h_inds = mint.arange(nh, dtype=ms.int64)
        w_inds = mint.arange(nw, dtype=ms.int64)

        position_ids = ops.meshgrid(t_inds, h_inds, w_inds, indexing="ij")
        position_ids = ops.stack(position_ids, axis=-1)
        position_ids = ops.reshape(position_ids, (-1, 3))

        t_inds, h_inds, w_inds = ops.unbind(position_ids, dim=-1)
        pos_embed_t = self.pos_embedding_table_t(t_inds)
        pos_embed_h = self.pos_embedding_table_h(h_inds)
        pos_embed_w = self.pos_embedding_table_w(w_inds)
        return pos_embed_t + pos_embed_h + pos_embed_w

    def unpatchify(self, hidden_states: Tensor, t: int, h: int, w: int) -> Tensor:
        """
        hidden_states: (N, T, patch_size[0] * patch_size[1] * patch_size[2] * C)
        """
        bs = hidden_states.shape[0]
        c = self.out_channels
        p0, p1, p2 = self.patch_size[0], self.patch_size[1], self.patch_size[2]
        nt, nh, nw = t // p0, h // p1, w // p2

        hidden_states = ops.reshape(hidden_states, (bs, nt, nh, nw, p0, p1, p2, c))
        # bs, nt, p0, c, nh, p1, nw, p2, c
        hidden_states = mint.permute(hidden_states, (0, 1, 4, 7, 2, 5, 3, 6))
        output = ops.reshape(hidden_states, (bs, nt * p0, c, nh * p1, nw * p2))
        return output

    def construct(
        self,
        latent_embedding: Tensor,
        timestep: Tensor,
        text_embedding: Tensor,
    ) -> Tensor:
        """
        latent_embedding: (N, T, C, H, W) tensor of inputs (latent representations of video)
        timestep: (N,) tensor to indicate denoising step
        text_embedding: (N, L, C') tensor of the text embedding
        """
        _, t, _, h, w = latent_embedding.shape

        # create position embedding to be shared across the decoder layers
        position_embedding = self.learnable_position_embedding(latent_embedding)

        # patchify and embed latent in transformer hidden dim.
        latent_embedding = self.latent_embedder(latent_embedding)

        # 6.1.2 shared timestep embedding & modulation. It does not mention the detail structure, we follow PixArt-Alpha here
        timestep_embedding = self.timestep_embedder(timestep)
        modulation_parameters = self.adaLN_modulation(timestep_embedding)

        # 3.1.4 text embedding
        text_embedding = self.caption_embedder(text_embedding)

        # main block
        hidden_states = latent_embedding
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, text_embedding, modulation_parameters, position_embedding)

        # final layer
        hidden_states = self.final_layer(hidden_states, timestep_embedding)

        # unpatchify
        output = self.unpatchify(hidden_states, t, h, w)
        return output


def llama3_1B(from_pretrained=None, **kwargs):
    model = LlamaModel(
        attention_bias=False,
        attention_dropout=0.0,
        hidden_act="silu",
        hidden_size=1536,
        initializer_range=0.02,
        intermediate_size=4096,
        num_attention_heads=16,
        num_hidden_layers=24,
        num_key_value_heads=16,
        rms_norm_eps=1e-05,
        **kwargs,
    )
    if from_pretrained is not None:
        load_checkpoint(from_pretrained, model)
    return model


def llama3_5B(from_pretrained=None, **kwargs):
    model = LlamaModel(
        attention_bias=False,
        attention_dropout=0.0,
        hidden_act="silu",
        hidden_size=3072,
        initializer_range=0.02,
        intermediate_size=8192,
        num_attention_heads=24,
        num_hidden_layers=32,
        num_key_value_heads=24,
        rms_norm_eps=1e-05,
        **kwargs,
    )
    if from_pretrained is not None:
        load_checkpoint(from_pretrained, model)
    return model


def llama3_30B(from_pretrained=None, **kwargs):
    model = LlamaModel(
        attention_bias=False,
        attention_dropout=0.0,
        hidden_act="silu",
        hidden_size=6144,
        initializer_range=0.02,
        intermediate_size=16384,
        num_attention_heads=48,
        num_hidden_layers=48,
        num_key_value_heads=48,
        rms_norm_eps=1e-05,
        **kwargs,
    )
    if from_pretrained is not None:
        load_checkpoint(from_pretrained, model)
    return model
