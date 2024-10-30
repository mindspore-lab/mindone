from __future__ import annotations

from typing import Literal, Optional, Tuple, Union

import numpy as np
from moviegen.parallel import GatherForwardSplitBackward, SplitForwardGatherBackward
from moviegen.parallel.parallel_states import get_model_parallel_group

import mindspore as ms
import mindspore.mint as mint
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, Tensor, load_checkpoint, load_param_into_net
from mindspore.communication import GlobalComm, get_group_size

from mindone.models.utils import normal_, zeros_

from .activation import ACT2FN
from .block import (
    CaptionEmbedder,
    ContextParallelLlamaAttention,
    ContextParallelLlamaFlashAttention,
    FusedTensorParallelLlamaMLP,
    LinearPatchEmbed3D,
    LlamaAttention,
    LlamaFlashAttention,
    LlamaMLP,
    LlamaRMSNorm,
    PatchEmbed3D,
    TensorParallelLlamaMLP,
    TimestepEmbedder,
)

__all__ = ["LlamaModel", "llama3_1B", "llama3_5B", "llama3_30B"]

Llama_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attention": LlamaFlashAttention,
}

CONTEXT_PARALLEL_Llama_ATTENTION_CLASSES = {
    "eager": ContextParallelLlamaAttention,
    "flash_attention": ContextParallelLlamaFlashAttention,
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

        self.scale_shift_table = Parameter(Tensor(np.random.randn(1, 6, hidden_size), dtype=dtype) / hidden_size**0.5)
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
        hidden_states = hidden_states + position_embedding

        # 3.1.3 Adaptive Layer Norm
        modulation_parameters = self.scale_shift_table.to(hidden_states.dtype) + ops.reshape(
            modulation_parameters, (B, 6, -1)
        )
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mint.chunk(modulation_parameters, 6, dim=1)

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


class ModelParallelLlamaDecoderLayer(nn.Cell):
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
        fused_tensor_parallel: bool = True,
        group: str = GlobalComm.WORLD_COMM_GROUP,
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__()

        # 3.1.6 Context Parallelism
        self.self_attn = CONTEXT_PARALLEL_Llama_ATTENTION_CLASSES[attn_implementation](
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

        # 3.1.6 Tensor Parallelism
        if fused_tensor_parallel:
            self.mlp = FusedTensorParallelLlamaMLP(
                intermediate_size=intermediate_size,
                hidden_size=hidden_size,
                hidden_act=hidden_act,
                dim=1,
                group=group,
                dtype=dtype,
            )
        else:
            self.mlp = TensorParallelLlamaMLP(
                intermediate_size=intermediate_size,
                hidden_size=hidden_size,
                hidden_act=hidden_act,
                group=group,
                dtype=dtype,
            )

        self.scale_shift_table = Parameter(Tensor(np.random.randn(1, 6, hidden_size), dtype=dtype) / hidden_size**0.5)
        self.input_layernorm = LlamaRMSNorm(hidden_size, eps=rms_norm_eps, dtype=dtype)
        self.post_attention_layernorm = LlamaRMSNorm(hidden_size, eps=rms_norm_eps, dtype=dtype)

        if not fused_tensor_parallel:
            self.split_forward_gather_backward = SplitForwardGatherBackward(dim=1, grad_scale="down", group=group)
            self.gather_forward_split_backward = GatherForwardSplitBackward(dim=1, grad_scale="up", group=group)
        else:
            self.split_forward_gather_backward = nn.Identity()
            self.gather_forward_split_backward = nn.Identity()

    def construct(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        modulation_parameters: Tensor,
        position_embedding: Tensor,
    ) -> Tensor:
        B = hidden_states.shape[0]

        # 3.1.3 Positional Embedding
        hidden_states = hidden_states + position_embedding

        # 3.1.3 Adaptive Layer Norm
        modulation_parameters = self.scale_shift_table.to(hidden_states.dtype) + ops.reshape(
            modulation_parameters, (B, 6, -1)
        )
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mint.chunk(modulation_parameters, 6, dim=1)

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
        hidden_states = self.gather_forward_split_backward(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.split_forward_gather_backward(hidden_states)
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
        shift, scale = mint.chunk(
            ops.unsqueeze(self.scale_shift_table, 0) + ops.unsqueeze(timestep_embedding, 1), 2, dim=1
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
        model_parallelism: bool = False,
        fused_tensor_parallel: bool = True,
        post_init_weight: bool = True,
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.max_length = max_length
        self.model_parallelism = model_parallelism
        self._dtype = dtype
        mp_group = get_model_parallel_group()

        if self.model_parallelism:
            self.layers = nn.CellList(
                [
                    ModelParallelLlamaDecoderLayer(
                        hidden_size=self.hidden_size,
                        intermediate_size=intermediate_size,
                        num_attention_heads=self.num_attention_heads,
                        num_key_value_heads=self.num_key_value_heads,
                        rms_norm_eps=rms_norm_eps,
                        attention_dropout=attention_dropout,
                        attention_bias=attention_bias,
                        hidden_act=hidden_act,
                        attn_implementation=attn_implementation,
                        fused_tensor_parallel=fused_tensor_parallel,
                        group=mp_group,
                        dtype=dtype,
                    )
                    for _ in range(num_hidden_layers)
                ]
            )
        else:
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

        # TODO: drop this
        self.caption_embedder = CaptionEmbedder(caption_channels, self.hidden_size, eps=rms_norm_eps, dtype=dtype)

        if self.model_parallelism:
            self.group_size = get_group_size(mp_group)
            self.split_forward_gather_backward = SplitForwardGatherBackward(dim=1, grad_scale="down", group=mp_group)
            self.gather_forward_split_backward = GatherForwardSplitBackward(dim=1, grad_scale="up", group=mp_group)

        # post-init
        if post_init_weight:
            self.initializer_range = initializer_range
            self.init_weights()

        # recompute
        if gradient_checkpointing:
            self.layers.recompute()

    @property
    def dtype(self):
        return self._dtype

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

        # Zero-out adaLN modulation block:
        zeros_(self.adaLN_modulation[-1].weight)
        if self.adaLN_modulation[-1].bias is not None:
            zeros_(self.adaLN_modulation[-1].bias)

        # Zero-out final block as DiT does
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
        pos_embed = pos_embed_t + pos_embed_h + pos_embed_w
        pos_embed = ops.unsqueeze(pos_embed, 0)
        return pos_embed

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
        position_embedding = position_embedding.to(latent_embedding.dtype)

        # patchify and embed latent in transformer hidden dim.
        latent_embedding = self.latent_embedder(latent_embedding)

        # 6.1.2 shared timestep embedding & modulation. It does not mention the detail structure, we follow PixArt-Alpha here
        timestep_embedding = self.timestep_embedder(timestep)
        modulation_parameters = self.adaLN_modulation(timestep_embedding)

        # 3.1.4 text embedding
        text_embedding = self.caption_embedder(text_embedding)

        # main blocks
        hidden_states = latent_embedding

        # 3.1.6 Sequence Parallelism Start
        if self.model_parallelism:
            assert hidden_states.shape[1] % self.group_size == 0
            hidden_states = self.split_forward_gather_backward(hidden_states)
            position_embedding = self.split_forward_gather_backward(position_embedding)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, text_embedding, modulation_parameters, position_embedding)

        # 3.1.6 Sequence Parallelism End
        if self.model_parallelism:
            hidden_states = self.gather_forward_split_backward(hidden_states)

        # final block
        hidden_states = self.final_layer(hidden_states, timestep_embedding)

        # unpatchify
        output = self.unpatchify(hidden_states, t, h, w)
        return output

    def construct_with_cfg(
        self,
        latent_embedding: Tensor,
        timestep: Tensor,
        text_embedding: Tensor,
        cfg_scale: Union[Tensor, float] = 7.5,
    ) -> Tensor:
        """
        latent_embedding: (2N, T, C, H, W) tensor of inputs (latent representations of video)
        timestep: (2N,) tensor to indicate denoising step
        text_embedding: (2N, L, C') tensor of the text embedding
        cfg_scale: CFG scale
        """
        model_out = self(latent_embedding, timestep, text_embedding)
        cond_model_out, uncond_model_out = mint.chunk(model_out, 2, dim=0)
        model_out = uncond_model_out + cfg_scale * (cond_model_out - uncond_model_out)
        model_out = mint.tile(model_out, (2, 1, 1, 1, 1))
        return model_out

    def load_weight_from_non_parallel_cell(self, target: LlamaModel):
        param_dict = target.parameters_dict()

        # filter tensor-parallel block
        names = ["gate_proj", "up_proj", "down_proj"]
        param_dict = {k: v for k, v in param_dict.items() if not any([name in k for name in names])}
        load_param_into_net(self, param_dict)

        # load tensor-parallel block
        for layer, target_layer in zip(self.layers, target.layers):
            layer.mlp.load_weight_from_non_parallel_cell(target_layer.mlp)


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
