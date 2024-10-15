from typing import Literal, Tuple

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, load_checkpoint
from mindspore.common.initializer import Normal, Zero, initializer

from .layer import LlamaAttention, LlamaFlashAttention, LlamaMLP, LlamaRMSNorm, PatchEmbed3D

Llama_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attention": LlamaFlashAttention,
}


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
        self.hidden_size = hidden_size

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
        self.input_layernorm = LlamaRMSNorm(hidden_size, eps=rms_norm_eps, dtype=dtype)
        self.post_attention_layernorm = LlamaRMSNorm(hidden_size, eps=rms_norm_eps, dtype=dtype)

    def construct(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        position_embeddings: Tensor,
    ) -> Tensor:
        # 3.1.3 Add Positional Embedding
        hidden_states = hidden_states + position_embeddings.to(hidden_states.dtype)

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention (Bi-Directional Attention)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        # 3.1.3 Cross Attention
        hidden_states = self.cross_attn(hidden_states, encoder_hidden_states=encoder_hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class LlamaModel(nn.Cell):
    def __init__(
        self,
        in_channels: int = 8,
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
        kernel_size: Tuple[int, int, int] = (1, 2, 2),
        max_length: Tuple[int, int, int] = (16, 24, 44),
        attn_implementation: Literal["eager", "flash_attention"] = "eager",
        gradient_checkpointing: bool = False,
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.layers = nn.CellList(
            [
                LlamaDecoderLayer(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
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
        self.norm = LlamaRMSNorm(hidden_size, eps=rms_norm_eps, dtype=dtype)

        self.pos_embedding_table_h = nn.Embedding(max_length[0], hidden_size, dtype=dtype)
        self.pos_embedding_table_w = nn.Embedding(max_length[1], hidden_size, dtype=dtype)
        self.pos_embedding_table_t = nn.Embedding(max_length[2], hidden_size, dtype=dtype)

        self.latent_embedder = PatchEmbed3D(kernel_size, in_channels, hidden_size, dtype=dtype)

        # post-init
        self.initializer_range = initializer_range
        self.init_weights()

        # recompute
        if gradient_checkpointing:
            self.layers.recompute()

    def init_weights(self):
        def _init_weights(module):
            std = self.initializer_range
            if isinstance(module, nn.Dense):
                module.weight.set_data(initializer(Normal(std, 0.0), module.weight.shape, module.weight.dtype))
                if module.bias is not None:
                    module.bias.set_data(initializer(Zero(), module.bias.shape, module.bias.dtype))
            elif isinstance(module, nn.Embedding):
                module.embedding_table.set_data(
                    initializer(Normal(std, 0.0), module.embedding_table.shape, module.embedding_table.dtype)
                )

        self.apply(_init_weights)

    def learnable_position_embedding(self, latent_embedding: Tensor) -> Tensor:
        # 3.1.3
        _, t, _, h, w = latent_embedding.shape
        t_inds = ops.arange(t // self.kernel_size[0], dtype=ms.int64)
        h_inds = ops.arange(h // self.kernel_size[1], dtype=ms.int64)
        w_inds = ops.arange(w // self.kernel_size[2], dtype=ms.int64)

        position_ids = ops.meshgrid(t_inds, h_inds, w_inds, indexing="ij")
        position_ids = ops.stack(position_ids, axis=-1)
        position_ids = ops.reshape(position_ids, (-1, 3))

        h_inds, w_inds, t_inds = ops.unbind(position_ids, dim=-1)
        pos_embed_h = self.pos_embedding_table_h(h_inds)
        pos_embed_w = self.pos_embedding_table_w(w_inds)
        pos_embed_t = self.pos_embedding_table_t(t_inds)
        return pos_embed_h + pos_embed_w + pos_embed_t

    def construct(
        self,
        latent_embedding: Tensor,
        text_embedding: Tensor,
    ) -> Tensor:
        """
        latent_embedding: (N, T, C, H, W) tensor of inputs (latent representations of video)
        text_embedding: (N, L, C') tensor of the text embedding
        """
        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.learnable_position_embedding(latent_embedding)

        # patchify
        inputs_embeds = self.latent_embedder(latent_embedding)

        hidden_states = inputs_embeds

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, text_embedding, position_embeddings)

        hidden_states = self.norm(hidden_states)
        return hidden_states


def llama3_8B(from_pretrained=None, **kwargs):
    model = LlamaModel(
        attention_bias=False,
        attention_dropout=0.0,
        hidden_act="silu",
        hidden_size=4096,
        initializer_range=0.02,
        intermediate_size=14336,
        num_attention_heads=32,
        num_hidden_layers=32,
        num_key_value_heads=32,
        rms_norm_eps=1e-05,
        **kwargs,
    )
    if from_pretrained is not None:
        load_checkpoint(from_pretrained, model)
    return model
