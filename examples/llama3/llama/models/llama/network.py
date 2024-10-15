from typing import Literal, Optional

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.common.initializer import Normal, Zero, initializer

from .layer import LlamaAttention, LlamaFlashAttention, LlamaMLP, LlamaRMSNorm, LlamaRotaryEmbedding

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
        position_embeddings: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states,
            position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class LlamaModel(nn.Cell):
    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 32,
        num_hidden_layers: int = 32,
        num_key_value_heads: int = 8,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 1000000.0,
        attention_dropout: float = 0.0,
        hidden_act: str = "silu",
        initializer_range: float = 0.02,
        attn_implementation: Literal["eager", "flash_attention"] = "eager",
        gradient_checkpointing: bool = False,
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__()
        self.layers = nn.CellList(
            [
                LlamaDecoderLayer(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    rms_norm_eps=rms_norm_eps,
                    attention_dropout=attention_dropout,
                    hidden_act=hidden_act,
                    attn_implementation=attn_implementation,
                    dtype=dtype,
                )
                for _ in range(num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(hidden_size, eps=rms_norm_eps, dtype=dtype)
        self.rotary_emb = LlamaRotaryEmbedding(
            hidden_size // num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )

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

        self.apply(_init_weights)

    def construct(
        self,
        inputs_embeds: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        position_ids = ops.arange(0, inputs_embeds.shape[1], dtype=ms.int64)
        position_ids = ops.unsqueeze(position_ids, 0)

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings,
                attention_mask=attention_mask,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states
