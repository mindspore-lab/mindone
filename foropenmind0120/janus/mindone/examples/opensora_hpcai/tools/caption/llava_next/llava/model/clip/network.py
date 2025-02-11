from typing import Tuple

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor

from ..common_layer import LayerNorm
from .layer import CLIPMLP, CLIPAttention, CLIPVisionEmbeddings


class CLIPEncoderLayer(nn.Cell):
    def __init__(
        self,
        hidden_size: int = 1024,
        intermediate_size: int = 4096,
        num_attention_heads: int = 16,
        attention_dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
        hidden_act: str = "quick_gelu",
        dtype: ms.dtype = ms.float32,
    ) -> None:
        super().__init__()
        self.embed_dim = hidden_size
        self.self_attn = CLIPAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            dtype=dtype,
        )
        self.layer_norm1 = LayerNorm((self.embed_dim,), epsilon=layer_norm_eps, dtype=dtype)
        self.mlp = CLIPMLP(
            hidden_size=hidden_size, intermediate_size=intermediate_size, hidden_act=hidden_act, dtype=dtype
        )
        self.layer_norm2 = LayerNorm((self.embed_dim,), epsilon=layer_norm_eps, dtype=dtype)

    def construct(self, hidden_states: Tensor) -> Tensor:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class CLIPEncoder(nn.Cell):
    def __init__(
        self,
        num_hidden_layers: int = 24,
        hidden_size: int = 1024,
        intermediate_size: int = 4096,
        num_attention_heads: int = 16,
        attention_dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
        hidden_act: str = "quick_gelu",
        dtype: ms.dtype = ms.float32,
    ) -> None:
        super().__init__()
        self.layers = nn.CellList(
            [
                CLIPEncoderLayer(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_attention_heads=num_attention_heads,
                    attention_dropout=attention_dropout,
                    layer_norm_eps=layer_norm_eps,
                    hidden_act=hidden_act,
                    dtype=dtype,
                )
                for _ in range(num_hidden_layers)
            ]
        )

    def construct(self, inputs_embeds: Tensor) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        encoder_states = ()

        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            encoder_states = encoder_states + (hidden_states,)
            hidden_states = encoder_layer(hidden_states)

        encoder_states = encoder_states + (hidden_states,)
        return hidden_states, encoder_states


class CLIPVisionTransformer(nn.Cell):
    def __init__(
        self,
        image_size: int = 336,
        patch_size: int = 14,
        num_channels: int = 3,
        num_hidden_layers: int = 24,
        hidden_size: int = 1024,
        intermediate_size: int = 4096,
        num_attention_heads: int = 16,
        attention_dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
        hidden_act: str = "quick_gelu",
        dtype: ms.dtype = ms.float32,
    ) -> None:
        super().__init__()
        embed_dim = hidden_size

        self.embeddings = CLIPVisionEmbeddings(
            hidden_size=hidden_size,
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            dtype=dtype,
        )
        self.pre_layrnorm = LayerNorm((embed_dim,), epsilon=layer_norm_eps, dtype=dtype)
        self.encoder = CLIPEncoder(
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            layer_norm_eps=layer_norm_eps,
            hidden_act=hidden_act,
            dtype=dtype,
        )
        self.post_layernorm = LayerNorm((embed_dim,), epsilon=layer_norm_eps, dtype=dtype)

    def construct(self, pixel_values: Tensor) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)

        last_hidden_state, encoder_states = self.encoder(
            inputs_embeds=hidden_states,
        )

        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        return pooled_output, encoder_states


class CLIPVisionModel(nn.Cell):
    def __init__(
        self,
        image_size: int = 336,
        patch_size: int = 14,
        num_channels: int = 3,
        num_hidden_layers: int = 24,
        hidden_size: int = 1024,
        intermediate_size: int = 4096,
        num_attention_heads: int = 16,
        attention_dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
        hidden_act: str = "quick_gelu",
        dtype: ms.dtype = ms.float32,
        **kwargs,
    ) -> None:
        super().__init__()

        self.vision_model = CLIPVisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            layer_norm_eps=layer_norm_eps,
            hidden_act=hidden_act,
            dtype=dtype,
        )

    def get_input_embeddings(self) -> nn.Cell:
        return self.vision_model.embeddings.patch_embedding

    @ms.jit
    def construct(self, pixel_values: Tensor) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        pooled_output, hidden_states = self.vision_model(pixel_values)

        return pooled_output, hidden_states


class CLIPVisionModelWithProjection(nn.Cell):
    def __init__(
        self,
        projection_dim: int = 768,
        image_size: int = 336,
        patch_size: int = 14,
        num_channels: int = 3,
        num_hidden_layers: int = 24,
        hidden_size: int = 1024,
        intermediate_size: int = 4096,
        num_attention_heads: int = 16,
        attention_dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
        hidden_act: str = "quick_gelu",
        dtype: ms.dtype = ms.float32,
        **kwargs,
    ) -> None:
        super().__init__()

        self.vision_model = CLIPVisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            layer_norm_eps=layer_norm_eps,
            hidden_act=hidden_act,
            dtype=dtype,
        )

        self.visual_projection = nn.Dense(hidden_size, projection_dim, has_bias=False, dtype=dtype)

    def get_input_embeddings(self) -> nn.Cell:
        return self.vision_model.embeddings.patch_embedding

    @ms.jit
    def construct(self, pixel_values: Tensor) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        pooled_output, hidden_states = self.vision_model(pixel_values)
        image_embeds = self.visual_projection(pooled_output)

        return image_embeds, hidden_states
