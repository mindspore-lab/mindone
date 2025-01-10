import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

from ..activation import ACT2FN
from ..common_layer import Embedding


class CLIPVisionEmbeddings(nn.Cell):
    def __init__(
        self,
        hidden_size: int = 1024,
        image_size: int = 336,
        patch_size: int = 14,
        num_channels: int = 3,
        dtype: ms.dtype = ms.float32,
    ) -> None:
        super().__init__()
        self.embed_dim = hidden_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels

        self.class_embedding = ms.Parameter(Tensor(ops.randn(self.embed_dim), dtype=dtype))

        self.patch_embedding = nn.Conv2d(
            in_channels=num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            has_bias=False,
            dtype=dtype,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = Embedding(self.num_positions, self.embed_dim, dtype=dtype)
        self.position_ids = ops.broadcast_to(ops.arange(self.num_positions), (1, -1))

    def construct(self, pixel_values: Tensor) -> Tensor:
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds = ops.flatten(patch_embeds, start_dim=2)
        patch_embeds = ops.transpose(patch_embeds, (0, 2, 1))

        class_embeds = ops.broadcast_to(self.class_embedding, (batch_size, 1, -1))
        embeddings = ops.concat([class_embeds, patch_embeds], axis=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class CLIPAttention(nn.Cell):
    def __init__(
        self,
        hidden_size: int = 1024,
        num_attention_heads: int = 16,
        attention_dropout: float = 0.0,
        dtype: ms.dtype = ms.float32,
    ) -> None:
        super().__init__()
        self.embed_dim = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = attention_dropout

        self.k_proj = nn.Dense(self.embed_dim, self.embed_dim, dtype=dtype)
        self.v_proj = nn.Dense(self.embed_dim, self.embed_dim, dtype=dtype)
        self.q_proj = nn.Dense(self.embed_dim, self.embed_dim, dtype=dtype)
        self.out_proj = nn.Dense(self.embed_dim, self.embed_dim, dtype=dtype)

    def _shape(self, tensor: Tensor, seq_len: int, bsz: int) -> Tensor:
        tensor = ops.reshape(tensor, (bsz, seq_len, self.num_heads, self.head_dim))
        tensor = ops.transpose(tensor, (0, 2, 1, 3))
        return tensor

    def construct(self, hidden_states: Tensor) -> Tensor:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.shape

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz)
        query_states = ops.reshape(query_states, proj_shape)
        key_states = ops.reshape(key_states, proj_shape)
        value_states = ops.reshape(value_states, proj_shape)

        attn_weights = ops.bmm(query_states, ops.transpose(key_states, (0, 2, 1)))
        attn_weights = ops.softmax(attn_weights, axis=-1)
        attn_probs = ops.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = ops.bmm(attn_probs, value_states)
        attn_output = ops.reshape(attn_output, (bsz, self.num_heads, tgt_len, self.head_dim))
        attn_output = ops.transpose(attn_output, (0, 2, 1, 3))
        attn_output = ops.reshape(attn_output, (bsz, tgt_len, embed_dim))

        attn_output = self.out_proj(attn_output)

        return attn_output


class CLIPMLP(nn.Cell):
    def __init__(
        self,
        hidden_size: int = 1024,
        intermediate_size: int = 4096,
        hidden_act: str = "quick_gelu",
        dtype: ms.dtype = ms.float32,
    ) -> None:
        super().__init__()
        self.activation_fn = ACT2FN[hidden_act]
        self.fc1 = nn.Dense(hidden_size, intermediate_size, dtype=dtype)
        self.fc2 = nn.Dense(intermediate_size, hidden_size, dtype=dtype)

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states
