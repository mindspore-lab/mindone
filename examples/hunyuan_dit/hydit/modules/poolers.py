import mindspore as ms
from mindspore import nn, ops

from .transformers import multi_head_attention_forward


class AttentionPool(nn.Cell):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = ms.Parameter(
            ops.randn(spacial_dim + 1, embed_dim) / embed_dim**0.5, name="positional_embedding"
        )
        self.k_proj = nn.Dense(embed_dim, embed_dim)
        self.q_proj = nn.Dense(embed_dim, embed_dim)
        self.v_proj = nn.Dense(embed_dim, embed_dim)
        self.c_proj = nn.Dense(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def construct(self, x):
        x = x.permute(1, 0, 2)  # NLC -> LNC
        x = ops.cat([x.mean(axis=0, keep_dims=True), x], axis=0)  # (L+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (L+1)NC
        x, _ = multi_head_attention_forward(
            query=x[:1],
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            # TODO: this is for lora, maybe there is a better method
            in_proj_bias=ops.cat([self.q_proj.base_layer.bias, self.k_proj.bias, self.v_proj.bias])
            if hasattr(self.q_proj, "base_layer")
            else ops.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )
        return x.squeeze(0)
