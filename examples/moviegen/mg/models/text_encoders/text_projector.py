from typing import Type

import mindspore as ms
from mindspore import Tensor, mint, nn


class TextProjector(nn.Cell):
    def __init__(
        self,
        ul2_in_features: int = 4096,
        metaclip_in_features: int = 1280,
        byt5_in_features: int = 1472,
        out_features: int = 6144,
        layer_norm: Type[nn.Cell] = mint.nn.LayerNorm,
        norm_eps: float = 1e-5,
        dtype: ms.Type = ms.float32,
    ):
        super().__init__()
        # split layers for easier exclusion from weight decay
        self.ul2_linear = nn.Dense(ul2_in_features, out_features, has_bias=False, dtype=dtype)
        self.ul2_layernorm = layer_norm((out_features,), eps=norm_eps, dtype=dtype)

        self.metaclip_linear = nn.Dense(metaclip_in_features, out_features, has_bias=False, dtype=dtype)
        self.metaclip_layernorm = layer_norm((out_features,), eps=norm_eps, dtype=dtype)

        self.byt5_linear = nn.Dense(byt5_in_features, out_features, has_bias=False, dtype=dtype)
        self.byt5_layernorm = layer_norm((out_features,), eps=norm_eps, dtype=dtype)

    def construct(self, ul2_emb: Tensor, metaclip_emb: Tensor, byt5_emb: Tensor) -> Tensor:
        ul2_hidden_states = self.ul2_layernorm(self.ul2_linear(ul2_emb))
        metaclip_hidden_states = self.metaclip_layernorm(self.metaclip_linear(metaclip_emb))
        byt5_hidden_states = self.byt5_layernorm(self.byt5_linear(byt5_emb))

        return mint.cat((ul2_hidden_states, metaclip_hidden_states, byt5_hidden_states), dim=1)
