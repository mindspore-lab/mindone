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
        self.ul2_projector = nn.SequentialCell(
            [
                mint.nn.Linear(ul2_in_features, out_features, bias=False, dtype=dtype),
                layer_norm((out_features,), eps=norm_eps, dtype=dtype),
            ]
        )
        self.metaclip_projector = nn.SequentialCell(
            [
                mint.nn.Linear(metaclip_in_features, out_features, bias=False, dtype=dtype),
                layer_norm((out_features,), eps=norm_eps, dtype=dtype),
            ]
        )
        self.byt5_projector = nn.SequentialCell(
            [
                mint.nn.Linear(byt5_in_features, out_features, bias=False, dtype=dtype),
                layer_norm((out_features,), eps=norm_eps, dtype=dtype),
            ]
        )

    def construct(self, ul2_text: Tensor, metaclip_text: Tensor, byt5_text: Tensor) -> Tensor:
        ul2_hidden_states = self.ul2_projector(ul2_text)
        metaclip_hidden_states = self.metaclip_projector(metaclip_text)
        byt5_hidden_states = self.byt5_projector(byt5_text)

        return mint.cat((ul2_hidden_states, metaclip_hidden_states, byt5_hidden_states), dim=1)
