from mindspore import Tensor, mint, nn, ops


class TextProjector(nn.Cell):
    def __init__(
        self,
        ul2_in_features: int = 4096,
        metaclip_in_features: int = 1280,
        byt5_in_features: int = 1472,
        out_features: int = 6144,
    ):
        super().__init__()
        self.ul2_projector = nn.SequentialCell(
            [mint.nn.Linear(ul2_in_features, out_features, bias=False), mint.nn.LayerNorm((out_features,))]
        )
        self.metaclip_projector = nn.SequentialCell(
            [mint.nn.Linear(metaclip_in_features, out_features, bias=False), mint.nn.LayerNorm((out_features,))]
        )
        self.byt5_projector = nn.SequentialCell(
            [mint.nn.Linear(byt5_in_features, out_features, bias=False), mint.nn.LayerNorm((out_features,))]
        )

    def construct(self, ul2_text: Tensor, metaclip_text: Tensor, byt5_text: Tensor) -> Tensor:
        ul2_hidden_states = self.ul2_projector(ul2_text)
        metaclip_hidden_states = self.metaclip_projector(metaclip_text)
        byt5_hidden_states = self.byt5_projector(byt5_text)

        return ops.concat((ul2_hidden_states, metaclip_hidden_states, byt5_hidden_states), axis=1)
