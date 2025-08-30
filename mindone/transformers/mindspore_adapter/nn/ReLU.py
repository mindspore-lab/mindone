import mindspore as ms
from mindspore import mint, nn


class ReLU(nn.Cell):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def construct(self, input: ms.Tensor) -> ms.Tensor:
        if self.inplace:
            return mint.nn.functional.relu_(input)
        else:
            return mint.nn.functional.relu(input)
