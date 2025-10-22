import mindspore as ms
from mindspore import mint, nn


class LeakyReLU(nn.Cell):
    def __init__(self, negative_slope: float = 1e-2, inplace: bool = False) -> None:
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def construct(self, input: ms.Tensor) -> ms.Tensor:
        if self.inplace:
            input.copy_(mint.nn.functional.leaky_relu(input, self.negative_slope))
            return input
        else:
            return mint.nn.functional.leaky_relu(input, self.negative_slope)
