from collections import OrderedDict

from mindspore import Tensor, nn, ops


class QuickGELU(nn.Cell):
    def construct(self, x: Tensor) -> Tensor:
        return x * ops.sigmoid(1.702 * x)


class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)


ACT2CLS = {
    "gelu": (nn.GELU, {"approximate": False}),
    "quick_gelu": QuickGELU,
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "sigmoid": nn.Sigmoid,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}
ACT2FN = ClassInstantier(ACT2CLS)
