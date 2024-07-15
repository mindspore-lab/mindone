import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, Tensor

__all__ = ["EMA"]

_ema_update = ops.MultitypeFuncGraph("_ema_update")


@_ema_update.register("Number", "Tensor", "Tensor")
def update_weights(factor: float, ema_weight: Parameter, weight: Tensor) -> None:
    return ops.assign(ema_weight, ema_weight * factor + weight * (1 - factor))


class EMA(nn.Cell):
    def __init__(self, network: nn.Cell, ema_decay: float = 0.9999) -> None:
        super().__init__()
        self.net_weight = ms.ParameterTuple(network.get_parameters())
        self.ema_weight = self.net_weight.clone(prefix="ema")
        self.hyper_map = ops.HyperMap()
        self.ema_decay = ema_decay

    @ms.jit
    def ema_update(self) -> None:
        self.hyper_map(ops.partial(_ema_update, self.ema_decay), self.ema_weight, self.net_weight)
