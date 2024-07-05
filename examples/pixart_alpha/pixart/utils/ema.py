import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

__all__ = ["EMA"]

ema_update = ops.MultitypeFuncGraph("ema_op")


@ema_update.register("Tensor", "Tensor", "Tensor")
def update_weights(factor, ema_weight, weight):
    return ops.assign(ema_weight, ema_weight * factor + weight * (1 - factor))


class EMA(nn.Cell):
    def __init__(self, network: nn.Cell, ema_decay: float = 0.9999) -> None:
        super().__init__()
        self.net_weight = ms.ParameterTuple(network.get_parameters())
        self.ema_weight = self.net_weight.clone(prefix="ema")
        self.ema_decay = Tensor(ema_decay, dtype=ms.float32)
        self.hyper_map = ops.HyperMap()

    def ema_update(self) -> None:
        self.hyper_map(ema_update, self.ema_decay, self.ema_weight, self.net_weight)
