import mindspore as ms
from mindspore import Parameter, Tensor, nn, ops
from mindspore.ops import composite as C
from mindspore.ops import functional as F

__all__ = ["EMA"]

_ema_op = C.MultitypeFuncGraph("grad_ema_op")


@_ema_op.register("Tensor", "Tensor", "Tensor")
def _ema_weights(factor, ema_weight, weight):
    return F.assign(ema_weight, ema_weight * factor + weight * (1 - factor))


class EMA(nn.Cell):
    """
    Args:
        updates: number of ema updates, which can be restored from resumed training.
        offloading: if True, offload the assign computation to CPU to avoid OOM issue.
    """

    def __init__(
        self,
        network: nn.Cell,
        ema_decay: float = 0.9999,
        updates: int = 0,
        trainable_only: bool = True,
        offloading: bool = True,
    ):
        super().__init__()
        # TODO: net.trainable_params() is more reasonable?
        if trainable_only:
            self.net_weight = ms.ParameterTuple(network.trainable_params())
        else:
            self.net_weight = ms.ParameterTuple(network.get_parameters())
        self.ema_weight = self.net_weight.clone(prefix="ema")
        self.swap_cache = self.net_weight.clone(prefix="swap", init="zeros")

        self.ema_decay = ema_decay
        self.updates = Parameter(Tensor(updates, ms.float32), requires_grad=False)

        self.hyper_map = C.HyperMap()
        self.map = ops.HyperMap()
        self.offloading = offloading
        if not offloading:
            self.assign = ops.Assign()

    def ema_update(self):
        """Update EMA parameters."""
        self.updates += 1
        d = self.ema_decay * (1 - F.exp(-self.updates / 2000))
        # update trainable parameters
        success = self.hyper_map(F.partial(_ema_op, d), self.ema_weight, self.net_weight)
        self.updates = F.depend(self.updates, success)
        return self.updates

    def swap_data(self, ori_datas, tgt_datas):
        for ori_data, tgt_data in zip(ori_datas, tgt_datas):
            tgt_data.set_data(ori_data)

    def swap_before_eval(self):
        if self.offloading:
            self.swap_data(self.net_weight, self.swap_cache)
            self.swap_data(self.ema_weight, self.net_weight)
            return True
        # net -> swap
        success = self.map(self.assign, self.swap_cache, self.net_weight)
        # ema -> net
        success = F.depend(success, self.map(self.assign, self.net_weight, self.ema_weight))
        return success

    def swap_after_eval(self):
        # swap -> net
        if self.offloading:
            self.swap_data(self.swap_cache, self.net_weight)
            return True
        success = self.map(self.assign, self.net_weight, self.swap_cache)
        return success
