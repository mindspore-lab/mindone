from mindspore.ops import composite as C
from mindspore.ops import functional as F

from mindone.trainers.ema import EMA as EMA_

_ema_op = C.MultitypeFuncGraph("grad_ema_op")


@_ema_op.register("Number", "Tensor", "Tensor")
def _ema_weights(factor, ema_weight, weight):
    return F.assign(ema_weight, ema_weight * factor + weight * (1 - factor))


class EMA(EMA_):
    def ema_update(self):
        """Update EMA parameters."""
        self.updates += 1
        # update trainable parameters
        success = self.hyper_map(F.partial(_ema_op, self.ema_decay), self.ema_weight, self.net_weight)
        self.updates = F.depend(self.updates, success)

        return self.updates


def save_ema_ckpts(net, ema, ckpt_manager, ckpt_name):
    if ema is not None:
        ema.swap_before_eval()

    ckpt_manager.save(net, None, ckpt_name=ckpt_name, append_dict=None)

    if ema is not None:
        ema.swap_after_eval()
        ckpt_manager.save(
            net,
            None,
            ckpt_name=ckpt_name.replace(".ckpt", "_nonema.ckpt"),
            append_dict=None,
        )
