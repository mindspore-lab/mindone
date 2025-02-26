from typing import Union

from mindspore.ops import composite as C
from mindspore.ops import functional as F

from mindone.trainers.ema import EMA as EMA_

_ema_op = C.MultitypeFuncGraph("grad_ema_op")


@_ema_op.register("Number", "Tensor", "Tensor")
def _ema_weights(factor, ema_weight, weight):
    return F.assign(ema_weight, ema_weight * factor + weight * (1 - factor))


class EMA(EMA_):
    def __init__(
        self,
        network,
        ema_decay=0.9999,
        updates=0,
        trainable_only=True,
        offloading=True,
        min_decay: float = 0.0,
        update_after_step: int = 0,
        use_ema_warmup: bool = False,
        inv_gamma: Union[float, int] = 1.0,
        power: Union[float, int] = 2 / 3,
    ):
        """
        Args:
            network (Iterable[ms.Parameter]): The parameters to track.
            ema_decay (float): The decay factor for the exponential moving average.
            updates (int): the current optimization steps.
            trainable_only (bool): whether to apply ema for trainable parameters only.
            offloading (bool): whether to offload ema ops to cpu.
            min_decay (float): The minimum decay factor for the exponential moving average.
            update_after_step (int): The number of steps to wait before starting to update the EMA weights.
            use_ema_warmup (bool): Whether to use EMA warmup.
            inv_gamma (float):
                Inverse multiplicative factor of EMA warmup. Default: 1. Only used if `use_ema_warmup` is True.
            power (float): Exponential factor of EMA warmup. Default: 2/3. Only used if `use_ema_warmup` is True.
            device (Optional[Union[str, torch.device]]): The device to store the EMA weights on. If None, the EMA
                        weights will be stored on CPU.
        @crowsonkb's notes on EMA Warmup:
            If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are good values for models you plan
            to train for a million or more steps (reaches decay factor 0.999 at 31.6K steps, 0.9999 at 1M steps),
            gamma=1, power=3/4 for models you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999
            at 215.4k steps).
        """
        super().__init__(network, ema_decay, updates, trainable_only, offloading)
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.use_ema_warmup = use_ema_warmup
        self.inv_gamma = inv_gamma
        self.power = power

    def get_decay(self, updates: int) -> float:
        """
        Compute the decay factor for the exponential moving average.
        """
        step = int(updates.value()) - self.update_after_step - 1
        step = step if step > 0 else 0

        if step <= 0:
            return 0.0

        if self.use_ema_warmup:
            cur_decay_value = 1 - (1 + step / self.inv_gamma) ** -self.power
        else:
            cur_decay_value = (1 + step) / (10 + step)

        cur_decay_value = cur_decay_value if (cur_decay_value < self.ema_decay) else self.ema_decay
        # make sure decay is not smaller than min_decay
        cur_decay_value = cur_decay_value if (cur_decay_value > self.min_decay) else self.min_decay
        return cur_decay_value

    def ema_update(self):
        """Update EMA parameters."""
        self.updates += 1
        # Compute the decay factor for the exponential moving average.
        decay = self.get_decay(self.updates)
        # update trainable parameters
        success = self.hyper_map(F.partial(_ema_op, decay), self.ema_weight, self.net_weight)
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
