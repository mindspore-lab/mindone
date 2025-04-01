import math

import mindspore as ms
from mindspore.common.api import jit_class
from mindspore.experimental.optim.lr_scheduler import LRScheduler


def linear_refined_lr(start_factor, end_factor, warmup_steps, *, lr, total_steps):
    lrs = []
    start_lr = lr * start_factor
    end_lr = lr * end_factor
    for i in range(total_steps):
        multiplier = min(i, warmup_steps) / warmup_steps
        lrs.append(start_lr + multiplier * (end_lr - start_lr))
    return lrs


def cosine_decay_refined_lr(decay_steps, eta_min, *, eta_max, total_steps, num_cycles=1, cycle_decay=1.0):
    lrs = []

    for c in range(int(num_cycles)):
        lr_max = eta_max * (cycle_decay**c)
        delta = 0.5 * (lr_max - eta_min)
        for i in range(decay_steps):
            t_cur = min(i, decay_steps)
            lr_cur = eta_min + delta * (1.0 + math.cos(math.pi * t_cur / decay_steps))
            if len(lrs) < total_steps:
                lrs.append(lr_cur)
            else:
                break

    if total_steps > num_cycles * decay_steps:
        for i in range(total_steps - (num_cycles * decay_steps)):
            lrs.append(eta_min)

    return lrs


@jit_class
class WarmupCosineDecayLR(LRScheduler):
    def __init__(self, optimizer, lr_max, lr_min, decay_steps, warmup_steps=0, last_epoch=-1):
        warmup_lrs = []
        if warmup_steps > 0:
            warmup_lrs = linear_refined_lr(
                start_factor=0.0,
                end_factor=1.0,
                warmup_steps=warmup_steps,
                lr=lr_max,
                total_steps=warmup_steps,
            )
        main_lrs = cosine_decay_refined_lr(
            decay_steps=decay_steps,
            eta_min=lr_min,
            eta_max=lr_max,
            total_steps=decay_steps,
            num_cycles=1,
            cycle_decay=1.0,
        )

        self.lr_seq = warmup_lrs + main_lrs
        self.lr_seq = ms.Tensor(self.lr_seq, dtype=ms.float32)

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # self.last_epoch is the current step, begin from 0 if self.last_epoch is not resumed
        cur_step = self.last_epoch.value().to(ms.int32)
        # lr for each parameter group on current step
        if cur_step <= self.lr_seq.shape[0] - 1:
            group_lrs = [self.lr_seq[cur_step] for lr in self._last_lr]
        else:
            # TODO: support cycle
            group_lrs = [self.lr_seq[-1] for lr in self._last_lr]

        return group_lrs
