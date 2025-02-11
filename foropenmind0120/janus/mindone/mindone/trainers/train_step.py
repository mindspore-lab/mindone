"""Train step wrapper supporting setting drop overflow update, ema etc"""
from typing import Optional

from packaging import version

import mindspore as ms
import mindspore.context as context
from mindspore import Tensor, nn, ops
from mindspore.boost.grad_accumulation import gradient_accumulation_op as _grad_accum_op
from mindspore.boost.grad_accumulation import gradient_clear_op as _grad_clear_op
from mindspore.common import RowTensor
from mindspore.common import dtype as mstype
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P

from .ema import EMA

_grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()
_grad_overflow = C.MultitypeFuncGraph("_grad_overflow")


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * F.cast(reciprocal(scale), F.dtype(grad))


@_grad_scale.register("Tensor", "RowTensor")
def tensor_grad_scale_row_tensor(scale, grad):
    return RowTensor(
        grad.indices,
        grad.values * F.cast(reciprocal(scale), F.dtype(grad.values)),
        grad.dense_shape,
    )


class TrainOneStepWrapper(nn.TrainOneStepWithLossScaleCell):
    """TrainStep with ema and clip grad.

    Args:
        drop_overflow_update: if True, network will not be updated when gradient is overflow.
        scale_sense (Union[Tensor, Cell]): If this value is a Cell, it will be called
            to update loss scale. If this value is a Tensor, the loss scale can be modified by `set_sense_scale`,
            the shape should be :math:`()` or :math:`(1,)`.
        zero_helper (class): Zero redundancy optimizer(ZeRO) build helper, default is None.

    Returns:
        Tuple of 3 Tensor, the loss, overflow flag and current loss scale value.
        loss (Tensor) -  A scalar, the loss value.
        overflow (Tensor) -  A scalar, whether overflow occur or not, the type is bool.
        loss scale (Tensor) -  The loss scale value, the shape is :math:`()` or :math:`(1,)`.

    """

    def __init__(
        self,
        network,
        optimizer,
        scale_sense=1.0,
        ema: Optional[EMA] = None,
        updates=0,
        drop_overflow_update=True,
        gradient_accumulation_steps=1,
        clip_grad=False,
        clip_norm=1.0,
        verbose=False,
        zero_helper=None,
    ):
        super().__init__(network, optimizer, scale_sense)
        self.ema = ema
        self.drop_overflow_update = drop_overflow_update

        assert isinstance(clip_grad, bool), f"Invalid type of clip_grad, got {type(clip_grad)}, expected bool"
        assert clip_norm > 0.0 and isinstance(clip_norm, float), f"clip_norm must be float > 1.0, but got {clip_norm}"
        self.clip_grad = clip_grad
        self.clip_norm = clip_norm

        assert gradient_accumulation_steps >= 1
        self.accum_steps = gradient_accumulation_steps
        if gradient_accumulation_steps > 1:
            self.accumulated_grads = optimizer.parameters.clone(prefix="grad_accumulated_", init="zeros")

            self.cur_accum_step = ms.Parameter(ms.Tensor(0, dtype=ms.int32), name="accum_step")
            self.zero = Tensor(0, ms.int32)

        self.verbose = verbose
        self.is_cpu_device = context.get_context("device_target") == "CPU"  # to support CPU in CI
        self.skip_start_overflow_check = version.parse(ms.__version__) >= version.parse("2.1")

        self.map = ops.Map()
        self.partial = ops.Partial()

        # zero init
        self.zero_helper = zero_helper
        self.zero_stage = zero_helper.zero_stage if zero_helper is not None else 0
        self.run_optimizer = zero_helper.run_optimizer if zero_helper is not None else self.optimizer
        self.grad_reducer = self.grad_reducer if self.zero_stage == 0 else nn.Identity()
        if self.zero_stage != 0:
            self.zero_helper.split_params()
            if gradient_accumulation_steps > 1:
                self.accumulated_grads = optimizer.parameters.clone(prefix="grad_accumulated_", init="zeros")

    def set_train(self, mode: bool = True):
        # Delegate the setting of training mode behavior to the network.
        self.network.set_train(mode)

    def construct(self, *inputs):
        # compute loss
        weights = self.weights
        loss = self.network(*inputs)  # mini-batch loss
        scaling_sens = self.scale_sense

        # check loss overflow. (after ms2.1, it's done together with gradient overflow checking)
        if self.skip_start_overflow_check:
            status = Tensor([0] * 8, mstype.int32)
        else:
            if not self.is_cpu_device:
                status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
            else:
                status = None

        scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))  # loss scale value

        # 1. compute gradients (of the up-scaled loss w.r.t. the model weights)
        grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)

        # Gradient communication
        if self.zero_helper is not None:
            grads = self.zero_helper.cal_gradients(grads)

        if self.accum_steps == 1:
            grads = self.grad_reducer(grads)
            scaling_sens = ops.depend(scaling_sens, grads)

        # 2. down-scale gradients by loss_scale. grads = grads / scaling_sense  / grad_accum_steps
        # also divide gradients by accumulation steps to avoid taking mean of  the accumulated gradients later
        grads = self.hyper_map(F.partial(_grad_scale, scaling_sens), grads)  # accum_steps division is done later

        # 3. check gradient overflow
        if not self.is_cpu_device:
            cond = self.get_overflow_status(status, grads)
            overflow = self.process_loss_scale(cond)
        else:
            overflow = ms.Tensor(False)
            cond = ms.Tensor(False)

        # accumulate gradients and update model weights if no overflow or allow to update even when overflow
        if (not self.drop_overflow_update) or (not overflow):
            # 4. gradient accumulation if enabled
            if self.accum_steps > 1:
                # self.accumulated_grads += grads / accum_steps
                loss = F.depend(
                    loss, self.hyper_map(F.partial(_grad_accum_op, self.accum_steps), self.accumulated_grads, grads)
                )

                # self.cur_accum_step += 1
                loss = F.depend(loss, ops.assign_add(self.cur_accum_step, Tensor(1, ms.int32)))

                if self.cur_accum_step >= self.accum_steps:
                    # 5. gradient reduction on distributed GPUs/NPUs
                    grads = self.grad_reducer(self.accumulated_grads)

                    # 6. clip grad
                    if self.clip_grad:
                        grads = ops.clip_by_global_norm(grads, self.clip_norm)
                    # 7. optimize
                    loss = F.depend(loss, self.run_optimizer(grads))

                    # clear gradient accumulation states
                    loss = F.depend(loss, self.hyper_map(F.partial(_grad_clear_op), self.accumulated_grads))
                    # self.cur_accum_step = 0
                    loss = F.depend(loss, ops.assign(self.cur_accum_step, self.zero))
                else:
                    # update LR in each gradient step but not optimize net parameter
                    # to ensure the LR curve is consistent
                    # FIXME: for ms>=2.2, get_lr() will not increase global step by 1. we need to do it manually.
                    loss = F.depend(loss, self.optimizer.get_lr())
            else:
                # 5. gradient reduction on distributed GPUs/NPUs
                # 6. clip grad
                if self.clip_grad:
                    grads = ops.clip_by_global_norm(grads, self.clip_norm)
                # 7. optimize
                loss = F.depend(loss, self.run_optimizer(grads))

            # 8.ema
            if self.ema is not None:
                self.ema.ema_update()
        # else:
        #    print("WARNING: Gradient overflow! update skipped.") # TODO: recover it after 910B in-graph print issue fixed

        return loss, cond, scaling_sens
