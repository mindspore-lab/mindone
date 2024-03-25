"""Train step wrapper supporting setting drop overflow update, ema etc"""
from packaging import version

import mindspore as ms
import mindspore.context as context
from mindspore import Parameter, Tensor, nn, ops
from mindspore.common import RowTensor
from mindspore.common import dtype as mstype
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P

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
        ema=None,
        updates=0,
        drop_overflow_update=True,
        gradient_accumulation_steps=1,
        clip_grad=False,
        clip_norm=1.0,
        verbose=False,
    ):
        super().__init__(network, optimizer, scale_sense)
        self.ema = ema
        self.drop_overflow_update = drop_overflow_update

        assert isinstance(clip_grad, bool), f"Invalid type of clip_grad, got {type(clip_grad)}, expected bool"
        assert clip_norm > 0.0 and isinstance(clip_norm, float), f"clip_norm must be float > 0, but got {clip_norm}"
        self.clip_grad = clip_grad
        self.clip_norm = clip_norm

        assert gradient_accumulation_steps >= 1
        self.grad_accu_steps = gradient_accumulation_steps
        if gradient_accumulation_steps > 1:
            # additionally caches network trainable parameters. overhead caused.
            # TODO: try to store it in CPU memory instead of GPU/NPU memory.
            self.accumulated_grads = optimizer.parameters.clone(prefix="grad_accumulated_", init="zeros")
            self.zeros = optimizer.parameters.clone(prefix="zeros_", init="zeros")
            self.cur_accu_step = Parameter(Tensor(0, ms.int32), "grad_accumulate_step_", requires_grad=False)
            self.zero = Tensor(0, ms.int32)
            for p in self.accumulated_grads:
                p.requires_grad = False
            for z in self.zeros:
                z.requires_grad = False

        self.verbose = verbose
        self.is_cpu_device = context.get_context("device_target") == "CPU"  # to support CPU in CI
        self.skip_start_overflow_check = version.parse(ms.__version__) >= version.parse("2.1")

        self.map = ops.Map()
        self.partial = ops.Partial()

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

        # 2. down-scale gradients by loss_scale. grads = grads / scaling_sense  / grad_accu_steps
        # also divide gradients by accumulation steps to avoid taking mean of  the accumulated gradients later
        grads = self.hyper_map(F.partial(_grad_scale, scaling_sens * self.grad_accu_steps), grads)

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
            if self.grad_accu_steps > 1:
                # self.accumulated_grads += grads
                loss = F.depend(loss, self.map(self.partial(ops.assign_add), self.accumulated_grads, grads))
                # self.cur_accu_step += 1
                loss = F.depend(loss, ops.assign_add(self.cur_accu_step, Tensor(1, ms.int32)))

                if self.cur_accu_step % self.grad_accu_steps == 0:
                    # 5. gradient reduction on distributed GPUs/NPUs
                    grads = self.grad_reducer(self.accumulated_grads)

                    # 6. clip grad
                    if self.clip_grad:
                        grads = ops.clip_by_global_norm(grads, self.clip_norm)
                    # 7. optimize
                    loss = F.depend(loss, self.optimizer(grads))

                    # clear gradient accumulation states
                    loss = F.depend(
                        loss, self.map(self.partial(ops.assign), self.accumulated_grads, self.zeros)
                    )  # self.accumulated_grads = 0
                    loss = F.depend(loss, ops.assign(self.cur_accu_step, self.zero))  # self.cur_accu_step = 0
                else:
                    # update LR in each gradient step but not optimize net parameter to ensure the LR curve is
                    # consistent
                    loss = F.depend(
                        loss, ops.assign_add(self.optimizer.global_step, self.optimizer.global_step_increase_tensor)
                    )
            else:
                # 5. gradient reduction on distributed GPUs/NPUs
                grads = self.grad_reducer(grads)
                # 6. clip grad
                if self.clip_grad:
                    grads = ops.clip_by_global_norm(grads, self.clip_norm)
                # 7. optimize
                loss = F.depend(loss, self.optimizer(grads))

            # 8.ema
            if self.ema is not None:
                self.ema.ema_update()
        # else:
        #    print("WARNING: Gradient overflow! update skipped.") # TODO: recover it after 910B in-graph print issue fixed
        #    pass

        return loss, cond, scaling_sens
