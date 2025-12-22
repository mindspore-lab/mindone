from typing import TYPE_CHECKING, Dict, Literal, Optional

import mindspore as ms
from mindspore import ParallelMode, Tensor, context, nn, ops
from mindspore.amp import LossScaler
from mindspore.ops import composite as C

if TYPE_CHECKING:
    from mindone.trainers.zero import ZeroHelper

try:
    from .adamw_zero import AdamWeightDecayZeRO1, AdamWeightDecayZeRO2

    is_adamw_zero_available = True
except ImportError:
    is_adamw_zero_available = False


_grad_accum_op = C.MultitypeFuncGraph("gradient_accumulation_op")


@_grad_accum_op.register("Tensor", "Tensor")
def cumulative_grad_process(cumulative_grad, grad):
    """Apply gradient accumulation to cumulative grad."""
    cumulative_grad.add_(grad)


_grad_clear_op = C.MultitypeFuncGraph("gradient_clear_op")


@_grad_clear_op.register("Tensor")
def clear_grad(cumulative_grad):
    """Clear grad."""
    cumulative_grad.mul_(0.0)


def _is_pynative_parallel():
    parallel_mode = context.get_auto_parallel_context("parallel_mode")
    return context.get_context("mode") == context.PYNATIVE_MODE and parallel_mode in (
        context.ParallelMode.SEMI_AUTO_PARALLEL,
        context.ParallelMode.AUTO_PARALLEL,
    )


def create_loss_scaler(ms_loss_scaler="static", scale_value=1024, scale_factor=2, scale_window=1000):
    if ms_loss_scaler == "dynamic":
        from mindspore.amp import DynamicLossScaler

        loss_scaler = DynamicLossScaler(scale_value=scale_value, scale_factor=scale_factor, scale_window=scale_window)
    elif ms_loss_scaler == "static":
        from mindspore.amp import StaticLossScaler

        loss_scaler = StaticLossScaler(scale_value=scale_value)
    elif ms_loss_scaler in ("none", "None"):
        from mindspore.amp import StaticLossScaler

        loss_scaler = StaticLossScaler(1.0)
    else:
        raise NotImplementedError(f"Not support ms_loss_scaler: {ms_loss_scaler}")

    return loss_scaler


def _is_parallel():
    is_parallel = (
        context.get_auto_parallel_context("parallel_mode") in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL)
        or _is_pynative_parallel()
    )
    return is_parallel


def _is_cpu():
    return context.get_context("device_target") == "CPU"


def return_true(*args, **kwargs):
    return ops.ones((), ms.bool_)


def create_grad_reducer(trainable_parameters):
    use_reducer = _is_parallel()

    if use_reducer:
        mean = context.get_auto_parallel_context("gradients_mean")
        degree = context.get_auto_parallel_context("device_num")
        grad_reducer = nn.DistributedGradReducer(trainable_parameters, mean, degree)
    else:
        grad_reducer = nn.Identity()
    return grad_reducer


class LossWithScaleSense(nn.Cell):
    def __init__(self, network: nn.Cell, scaler: LossScaler) -> None:
        super().__init__(auto_prefix=False)
        self.network = network
        self.scaler = scaler

    def set_train(self, mode: bool = True):
        # Delegate control of training-mode behavior to the network.
        self.network.set_train(mode)

    def construct(self, *args, **kwargs) -> Tensor:
        loss = self.network(*args, **kwargs)
        scale_sense = self.scaler.scale_value
        loss = loss * scale_sense.to(loss.dtype)
        return loss


class TrainOneStepWrapper(nn.Cell):
    """TrainStep with ema and clip grad.

    Returns:
        Tuple of 3 Tensor, the loss, overflow flag and current loss scale value.
        loss (Tensor) -  A scalar, the loss value.
        overflow (Tensor) -  A scalar, whether overflow occur or not, the type is bool.
        loss scale (Tensor) -  The loss scale value, the shape is :math:`()` or :math:`(1,)`.

    """

    def __init__(
        self,
        network: nn.Cell,
        optimizer: nn.Optimizer,
        ema: nn.Cell = None,
        drop_overflow_step: bool = True,
        scaler: Literal["default", "static", "auto", "dynamic", "none"] = "default",
        scaler_config: Dict = {},
        gradient_accumulation_steps: int = 1,
        clip_grad: Literal[
            "norm", "l2norm", "l2_norm", "global", "global_norm", "total", "total_norm", "local", "value", "none"
        ] = "none",
        clip_value: float = 1.0,
        zero_helper: Optional["ZeroHelper"] = None,
    ):
        super().__init__(auto_prefix=False)

        if is_adamw_zero_available and isinstance(optimizer, (AdamWeightDecayZeRO1, AdamWeightDecayZeRO2)):
            assert hasattr(optimizer, "grad_reduce")
            reducer = None
            if optimizer.shard_size > 1:
                is_zero = True
                self.reduce_op_for_clip_grad = ops.AllReduce(group=optimizer.comm_group)
            else:
                is_zero = False
        else:
            reducer = create_grad_reducer(network.trainable_params())
            is_zero = False

        # scaler and reducer
        assert "ms_loss_scaler" not in scaler_config
        if scaler.lower() in ("default", "static"):
            _scaler_config = {"scale_value": 1024}
            _scaler_config.update(scaler_config)
            scaler = create_loss_scaler("static", **_scaler_config)
        elif scaler.lower() in ("auto", "dynamic"):
            scaler = create_loss_scaler("dynamic", **scaler_config)
        elif scaler.lower() == "none":
            scaler = create_loss_scaler("none", **scaler_config)
        else:
            raise NotImplementedError

        self.scaler = scaler

        # wrap network with scale sense
        network = LossWithScaleSense(network, self.scaler)

        # grad accumulation
        assert gradient_accumulation_steps >= 1
        self.accum_steps = gradient_accumulation_steps
        if gradient_accumulation_steps > 1:
            self.hyper_map = ops.HyperMap()
            self.cur_accum_step = ms.Parameter(ms.Tensor(0, dtype=ms.int32), name="accum_step", requires_grad=False)

            if is_zero:
                self.accumulated_grads = optimizer.moments1.clone(prefix="accum_grad", init="zeros")  # split grads
            elif zero_helper is not None:
                self.accumulated_grads = optimizer.parameters.clone(prefix="accum_grad", init="zeros")

            class ScalingLossForGradAccum(nn.Cell):
                def __init__(self, net, accum_steps_):
                    super(ScalingLossForGradAccum, self).__init__(auto_prefix=False)
                    self.net = net
                    self.accum_steps_ = accum_steps_

                def set_train(self, mode: bool = True):
                    # Delegate control of training-mode behavior to the network.
                    self.net.set_train(mode)

                def construct(self, *args, **kwargs):
                    loss = self.net(*args, **kwargs)
                    return loss / self.accum_steps_

            network = ScalingLossForGradAccum(network, gradient_accumulation_steps)

        # grad and optimizer
        self.network = network

        # not suitable for multimodal llm
        # self.network.set_train()
        # self.network.set_grad()

        self.value_and_grad = ops.value_and_grad(network, grad_position=None, weights=optimizer.parameters)

        self.optimizer = optimizer
        self.ema = ema
        self.reducer = reducer
        self.is_zero = is_zero
        self.all_finite = ms.amp.all_finite if not _is_cpu() else return_true
        self.all_finite_reducer = ops.AllReduce() if _is_parallel() else nn.Identity()
        self.drop_overflow_step = Tensor(drop_overflow_step, ms.bool_)

        # clip grad
        assert clip_value > 0.0 and isinstance(
            clip_value, float
        ), f"clip_value must be float > 0., but got {clip_value}"
        self.clip_value = clip_value
        self.is_clip_norm = False
        if clip_grad.lower() in ("norm", "l2norm", "l2_norm", "global", "global_norm", "total", "total_norm"):
            self.is_clip_norm = True
            if self.is_zero:
                from mindone.transformers.mindspore_adapter.clip_grad import clip_grad_norm_for_zero

                clip_grad_fn = clip_grad_norm_for_zero
            else:
                from mindone.transformers.mindspore_adapter.clip_grad import clip_grad_norm

                clip_grad_fn = clip_grad_norm
        elif clip_grad.lower() in ("local", "value"):
            from mindone.transformers.mindspore_adapter.clip_grad import clip_grad_value

            clip_grad_fn = clip_grad_value
        elif clip_grad.lower() == "none":
            clip_grad_fn = None
        else:
            raise NotImplementedError
        self.clip_grad_fn = clip_grad_fn

        # zero init
        self.zero_helper = zero_helper
        self.zero_stage = zero_helper.zero_stage if zero_helper is not None else 0
        self.run_optimizer = zero_helper.run_optimizer if zero_helper is not None else self.optimizer
        self.reducer = self.reducer if self.zero_stage == 0 else nn.Identity()
        if self.zero_stage != 0:
            self.zero_helper.split_params()
            if gradient_accumulation_steps > 1:
                self.accumulated_grads = optimizer.parameters.clone(prefix="accum_grad", init="zeros")

    def set_train(self, mode: bool = True):
        # Delegate control of training-mode behavior to the network.
        self.network.set_train(mode)

    def do_optim(self, loss, grads):
        if self.accum_steps == 1:
            if self.clip_grad_fn is not None:
                if self.is_zero and self.is_clip_norm:
                    grads = self.clip_grad_fn(grads, self.clip_value, self.reduce_op_for_clip_grad)
                else:
                    grads = self.clip_grad_fn(grads, self.clip_value)
            loss = ops.depend(loss, self.run_optimizer(grads))
            if self.ema is not None:
                self.ema.ema_update()
        else:
            loss = ops.depend(loss, self.hyper_map(_grad_accum_op, self.accumulated_grads, grads))
            loss = ops.depend(loss, self.cur_accum_step.add_(1))
            if self.cur_accum_step % self.accum_steps == 0:
                if self.clip_grad_fn is not None:
                    if self.is_zero and self.is_clip_norm:
                        clipped_grads = self.clip_grad_fn(
                            self.accumulated_grads, self.clip_value, self.reduce_op_for_clip_grad
                        )
                    else:
                        clipped_grads = self.clip_grad_fn(self.accumulated_grads, self.clip_value)

                    loss = ops.depend(loss, self.run_optimizer(clipped_grads))
                else:
                    loss = ops.depend(loss, self.run_optimizer(self.accumulated_grads))

                loss = ops.depend(loss, self.hyper_map(ops.partial(_grad_clear_op), self.accumulated_grads))
                loss = ops.depend(loss, ops.assign(self.cur_accum_step, ms.Tensor(0, ms.int32)))
                if self.ema is not None:
                    self.ema.ema_update()
            else:
                # update the optimizer global step and learning rate, do not update the parameter
                loss = ops.depend(loss, self.optimizer.global_step.add_(self.optimizer.global_step_increase_tensor))

            # unscaling loss for grad accum
            loss = loss * self.accum_steps

        return loss

    def construct(self, *inputs):
        loss, grads = self.value_and_grad(*inputs)
        loss = self.scaler.unscale(loss)
        loss = loss * self.accum_steps

        if self.zero_helper is not None:
            grads = self.zero_helper.cal_gradients(grads)
        elif self.is_zero:
            grads = self.optimizer.grad_reduce(grads)
        else:
            grads = self.reducer(grads)

        unscaled_grads = self.scaler.unscale(grads)

        finite = self.all_finite(unscaled_grads)
        finite = ops.equal(
            self.all_finite_reducer(finite.to(ms.int32)), self.all_finite_reducer(ops.ones((), ms.int32))
        ).to(ms.bool_)
        finite = ops.depend(finite, self.scaler.adjust(finite)).to(ms.bool_)

        if not self.drop_overflow_step:
            loss = self.do_optim(loss, unscaled_grads)
            loss = loss.to(ms.float32)
        else:
            if finite:
                loss = self.do_optim(loss, unscaled_grads)
                loss = loss.to(ms.float32)
            else:
                # FIXME: has bug when run amp fp16 on MindSpore 2.3
                loss = loss.to(ms.float32)

        overflow_tag = not finite

        return loss, unscaled_grads, overflow_tag
