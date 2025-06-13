import math
import warnings
from functools import partial
from typing import Optional, Union

from transformers.trainer_utils import SchedulerType


def get_constant_schedule(base_lr):
    """
    Create a schedule with a constant learning rate, using the learning rate set in optimizer.
    """

    return base_lr


def _get_constant_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1.0, num_warmup_steps))
    return 1.0


def get_constant_schedule_with_warmup(base_lr: float, num_warmup_steps: int, num_training_steps: int):
    """
    Create a schedule with a constant learning rate preceded by a warmup period during which the learning rate
    increases linearly between 0 and the initial lr set in the optimizer.

    Args:
        base_lr (`float`):
            The base learning rate for scheduler.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.

    Return:
        `List` with the appropriate schedule.
    """

    lr_lambda = partial(_get_constant_schedule_with_warmup_lr_lambda, num_warmup_steps=num_warmup_steps)
    return [base_lr * lr_lambda(cur_step) for cur_step in range(num_training_steps)]


def _get_linear_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int, num_training_steps: int):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))


def get_linear_schedule_with_warmup(base_lr: float, num_warmup_steps: int, num_training_steps: int):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        base_lr (`float`):
            The base learning rate for scheduler.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.

    Return:
        `List` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_linear_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    return [base_lr * lr_lambda(cur_step) for cur_step in range(num_training_steps)]


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


def get_cosine_schedule_with_warmup(
    base_lr: float, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        base_lr (`float`):
            The base learning rate for scheduler.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).

    Return:
        `List` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return [base_lr * lr_lambda(cur_step) for cur_step in range(num_training_steps)]


def _get_cosine_with_hard_restarts_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: int
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    if progress >= 1.0:
        return 0.0
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))


def get_cosine_with_hard_restarts_schedule_with_warmup(
    base_lr: float, num_warmup_steps: int, num_training_steps: int, num_cycles: int = 1, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`int`, *optional*, defaults to 1):
            The number of hard restarts to use.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_cosine_with_hard_restarts_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return [base_lr * lr_lambda(cur_step) for cur_step in range(num_training_steps)]


def _get_polynomial_decay_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    lr_end: float,
    power: float,
    lr_init: int,
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    elif current_step > num_training_steps:
        return lr_end / lr_init  # as LambdaLR multiplies by lr_init
    else:
        lr_range = lr_init - lr_end
        decay_steps = num_training_steps - num_warmup_steps
        pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
        decay = lr_range * pct_remaining**power + lr_end
        return decay / lr_init  # as LambdaLR multiplies by lr_init


def get_polynomial_decay_schedule_with_warmup(
    base_lr: float, num_warmup_steps, num_training_steps, lr_end=1e-7, power=1.0, last_epoch=-1
):
    """
    Create a schedule with a learning rate that decreases as a polynomial decay from the initial lr set in the
    optimizer to end lr defined by *lr_end*, after a warmup period during which it increases linearly from 0 to the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        lr_end (`float`, *optional*, defaults to 1e-7):
            The end LR.
        power (`float`, *optional*, defaults to 1.0):
            Power factor.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Note: *power* defaults to 1.0 as in the fairseq implementation, which in turn is based on the original BERT
    implementation at
    https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/optimization.py#L37

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.

    """
    if not (base_lr > lr_end):
        raise ValueError(f"lr_end ({lr_end}) must be smaller than initial lr ({base_lr})")

    lr_lambda = partial(
        _get_polynomial_decay_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        lr_end=lr_end,
        power=power,
        lr_init=base_lr,
    )
    return [base_lr * lr_lambda(cur_step) for cur_step in range(num_training_steps)]


def _get_inverse_sqrt_schedule_lr_lambda(current_step: int, *, num_warmup_steps: int, timescale: int = None):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    shift = timescale - num_warmup_steps
    decay = 1.0 / math.sqrt((current_step + shift) / timescale)
    return decay


def get_inverse_sqrt_schedule(
    base_lr: float, num_warmup_steps: int, num_training_steps: int, timescale: int = None, last_epoch: int = -1
):
    """
    Create a schedule with an inverse square-root learning rate, from the initial lr set in the optimizer, after a
    warmup period which increases lr linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        timescale (`int`, *optional*, defaults to `num_warmup_steps`):
            Time scale.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    # Note: this implementation is adapted from
    # https://github.com/google-research/big_vision/blob/f071ce68852d56099437004fd70057597a95f6ef/big_vision/utils.py#L930

    if timescale is None:
        timescale = num_warmup_steps or 10_000

    lr_lambda = partial(_get_inverse_sqrt_schedule_lr_lambda, num_warmup_steps=num_warmup_steps, timescale=timescale)
    return [base_lr * lr_lambda(cur_step) for cur_step in range(num_training_steps)]


def _get_cosine_schedule_with_min_lr_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float, min_lr_rate: float = 0.0
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    factor = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
    factor = factor * (1 - min_lr_rate) + min_lr_rate
    return max(0, factor)


def get_cosine_with_min_lr_schedule_with_warmup(
    base_lr: float,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    min_lr: float = None,
    min_lr_rate: float = None,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to min_lr, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
        min_lr (`float`, *optional*):
            The minimum learning rate to reach after the cosine schedule.
        min_lr_rate (`float`, *optional*):
            The minimum learning rate as a ratio of the initial learning rate. If set, `min_lr` should not be set.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    if min_lr is not None and min_lr_rate is not None:
        raise ValueError("Only one of min_lr or min_lr_rate should be set")
    elif min_lr is not None:
        min_lr_rate = min_lr / base_lr
    elif min_lr_rate is None:
        raise ValueError("One of min_lr or min_lr_rate should be set through the `lr_scheduler_kwargs`")

    lr_lambda = partial(
        _get_cosine_schedule_with_min_lr_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        min_lr_rate=min_lr_rate,
    )
    return [base_lr * lr_lambda(cur_step) for cur_step in range(num_training_steps)]


def _get_wsd_scheduler_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_stable_steps: int,
    num_decay_steps: int,
    warmup_type: str,
    decay_type: str,
    min_lr_ratio: float,
    num_cycles: float,
):
    if current_step < num_warmup_steps:
        progress = float(current_step) / float(max(1, num_warmup_steps))
        if warmup_type == "linear":
            factor = progress
        elif warmup_type == "cosine":
            factor = 0.5 * (1.0 - math.cos(math.pi * progress))
        elif warmup_type == "1-sqrt":
            factor = 1.0 - math.sqrt(1.0 - progress)
        factor = factor * (1.0 - min_lr_ratio) + min_lr_ratio
        return max(0.0, factor)

    if current_step < num_warmup_steps + num_stable_steps:
        return 1.0

    if current_step < num_warmup_steps + num_stable_steps + num_decay_steps:
        progress = float(current_step - num_warmup_steps - num_stable_steps) / float(max(1, num_decay_steps))
        if decay_type == "linear":
            factor = 1.0 - progress
        elif decay_type == "cosine":
            factor = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        elif decay_type == "1-sqrt":
            factor = 1.0 - math.sqrt(progress)
        factor = factor * (1.0 - min_lr_ratio) + min_lr_ratio
        return max(0.0, factor)
    return min_lr_ratio


def get_wsd_schedule(
    base_lr: float,
    num_warmup_steps: int,
    num_decay_steps: int,
    num_training_steps: Optional[int] = None,
    num_stable_steps: Optional[int] = None,
    warmup_type: str = "linear",
    decay_type: str = "cosine",
    min_lr_ratio: float = 0,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that has three stages:
    1. warmup: increase from min_lr_ratio times the initial learning rate to the initial learning rate following a warmup_type.
    2. stable: constant learning rate.
    3. decay: decrease from the initial learning rate to min_lr_ratio times the initial learning rate following a decay_type.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_decay_steps (`int`):
            The number of steps for the decay phase.
        num_training_steps (`int`, *optional*):
            The total number of training steps. This is the sum of the warmup, stable and decay steps.
            If `num_stable_steps` is not provided, the stable phase will be `num_training_steps - num_warmup_steps - num_decay_steps`.
        num_stable_steps (`int`, *optional*):
            The number of steps for the stable phase. Please ensure that `num_warmup_steps + num_stable_steps + num_decay_steps` equals `num_training_steps`,
            otherwise the other steps will default to the minimum learning rate.
        warmup_type (`str`, *optional*, defaults to "linear"):
            The type of warmup to use. Can be 'linear', 'cosine' or '1-sqrt'.
        decay_type (`str`, *optional*, defaults to "cosine"):
            The type of decay to use. Can be 'linear', 'cosine' or '1-sqrt'.
        min_lr_ratio (`float`, *optional*, defaults to 0):
            The minimum learning rate as a ratio of the initial learning rate.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    if num_training_steps is None and num_stable_steps is None:
        raise ValueError("Either num_training_steps or num_stable_steps must be specified.")

    if num_training_steps is not None and num_stable_steps is not None:
        warnings.warn("Both num_training_steps and num_stable_steps are specified. num_stable_steps will be used.")

    if warmup_type not in ["linear", "cosine", "1-sqrt"]:
        raise ValueError(f"Unknown warmup type: {warmup_type}, expected 'linear', 'cosine' or '1-sqrt'")

    if decay_type not in ["linear", "cosine", "1-sqrt"]:
        raise ValueError(f"Unknown decay type: {decay_type}, expected 'linear', 'cosine' or '1-sqrt'")

    if num_stable_steps is None:
        num_stable_steps = num_training_steps - num_warmup_steps - num_decay_steps

    lr_lambda = partial(
        _get_wsd_scheduler_lambda,
        num_warmup_steps=num_warmup_steps,
        num_stable_steps=num_stable_steps,
        num_decay_steps=num_decay_steps,
        warmup_type=warmup_type,
        decay_type=decay_type,
        min_lr_ratio=min_lr_ratio,
        num_cycles=num_cycles,
    )
    return [base_lr * lr_lambda(cur_step) for cur_step in range(num_training_steps)]


TYPE_TO_SCHEDULER_FUNCTION = {
    SchedulerType.LINEAR: get_linear_schedule_with_warmup,
    SchedulerType.COSINE: get_cosine_schedule_with_warmup,
    SchedulerType.COSINE_WITH_RESTARTS: get_cosine_with_hard_restarts_schedule_with_warmup,
    SchedulerType.POLYNOMIAL: get_polynomial_decay_schedule_with_warmup,
    SchedulerType.CONSTANT: get_constant_schedule,
    SchedulerType.CONSTANT_WITH_WARMUP: get_constant_schedule_with_warmup,
    SchedulerType.INVERSE_SQRT: get_inverse_sqrt_schedule,
    SchedulerType.REDUCE_ON_PLATEAU: None,
    SchedulerType.COSINE_WITH_MIN_LR: get_cosine_with_min_lr_schedule_with_warmup,
    SchedulerType.WARMUP_STABLE_DECAY: get_wsd_schedule,
}


def get_scheduler(
    name: Union[str, SchedulerType],
    base_lr: Optional[float],
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    scheduler_specific_kwargs: Optional[dict] = None,
):
    """
    Unified API to get any scheduler from its name.

    Args:
        name (`str` or `SchedulerType`):
            The name of the scheduler to use.
        optimizer (`mindspore.nn.Optimizer`):
            The optimizer that will be used during training.
        num_warmup_steps (`int`, *optional*):
            The number of warmup steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_training_steps (`int``, *optional*):
            The number of training steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        scheduler_specific_kwargs (`dict`, *optional*):
            Extra parameters for schedulers such as cosine with restarts. Mismatched scheduler types and scheduler
            parameters will cause the scheduler function to raise a TypeError.
    """
    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]

    # Note: Not support `LayerWiseDummyOptimizer` now.
    # If a `LayerWiseDummyOptimizer` is passed we extract the optimizer dict and
    # recursively call `get_scheduler` to get the proper schedulers on each parameter
    # if optimizer is not None and isinstance(optimizer, LayerWiseDummyOptimizer):
    #     raise NotImplementedError

    if name == SchedulerType.CONSTANT:
        return schedule_func(base_lr=base_lr)

    if scheduler_specific_kwargs is None:
        scheduler_specific_kwargs = {}

    if name == SchedulerType.REDUCE_ON_PLATEAU:
        raise NotImplementedError

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    return schedule_func(
        base_lr=base_lr,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        **scheduler_specific_kwargs,
    )
