"""
Build optimizer for ms
"""
from typing import List, Union

from mindspore.nn import Cell
from mindspore.nn.optim import Adam, AdamWeightDecay, Momentum, Optimizer


def build_optimizer(
    model: Cell,
    name: str,
    lr: Union[float, List[float]],
    betas: List[float] = None,
    weight_decay: float = 1e-6,
) -> Optimizer:
    """
    Build and return an instance of the Optimizer class based on the specified parameters.

    Args:
        model: Model to which apply the optimizer.
        name: Name of the optimizer.
        lr: Learning rate or a list of learning rates for each step (if a scheduler is used).
        betas: Beta coefficients for computing running averages of gradient and its square.
            If not provided, [0.9, 0.999] is used as default.
        weight_decay: Weight decay (L2 penalty) coefficient. Default is 1e-6.

    Returns:
        Initialized optimizer.
    """
    if betas is None:
        betas = [0.9, 0.999]

    def decay_filter(x):
        return "layernorm" not in x.name.lower() and "bias" not in x.name.lower()

    param_optimizer = model.trainable_params()
    decay_params = list(filter(decay_filter, param_optimizer))
    other_params = list(filter(lambda x: not decay_filter(x), param_optimizer))
    group_params = []
    if len(decay_params) > 0:
        group_params.append({"params": decay_params, "weight_decay": weight_decay})  # 1e-6})
    if len(other_params) > 0:
        group_params.append({"params": other_params, "weight_decay": 0.0})
    group_params.append({"order_params": param_optimizer})

    if name.lower() == "adam":
        OptimCls = Adam
    elif name.lower() == "adamw":
        OptimCls = AdamWeightDecay
    elif name.lower() in ["sgd", "momentum"]:
        OptimCls = Momentum
    else:
        raise ValueError("invalid optimizer")

    if name.lower() in ["sgd", "momentum"]:
        optimizer = OptimCls(group_params, learning_rate=lr, momentum=0.9)
    else:
        optimizer = OptimCls(group_params, learning_rate=lr, beta1=betas[0], beta2=betas[1])

    return optimizer
