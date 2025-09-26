"""
Build optimizer for ms
"""
import logging
from typing import List, Optional, Union

from mindcv.optim.adamw import AdamW as AdamW_Refined

from mindspore.common.parameter import Parameter
from mindspore.nn.optim import Adam, AdamWeightDecay, Momentum, Optimizer

from mindone.trainers.came import CAME

_logger = logging.getLogger(__name__)


def create_optimizer(
    params: Union[List[Parameter], List[dict]],
    name: str,
    lr: Union[float, List[float]],
    betas: Optional[List[float]] = None,
    weight_decay: float = 1e-6,
    eps: Union[float, List[float]] = 1e-6,
    nowd_keys=None,
) -> Optimizer:
    """
    Build and return an instance of the Optimizer class based on the specified parameters.

    Args:
        params: Model parameters to be optimized.
        name: Name of the optimizer. adamw_re: refined adamw
        lr: Learning rate or a list of learning rates for each step (if a scheduler is used).
        betas: Beta coefficients for computing running averages of gradient and its square.
               If not provided, [0.9, 0.999] is used as default.
        weight_decay: Weight decay (L2 penalty) coefficient. Default is 1e-6.
        eps: epsilon in adam or adamw optimization, Default: 1e-6
        nowd_keys: No weight decay parameters.

    Returns:
        Initialized optimizer.
    """
    if betas is None:
        betas = [0.9, 0.999]

    def nodecay_filter(param):
        return all([param.value().ndim == 1 or param.name.endswith("bias") or any(k in param.name for k in nowd_keys)])

    param_optimizer = params
    other_params = list(filter(nodecay_filter, param_optimizer))
    decay_params = list(filter(lambda x: not nodecay_filter(x), param_optimizer))
    group_params = []
    if len(decay_params) > 0:
        group_params.append({"params": decay_params, "weight_decay": weight_decay})  # 1e-6})
    if len(other_params) > 0:
        group_params.append({"params": other_params, "weight_decay": 0.0})
    group_params.append({"order_params": param_optimizer})
    _logger.info(f"Parameter grouping result: weight decay {len(decay_params)}, no weight decay {len(other_params)}")

    if name.lower() == "adam":
        optim_cls = Adam
    elif name.lower() == "adamw":
        optim_cls = AdamWeightDecay
    elif name.lower() == "adamw_re":
        optim_cls = AdamW_Refined
    elif name.lower() in ["sgd", "momentum"]:
        optim_cls = Momentum
    elif name.lower() == "came":
        optim_cls = CAME
    else:
        raise ValueError("invalid optimizer")

    if name.lower() in ["sgd", "momentum"]:
        optimizer = optim_cls(group_params, learning_rate=lr, momentum=0.9)
    elif name.lower() in ["adamw_mf", "came"]:
        optimizer = optim_cls(group_params, learning_rate=lr, betas=betas, eps=eps)
    else:
        optimizer = optim_cls(group_params, learning_rate=lr, beta1=betas[0], beta2=betas[1], eps=eps)

    return optimizer
