"""
Build optimizer for ms
"""
import logging
from typing import List, Optional, Union

from mindcv.optim.adamw import AdamW as AdamW_Refined

from mindspore.common.parameter import Parameter
from mindspore.nn.optim import Adam, AdamWeightDecay, Momentum, Optimizer

from mindone.trainers.adamw_bf16 import BF16AdamW
from mindone.trainers.adamw_mf import AdamW as AdamW_MF
from mindone.trainers.adamw_mint import AdamW as AdamW_Mint
from mindone.trainers.adamw_zero1 import AdamWeightDecayZeRO1
from mindone.trainers.came import CAME

_logger = logging.getLogger(__name__)


def create_optimizer(
    params: Union[List[Parameter], List[dict]],
    name: str,
    lr: Union[float, List[float]],
    betas: Optional[List[float]] = None,
    weight_decay: float = 1e-6,
    eps: Union[float, List[float]] = 1e-6,
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

    Returns:
        Initialized optimizer.
    """
    if betas is None:
        betas = [0.9, 0.999]

    param_optimizer = params

    if name.lower() == "adam":
        optim_cls = Adam
    elif name.lower() == "adamw":
        optim_cls = AdamWeightDecay
    elif name.lower() == "adamw_re":
        optim_cls = AdamW_Refined
    elif name.lower() == "adamw_bf16":
        optim_cls = BF16AdamW
    elif name.lower() == "adamw_mf":
        optim_cls = AdamW_MF
    elif name.lower() == "adamw_zero1":
        optim_cls = AdamWeightDecayZeRO1
    elif name.lower() == "adamw_mint":
        optim_cls = AdamW_Mint
    elif name.lower() in ["sgd", "momentum"]:
        optim_cls = Momentum
    elif name.lower() == "came":
        optim_cls = CAME
    else:
        raise ValueError("invalid optimizer")

    if name.lower() in ["sgd", "momentum"]:
        optimizer = optim_cls(param_optimizer, learning_rate=lr, momentum=0.9)
    elif name.lower() in ["adamw_mf", "came"]:
        optimizer = optim_cls(param_optimizer, learning_rate=lr, betas=betas, eps=eps)
    else:
        optimizer = optim_cls(param_optimizer, learning_rate=lr, beta1=betas[0], beta2=betas[1], eps=eps)

    return optimizer
