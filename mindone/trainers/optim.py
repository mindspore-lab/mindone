"""
Build optimizer for ms
"""
import logging
from typing import List, Optional, Union

from mindcv.optim.adamw import AdamW as AdamW_Refined

from mindspore.common.parameter import Parameter
from mindspore.nn.optim import Adam, AdamWeightDecay, Momentum, Optimizer

from .adamw_mf import AdamW as AdamW_MF
from .adamw_zero1 import AdamWeightDecayZeRO1
from .came import CAME

_logger = logging.getLogger(__name__)


def create_optimizer(
    params: Union[List[Parameter], List[dict]],
    name: str,
    lr: Union[float, List[float]],
    betas: Optional[List[float]] = None,
    weight_decay: float = 1e-6,
    eps: Union[float, List[float]] = 1e-6,
    group_strategy: Optional[str] = None,
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
        group_strategy: The specific grouping startegy for weight decay. If it is None,
                        then only the weight decays for parameters in layernorm and all bias will be set to 0.

    Returns:
        Initialized optimizer.
    """
    if betas is None:
        betas = [0.9, 0.999]

    if group_strategy is not None:
        _logger.info("Applying `%s` strategy for weight decay.", group_strategy)

    def decay_filter(param):
        if group_strategy is None:
            filter_list = ["layernorm", "bias"]
        elif group_strategy.lower() == "unclip":
            # set decay of embedding to 0 should be beneficial for most of the cases
            filter_list = ["gamma", "beta", "bias", "label_emb", "time_embed", "emb_layers"]
        elif group_strategy.lower() == "norm_and_bias":
            # filter norm and bias
            filter_list = ["gamma", "beta", "bias"]
        elif group_strategy.lower() == "not_grouping":
            filter_list = []
        else:
            raise ValueError(f"Unsupported group_strategy: '{group_strategy}'")

        return all([x not in param.name.lower() for x in filter_list])

    param_optimizer = params
    decay_params = list(filter(decay_filter, param_optimizer))
    other_params = list(filter(lambda x: not decay_filter(x), param_optimizer))
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
    elif name.lower() == "adamw_mf":
        optim_cls = AdamW_MF
    elif name.lower() == "adamw_zero1":
        optim_cls = AdamWeightDecayZeRO1
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
