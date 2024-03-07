"""
Build optimizer for ms
"""
import logging
from typing import List, Optional, Union

from mindspore.nn import Cell
from mindspore.nn.optim import Adam, AdamWeightDecay, Momentum, Optimizer

_logger = logging.getLogger(__name__)


def build_optimizer(
    model: Cell,
    name: str,
    lr: Union[float, List[float]],
    betas: Optional[List[float]] = None,
    group_lr_scaler: float = 1.0,
    weight_decay: float = 1e-6,
    eps: float = 1e-6,
    group_strategy: Optional[str] = None,
) -> Optimizer:
    """
    Build and return an instance of the Optimizer class based on the specified parameters.

    Args:
        model: Model to which apply the optimizer.
        name: Name of the optimizer.
        lr: Learning rate or a list of learning rates for each step (if a scheduler is used).
        betas: Beta coefficients for computing running averages of gradient and its square.
            If not provided, [0.9, 0.999] is used as default.
        group_lr_scaler: Set different learning rate for particular group of params.
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
        else:
            raise ValueError(f"Unsupported group_strategy: `{group_strategy}`")

        return all([x not in param.name.lower() for x in filter_list])

    def _scale_lr(group_params, lr, scaler):
        new_groups = list()
        for group in group_params:
            scale_params, unscale_params = list(), list()
            for params in group["params"]:
                name = params.name.lower()
                if "zero_conv" in name or "input_hint_block" in name or "middle_block_out" in name:
                    scale_params.append(params)
                else:
                    unscale_params.append(params)

            new_groups.append(
                {
                    "params": scale_params,
                    "weight_decay": group["weight_decay"],
                    "lr": [i * scaler for i in lr],
                }
            )
            new_groups.append(
                {
                    "params": unscale_params,
                    "weight_decay": group["weight_decay"],
                    "lr": lr,
                }
            )
        _logger.info(f"Enable scale lr for zero conv layers, scale lr: {scaler * lr[0]}")
        return new_groups

    param_optimizer = model.trainable_params()
    decay_params = list(filter(decay_filter, param_optimizer))
    other_params = list(filter(lambda x: not decay_filter(x), param_optimizer))
    group_params = []
    if len(decay_params) > 0:
        group_params.append({"params": decay_params, "weight_decay": weight_decay})  # 1e-6})
    if len(other_params) > 0:
        group_params.append({"params": other_params, "weight_decay": 0.0})

    _info = "\n".join(
        [
            f"Enable optimizer group param, "
            f"decay params num: {len(decay_params)}, "
            f"no decay params num: {len(other_params)}, "
            f"full params num: {len(decay_params) + len(other_params)}"
        ]
    )
    _logger.info(_info)

    # set different lr for zero_conv/input_hint_block/middle_block_out layers of cldm
    if group_lr_scaler != 1.0:
        group_params = _scale_lr(group_params, lr, group_lr_scaler)
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
        optimizer = OptimCls(group_params, learning_rate=lr, beta1=betas[0], beta2=betas[1], eps=eps)

    return optimizer
