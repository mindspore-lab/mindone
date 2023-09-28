"""
 build optimizer for ms
"""
from mindspore import nn
from mindspore.nn.optim.adam import Adam, AdamWeightDecay


def build_optimizer(model, optim, lr, betas=[0.9, 0.999], weight_decay=1e-6):
    """

    :param model:
    :param optim:
    :param lr:
    :param betas:
    :param weight_decay:
    :return: optimizer
    """

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

    if optim == "adam":
        OptimCls = Adam
    elif optim == "adamw":
        OptimCls = AdamWeightDecay
    elif optim in ["sgd", "momentum"]:
        OptimCls = nn.Momentum
    else:
        raise ValueError("invalid optimizer")

    if optim in ["sgd", "momentum"]:
        optimizer = OptimCls(group_params, learning_rate=lr, momentum=0.9)
    else:
        optimizer = OptimCls(group_params, learning_rate=lr, beta1=betas[0], beta2=betas[1])

    return optimizer
