from typing import Optional

from mindspore import mint, nn

from ..nn.vo_ops import build_kwargs_from_config


__all__ = ["build_act"]


# register activation function here
REGISTERED_ACT_DICT: dict[str, type] = {
    "relu": mint.nn.ReLU,
    "relu6": mint.nn.ReLU6,
    "hswish": mint.nn.Hardswish,
    "silu": mint.nn.SiLU,
    "gelu": nn.GELU,
}


def build_act(name: str, **kwargs) -> Optional[nn.Cell]:
    if name in REGISTERED_ACT_DICT:
        act_cls = REGISTERED_ACT_DICT[name]
        args = build_kwargs_from_config(kwargs, act_cls)
        return act_cls(**args)
    else:
        return None
