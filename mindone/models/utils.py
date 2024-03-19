from typing import Any

from mindspore import Tensor
from mindspore.common.initializer import Constant, Normal, One, XavierNormal, XavierUniform, Zero, initializer


def exists(val: Any) -> bool:
    return val is not None


def default(val: Any, d: Any) -> Any:
    if exists(val):
        return val

    if isinstance(d, (Tensor, int, float)):
        return d
    return d()


def normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0) -> None:
    tensor.set_data(initializer(Normal(std, mean), tensor.shape, tensor.dtype))


def constant_(tensor: Tensor, val: float) -> None:
    tensor.set_data(initializer(Constant(val), tensor.shape, tensor.dtype))


def ones_(tensor: Tensor) -> None:
    tensor.set_data(initializer(One(), tensor.shape, tensor.dtype))


def zeros_(tensor: Tensor) -> None:
    tensor.set_data(initializer(Zero(), tensor.shape, tensor.dtype))


def xavier_uniform_(tensor: Tensor, gain: float = 1.0) -> None:
    tensor.set_data(initializer(XavierUniform(gain), tensor.shape, tensor.dtype))


def xavier_normal_(tensor: Tensor, gain: float = 1.0) -> None:
    tensor.set_data(initializer(XavierNormal(gain), tensor.shape, tensor.dtype))


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
