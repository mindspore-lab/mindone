from typing import Any

from mindspore import Parameter, Tensor
from mindspore.common.initializer import (
    Constant,
    Normal,
    One,
    TruncatedNormal,
    XavierNormal,
    XavierUniform,
    Zero,
    initializer,
)


def exists(val: Any) -> bool:
    return val is not None


def default(val: Any, d: Any) -> Any:
    if exists(val):
        return val

    if isinstance(d, (Tensor, int, float)):
        return d
    return d()


def normal_(tensor: Parameter, mean: float = 0.0, std: float = 1.0) -> None:
    tensor.set_data(initializer(Normal(std, mean), tensor.shape, tensor.dtype))


def trunc_normal_(tensor: Parameter, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0) -> None:
    tensor.set_data(initializer(TruncatedNormal(std, mean, a, b), tensor.shape, tensor.dtype))


def constant_(tensor: Parameter, val: float) -> None:
    tensor.set_data(initializer(Constant(val), tensor.shape, tensor.dtype))


def ones_(tensor: Parameter) -> None:
    tensor.set_data(initializer(One(), tensor.shape, tensor.dtype))


def zeros_(tensor: Parameter) -> None:
    tensor.set_data(initializer(Zero(), tensor.shape, tensor.dtype))


def xavier_uniform_(tensor: Parameter, gain: float = 1.0) -> None:
    tensor.set_data(initializer(XavierUniform(gain), tensor.shape, tensor.dtype))


def xavier_normal_(tensor: Parameter, gain: float = 1.0) -> None:
    tensor.set_data(initializer(XavierNormal(gain), tensor.shape, tensor.dtype))


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
