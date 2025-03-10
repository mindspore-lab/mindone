import math
from numbers import Number
from typing import Optional

import mindspore.ops as ops
from mindspore import Tensor


def pad_along_axis(
    x: Tensor,
    value: Optional[Number] = None,
    multiplier: int = 512,
    axis: int = -1,
    shift: int = 0,
    padding_direction: str = "right",
) -> Tensor:
    if axis >= 0:
        raise ValueError("Input `axis` must be a negative number.")

    shape = x.shape
    max_value = math.ceil(shape[axis] / multiplier) * multiplier
    pad_num = max(max_value - shape[axis] + shift, 0)

    if pad_num == 0:
        return x

    padding = (0, pad_num) if padding_direction == "right" else (pad_num, 0)
    padding = (-axis - 1) * (0, 0) + padding
    x = ops.pad(x, padding=padding, value=value)
    return x
