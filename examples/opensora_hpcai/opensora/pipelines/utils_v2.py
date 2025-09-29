from typing import Union

import numpy as np

from mindspore import Tensor


def time_shift(alpha: Union[float, Tensor], t: Union[np.ndarray, Tensor]) -> Union[np.ndarray, Tensor]:
    return alpha * t / (1 + (alpha - 1) * t)


def get_res_lin_function(x1: float = 256, y1: float = 1, x2: float = 4096, y2: float = 3) -> callable:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b
