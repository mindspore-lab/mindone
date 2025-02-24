
import numpy as np

import mindspore as ms
from mindspore import nn, ops


@ms.jit
def pynative_x_to_dtype(x: ms.Tensor, dtype: ms.Type = ms.float32):
    return x.to(dtype)
