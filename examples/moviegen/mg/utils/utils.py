import numpy as np

from mindspore import Tensor
from mindspore import dtype as mstype

__all__ = ["to_numpy"]


def to_numpy(x: Tensor) -> np.ndarray:
    if x.dtype == mstype.bfloat16:
        x = x.astype(mstype.float32)
    return x.asnumpy()
