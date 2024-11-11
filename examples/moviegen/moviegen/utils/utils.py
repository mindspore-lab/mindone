import numpy as np

import mindspore as ms

__all__ = ["to_numpy"]


def to_numpy(x: ms.Tensor) -> np.ndarray:
    if x.dtype == ms.bfloat16:
        x = x.astype(ms.float32)
    return x.asnumpy()
