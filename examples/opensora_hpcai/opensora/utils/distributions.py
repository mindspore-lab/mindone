import numpy as np

from mindspore import Tensor, dtype, ops


class LogisticNormal:
    def __init__(self, loc, scale):
        self.stdnormal = ops.StandardNormal()
        self.mean = loc
        self.std = scale
        self._min = Tensor(np.finfo(np.float32).tiny, dtype=dtype.float32)
        self._max = Tensor(1.0 - np.finfo(np.float32).eps, dtype=dtype.float32)

    def sample(self, shape):
        x = self.mean + self.std * self.stdnormal(shape)
        z = ops.clamp(ops.sigmoid(x), self._min, self._max)
        return ops.pad(z, [0, 1], value=1) * ops.pad(1 - z, [1, 0], value=1)
