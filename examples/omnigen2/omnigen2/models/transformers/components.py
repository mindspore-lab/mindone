from typing import Sequence, Union

import numpy as np

import mindspore as ms
import mindspore.mint.nn.functional as F
from mindspore import Parameter, Tensor, nn, ops, tensor


def swiglu(x, y):
    return F.silu(x.float()).to(x.dtype) * y


class RMSNorm(nn.Cell):
    def __init__(
        self,
        hidden_size: Union[int, Sequence[int]],
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        dtype: ms.Type = ms.float32,
    ):
        if not elementwise_affine:
            raise NotImplementedError("RMSNorm does not support `elementwise_affine=False`")
        super().__init__()
        self.weight = Parameter(tensor(np.ones(hidden_size), dtype=dtype))
        self.variance_epsilon = eps
        self._dtype = dtype

    def construct(self, hidden_states: Tensor) -> Tensor:
        if self._dtype == ms.float16:  # for faster graph building
            return ops.rms_norm(
                hidden_states.to(ms.float32), self.weight.to(ms.float32), epsilon=self.variance_epsilon
            )[0].to(ms.float16)
        return ops.rms_norm(hidden_states, self.weight, epsilon=self.variance_epsilon)[0]
