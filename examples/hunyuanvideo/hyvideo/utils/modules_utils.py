import numbers
from typing import Tuple

import numpy as np

import mindspore as ms
from mindspore import Parameter, Tensor, mint, nn


class LayerNorm(nn.Cell):
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine: bool = True, bias=True, dtype=ms.float32):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        _weight = np.ones(normalized_shape, dtype=ms.dtype_to_nptype(dtype))
        _bias = np.zeros(normalized_shape, dtype=ms.dtype_to_nptype(dtype))
        if self.elementwise_affine:
            self.weight = Parameter(ms.Tensor.from_numpy(_weight), name="weight")
            if bias:
                self.bias = Parameter(ms.Tensor.from_numpy(_bias), name="bias")
            else:
                self.bias = ms.Tensor.from_numpy(_bias)
        else:
            self.weight = ms.Tensor.from_numpy(_weight)
            self.bias = ms.Tensor.from_numpy(_bias)

    def construct(self, x: Tensor):
        x = mint.nn.functional.layer_norm(
            x, self.normalized_shape, self.weight.to(x.dtype), self.bias.to(x.dtype), self.eps
        )
        return x
