import numbers
from typing import List, Literal, Tuple, Union

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, Tensor
from mindspore.common.initializer import Initializer, initializer


class LayerNorm(nn.LayerNorm):
    def __init__(
        self,
        normalized_shape: Union[int, Tuple[int], List[int]],
        begin_norm_axis: int = -1,
        begin_params_axis: int = -1,
        gamma_init: Union[Tensor, str, Initializer, numbers.Number] = "ones",
        beta_init: Union[Tensor, str, Initializer, numbers.Number] = "zeros",
        epsilon: float = 1e-7,
        elementwise_affine: bool = False,
        dtype: ms.dtype = ms.float32,
    ):
        """Initialize LayerNorm."""
        super(nn.LayerNorm, self).__init__()
        if not isinstance(normalized_shape, (int, tuple, list)):
            raise TypeError(
                f"For '{self.cls_name}', the type of 'normalized_shape' must be int, tuple[int] or list[int], "
                f"but got {normalized_shape} and the type is {type(normalized_shape)}."
            )
        if not normalized_shape:
            raise ValueError(
                f"Expected normalized_shape to be at least 1-dimensional, i.e., containing at "
                f"least one element, but got normalized_shape = {normalized_shape}"
            )
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else normalized_shape
        self.begin_norm_axis = begin_norm_axis
        self.begin_params_axis = begin_params_axis
        self.epsilon = epsilon
        if elementwise_affine:
            self.weight = Parameter(initializer(gamma_init, normalized_shape, dtype=dtype))
            self.bias = Parameter(initializer(beta_init, normalized_shape, dtype=dtype))
        else:
            self.weight = ops.ones(normalized_shape, dtype=dtype)
            self.bias = ops.zeros(normalized_shape, dtype=dtype)
        self.layer_norm = ops.LayerNorm(
            begin_norm_axis=self.begin_norm_axis, begin_params_axis=self.begin_params_axis, epsilon=self.epsilon
        )

    def construct(self, input_x):
        y, _, _ = self.layer_norm(input_x, self.weight.astype(input_x.dtype), self.bias.astype(input_x.dtype))
        return y

    def extend_repr(self):
        return "normalized_shape={}, begin_norm_axis={}, begin_params_axis={}, gamma{}, beta={}".format(
            self.normalized_shape, self.begin_norm_axis, self.begin_params_axis, self.weight, self.bias
        )


class GELU(nn.GELU):
    def __init__(self, approximate: Literal["tanh", "none"] = "tanh"):
        if approximate == "none":
            super().__init__(approximate=False)
        else:
            super().__init__(approximate=True)
