import numbers
import numpy as np
from typing import Dict, Optional, Tuple
import mindspore as ms
from mindspore.common.initializer import initializer
from mindspore import Parameter, Tensor, nn, ops, mint


class LayerNorm(nn.Cell):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine: bool = True, bias=True, dtype=ms.float32):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(initializer("ones", normalized_shape, dtype=dtype))
            if bias: 
                self.bias = Parameter(initializer("zeros", normalized_shape, dtype=dtype))
            else:
                self.bias = ops.zeros(normalized_shape, dtype=dtype)
        else:
            self.weight = ops.ones(normalized_shape, dtype=dtype)
            self.bias = ops.zeros(normalized_shape, dtype=dtype)

    def construct(self, x: Tensor):
        normalized_shape = x.shape[-1:]
        # mint layer_norm fuses the operations in layer normorlization and it's faster than ops.LayerNorm
        x = mint.nn.functional.layer_norm(x, normalized_shape, self.weight, self.bias, self.eps)

        return x


class FP32LayerNorm(LayerNorm):
    def construct(self, x: Tensor):
        origin_dtype = x.dtype
        normalized_shape = x.shape[-1:]
        # mint layer_norm fuses the operations in layer normorlization and it's faster than ops.LayerNorm
        x = mint.nn.functional.layer_norm(x.float(), normalized_shape, self.weight.float(), self.bias.float(), self.eps)

        return x.to(origin_dtype)

'''
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
        # _weight = np.ones(normalized_shape, dtype=ms.dtype_to_nptype(dtype))
        # _bias = np.zeros(normalized_shape, dtype=ms.dtype_to_nptype(dtype))
        if self.elementwise_affine:
            self.weight = Parameter(ops.ones(normalized_shape, dtype=dtype), name="weight")
            if bias:
                # self.bias = Parameter(ms.Tensor.from_numpy(_bias), name="bias")
                self.bias = Parameter(ops.zeros(normalized_shape, dtype=dtype), name="bias")
            else:
                self.bias = ops.zeros(normalized_shape, dtype=dtype)
        else:
            self.weight = ops.ones(normalized_shape, dtype=dtype)
            self.bias = ops.zeros(normalized_shape, dtype=dtype)
        # TODO: In fact, we need -len(normalized_shape) instead of -1, but LayerNorm doesn't allow it.
        #  For positive axis, the ndim of input is needed. Put it in construct?
        self.layer_norm = ops.LayerNorm(-1, -1, epsilon=eps)

    def construct(self, x: Tensor):
        # TODO: use minit layernorm for better speed
        # AMP: sum fp32
        x, _, _ = self.layer_norm(x, self.weight.to(x.dtype), self.bias.to(x.dtype))
        return x

class FP32LayerNorm(LayerNorm):
    def construct(self, inputs: ms.Tensor) -> ms.Tensor:
        origin_dtype = inputs.dtype
        x, _, _ = self.layer_norm(
            inputs.float(),
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
        )
        return x.to(origin_dtype)

'''

class RMSNorm(nn.Cell):
    def __init__(
        self,
        dim: int,
        elementwise_affine=True,
        eps: float = 1e-6,
        dtype=None,
    ):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        factory_kwargs = {"dtype": dtype}
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = ms.Parameter(ops.ones(dim, dtype=dtype), name="weight")
        else:
            # gamma
            self.weight = None

    def _norm(self, x):
        variance = x.pow(2).mean(-1, keep_dims=True)

        return x * ops.rsqrt(variance + self.eps)

    def construct(self, x):
        input_dtype = x.dtype
        # AMP: pt also cast x to float32 for rmsnorm
        output = self._norm(x.float())
        if self.weight is not None:
            output = output.to(self.weight.dtype) * self.weight
        else:
            output = output.to(input_dtype)
        return output


def get_norm_layer(norm_layer):
    """
    Get the normalization layer.

    Args:
        norm_layer (str): The type of normalization layer.

    Returns:
        norm_layer (nn.Module): The normalization layer.
    """
    if norm_layer == "layer":
        # return LayerNorm
        return FP32LayerNorm
    elif norm_layer == "rms":
        return RMSNorm
    else:
        raise NotImplementedError(f"Norm layer {norm_layer} is not implemented")
