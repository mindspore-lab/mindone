from hyvideo.utils.modules_utils import LayerNorm

import mindspore as ms
from mindspore import nn, ops


# from typing import Tuple
# import numpy as np
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
        # factory_kwargs = {"dtype": dtype}
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
        return LayerNorm
    elif norm_layer == "rms":
        return RMSNorm
    else:
        raise NotImplementedError(f"Norm layer {norm_layer} is not implemented")
