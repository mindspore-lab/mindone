from typing import TYPE_CHECKING

import mindspore as ms
from mindspore import ops

if TYPE_CHECKING:
    from mindspore import Tensor


__all__ = ["unflatten", "view_as_complex"]


# ================================================================================
# unflatten
# ================================================================================
def _unflatten(x: "Tensor", dim: int, sizes: tuple[int, ...]) -> "Tensor":
    """
    Equivalence of `torch.unflatten`

    Args:
        x (ms.Tensor): The input tensor to unflatten.
        dim (int): The dimension to unflatten.
        sizes (tuple[int]): The target shape for the specified dimension.

    Returns:
        Tensor: A tensor with the specified dimension unflattened into the target shape.

    Raises:
        ValueError: If the specified dimension is out of range or if the product
                    of sizes does not match the size of the given dimension.
    """
    shape = x.shape

    dim = dim if dim >= 0 else dim + x.ndim

    # check validation of dim
    if dim < 0 or dim >= len(shape):
        raise ValueError(f"Invalid dimension {dim} for tensor with shape {x.shape}")

    # Calculate the product of sizes, excluding -1
    sizes_prod = 1
    num_unknown = 0
    for size in sizes:
        if size == -1:
            num_unknown += 1
        else:
            sizes_prod *= size

    # If there is one unknown size, calculate it
    if num_unknown == 1:
        sizes = tuple(size if size != -1 else shape[dim] // sizes_prod for size in sizes)

    new_shape = shape[:dim] + sizes + shape[dim + 1 :]

    return x.reshape(new_shape)


unflatten = _unflatten


# ================================================================================
# view_as_complex
# ================================================================================
def _view_as_complex(input: ms.Tensor) -> ms.Tensor:
    r"""
    Equivalence of `torch.view_as_complex`.

    Args:
        input (ms.Tensor): the input tensor.

    Example:

        >>> import mindspore as ms
        >>> x = ms.ops.randn(4, 2)
        >>> x
        [[ 1.6116, -0.5772]
         [-1.4606, -0.9120]
         [ 0.0786, -1.7497]
         [-0.6561, -1.6623]]
        >>> view_as_complex(x)
        [1.6116-0.5772j   -1.4606-0.9120j   0.0786-1.7497j   -0.6561-1.6623j]
    """
    assert input.shape[-1] == 2, "Tensor must have a last dimension of size 2"
    real_part, imag_part = input.chunk(2, dim=-1)
    # todo: unavailable mint interface ops.Complex
    output = ops.Complex()(real_part, imag_part).squeeze(axis=-1)
    return output


view_as_complex = _view_as_complex
