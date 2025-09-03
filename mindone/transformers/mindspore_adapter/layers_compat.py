from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mindspore import Tensor


__all__ = ["unflatten"]


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
