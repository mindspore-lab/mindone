from typing import Optional

import mindspore as ms
from mindspore import Tensor, mint


@ms._no_grad()
def pack_info(ray_indices: Tensor, n_rays: Optional[int] = None) -> Tensor:
    """Pack `ray_indices` to `packed_info`. Useful for converting per sample data to per ray data.

    Note:
        this function is not differentiable to any inputs.

    Args:
        ray_indices: Ray indices of the samples. LongTensor with shape (n_sample).
        n_rays: Number of rays. If None, it is inferred from `ray_indices`. Default is None.

    Returns:
        A LongTensor of shape (n_rays, 2) that specifies the start and count
        of each chunk in the flattened input tensor, with in total n_rays chunks.

    Example:

    .. code-block:: python

        >>> ray_indices = ms.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2])
        >>> packed_info = pack_info(ray_indices, n_rays=3)
        >>> packed_info
        tensor([[0, 2], [2, 3], [5, 4]])

    """
    assert ray_indices.dim() == 1, "ray_indices must be a 1D tensor with shape (n_samples)."
    dtype = ray_indices.dtype
    if n_rays is None:
        n_rays = ray_indices.max().item() + 1
    chunk_cnts = mint.zeros((n_rays,), dtype=dtype)
    chunk_cnts.index_add(0, ray_indices, mint.ones_like(ray_indices))
    chunk_starts = chunk_cnts.cumsum(axis=0, dtype=dtype) - chunk_cnts
    packed_info = mint.stack([chunk_starts, chunk_cnts], dim=-1)
    return packed_info
