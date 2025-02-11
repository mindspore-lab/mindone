from dataclasses import dataclass
from typing import Optional

import mindspore as ms
from mindspore import Tensor, mint


class RaySegmentsSpec:
    def __init__(self):
        self.vals: Tensor  # [n_edges] or [n_rays, n_edges_per_ray]
        # for flattened
        self.chunk_starts: Tensor  # [n_rays]
        self.chunk_cnts: Tensor  # [n_rays]
        self.ray_indices: Tensor  # [n_edges]
        self.is_left: Tensor  # [n_edges] have n_bins true values
        self.is_right: Tensor  # [n_edges] have n_bins true values

    def memalloc_cnts(self, n_rays: Tensor, zero_init: bool = True):
        if zero_init:
            self.chunk_cnts = mint.zeros(n_rays, dtype=ms.int32)
        else:
            self.chunk_cnts = ms.Tensor().Tensor(data_type=ms.int32, shape=n_rays)


class PackedRaySegmentsSpec(RaySegmentsSpec):
    def __init__(self):
        super().__init__()
        self.is_batched: bool
        self.is_valid: bool
        self.n_edges: Tensor
        self.n_rays: Tensor
        self.n_edges_per_ray: Tensor


@dataclass
class RaySamples:
    """Ray samples that supports batched and flattened data.

    Note:
        When `vals` is flattened, either `packed_info` or `ray_indices` must
        be provided.

    Args:
        vals: Batched data with shape (n_rays, n_samples) or flattened data
            with shape (all_samples,)
        packed_info: Optional. A tensor of shape (n_rays, 2) that specifies
            the start and count of each chunk in flattened `vals`, with in
            total n_rays chunks. Only needed when `vals` is flattened.
        ray_indices: Optional. A tensor of shape (all_samples,) that specifies
            the ray index of each sample. Only needed when `vals` is flattened.

    Examples:

    .. code-block:: python

        >>> # Batched data
        >>> ray_samples = RaySamples(mint.rand(10, 100))
        >>> # Flattened data
        >>> ray_samples = RaySamples(
        >>>     mint.rand(1000),
        >>>     packed_info=Tensor([[0, 100], [100, 200], [300, 700]]),
        >>> )

    """

    vals: Tensor
    packed_info: Optional[Tensor] = None
    ray_indices: Optional[Tensor] = None
    is_valid: Optional[Tensor] = None

    def _to(self):
        """
        Generate object to pass to C++
        """

        spec = RaySegmentsSpec()
        spec.vals = self.vals.contiguous()
        if self.packed_info is not None:
            spec.chunk_starts = self.packed_info[:, 0].contiguous()
        if self.chunk_cnts is not None:
            spec.chunk_cnts = self.packed_info[:, 1].contiguous()
        if self.ray_indices is not None:
            spec.ray_indices = self.ray_indices.contiguous()
        return spec

    @classmethod
    def _from(cls, spec: RaySegmentsSpec):
        """
        Wrap a spec obj into a ray interval.
        """
        if spec.chunk_starts is not None and spec.chunk_cnts is not None:
            packed_info = mint.stack([spec.chunk_starts, spec.chunk_cnts], -1)
        else:
            packed_info = None
        ray_indices = spec.ray_indices
        if spec.is_valid is not None:
            is_valid = spec.is_valid
        else:
            is_valid = None
        vals = spec.vals
        return cls(vals=vals, packed_info=packed_info, ray_indices=ray_indices, is_valid=is_valid)


@dataclass
class RayIntervals:
    """Ray intervals that supports batched and flattened data.

    Each interval is defined by two edges (left and right). The attribute `vals`
    stores the edges of all intervals along the rays. The attributes `is_left`
    and `is_right` are for indicating whether each edge is a left or right edge.
    This class unifies the representation of both continuous and non-continuous ray
    intervals.

    Note:
        When `vals` is flattened, either `packed_info` or `ray_indices` must
        be provided. Also both `is_left` and `is_right` must be provided.

    Args:
        vals: Batched data with shape (n_rays, n_edges) or flattened data
            with shape (all_edges,)
        packed_info: Optional. A tensor of shape (n_rays, 2) that specifies
            the start and count of each chunk in flattened `vals`, with in
            total n_rays chunks. Only needed when `vals` is flattened.
        ray_indices: Optional. A tensor of shape (all_edges,) that specifies
            the ray index of each edge. Only needed when `vals` is flattened.
        is_left: Optional. A boolen tensor of shape (all_edges,) that specifies
            whether each edge is a left edge. Only needed when `vals` is flattened.
        is_right: Optional. A boolen tensor of shape (all_edges,) that specifies
            whether each edge is a right edge. Only needed when `vals` is flattened.

    Examples:

    .. code-block:: python

        >>> # Batched data
        >>> ray_intervals = RayIntervals(mint.rand(10, 100))
        >>> # Flattened data
        >>> ray_intervals = RayIntervals(
        >>>     mint.rand(6),
        >>>     packed_info=Tensor([[0, 2], [2, 0], [2, 4]]),
        >>>     is_left=Tensor([True, False, True, True, True, False]),
        >>>     is_right=Tensor([False, True, False, True, True, True]),
        >>> )

    """

    vals: Tensor
    packed_info: Optional[Tensor] = None
    ray_indices: Optional[Tensor] = None
    is_left: Optional[Tensor] = None
    is_right: Optional[Tensor] = None

    def _to(self):
        """
        Generate object to pass to C++
        """

        spec = RaySegmentsSpec()
        spec.vals = self.vals.contiguous()
        if self.packed_info is not None:
            spec.chunk_starts = self.packed_info[:, 0].contiguous()
        if self.packed_info is not None:
            spec.chunk_cnts = self.packed_info[:, 1].contiguous()
        if self.ray_indices is not None:
            spec.ray_indices = self.ray_indices.contiguous()
        if self.is_left is not None:
            spec.is_left = self.is_left.contiguous()
        if self.is_right is not None:
            spec.is_right = self.is_right.contiguous()
        return spec

    @classmethod
    def _from(cls, spec: RaySegmentsSpec):
        """
        Wrap a spec obj into a ray interval.
        """
        if spec.chunk_starts is not None and spec.chunk_cnts is not None:
            packed_info = mint.stack([spec.chunk_starts, spec.chunk_cnts], -1)
        else:
            packed_info = None
        ray_indices = spec.ray_indices
        is_left = spec.is_left
        is_right = spec.is_right
        return cls(
            vals=spec.vals,
            packed_info=packed_info,
            ray_indices=ray_indices,
            is_left=is_left,
            is_right=is_right,
        )
