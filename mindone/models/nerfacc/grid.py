import time
from typing import Optional, Tuple, Type

import numpy as np

import mindspore as ms
from mindspore import Tensor, mint, ops

from .data_specs import RayIntervals, RaySamples


class RaySegmentsSpec:
    """Mindspore equivalent of the cuda RaySegmentsSpec struct."""

    def __init__(self):
        # [n_edges] or [n_rays, n_edges_per_ray]
        self.vals: Optional[Tensor] = None

        # common for intervals and samples
        self.ray_indices: Optional[Tensor] = None  # [n_edges]
        self.chunk_starts: Optional[Tensor] = None  # [n_rays]
        self.chunk_cnts: Optional[Tensor] = None  # [n_rays]
        # for intervals
        self.is_left: Optional[Tensor] = None  # [n_edges]
        self.is_right: Optional[Tensor] = None  # [n_edges]
        # for samples
        self.is_valid: Optional[Tensor] = None  # [n_edges]

    def check(self):
        """Validate tensor dimensions and shapes."""
        assert self.vals is not None, "vals must be defined"
        assert isinstance(self.vals, Tensor), "vals must be a Tensor"

        # batched tensor [..., n_edges_per_ray]
        if self.vals.ndim > 1:
            return

        # flattened tensor [n_edges]
        assert isinstance(self.chunk_starts, Tensor), "chunk_starts must be a Tensor"
        assert isinstance(self.chunk_cnts, Tensor), "chunk_cnts must be a Tensor"
        assert self.chunk_starts.ndim == 1, "chunk_starts must be 1D"
        assert self.chunk_cnts.ndim == 1, "chunk_cnts must be 1D"
        assert self.chunk_starts.numel() == self.chunk_cnts.numel()

        if self.ray_indices is not None:
            assert isinstance(self.ray_indices, Tensor)
            assert self.ray_indices.ndim == 1
            assert self.vals.numel() == self.ray_indices.numel()

        if self.is_left is not None:
            assert isinstance(self.is_left, Tensor)
            assert self.is_left.ndim == 1
            assert self.vals.numel() == self.is_left.numel()

        if self.is_right is not None:
            assert isinstance(self.is_right, Tensor)
            assert self.is_right.ndim == 1
            assert self.vals.numel() == self.is_right.numel()

        if self.is_valid is not None:
            assert isinstance(self.is_valid, Tensor)
            assert self.is_valid.ndim == 1
            assert self.vals.numel() == self.is_valid.numel()

    def memalloc_cnts(self, n_rays: int, options: ms.dtype, zero_init: bool = True):
        """Allocate memory for chunk counts."""
        assert self.chunk_cnts is None, "chunk_cnts already allocated"
        if zero_init:
            self.chunk_cnts = mint.zeros(n_rays, dtype=ms.int32)

    def memalloc_data(self, size: int, alloc_masks: bool = True, zero_init: bool = True, alloc_valid: bool = False):
        """Allocate memory for segment data."""
        assert self.chunk_cnts is not None, "chunk_cnts must be allocated first"
        assert self.vals is None, "vals already allocated"

        self.vals = mint.zeros(size, dtype=ms.float32)
        self.ray_indices = mint.zeros(size, dtype=ms.int32)
        if alloc_masks:
            self.is_left = mint.zeros(size, dtype=ms.bool_)
            self.is_right = mint.zeros(size, dtype=ms.bool_)

        if alloc_valid:
            self.is_valid = mint.zeros(size, dtype=ms.bool_)

    def memalloc_data_from_chunk(
        self, alloc_masks: bool = True, zero_init: bool = True, alloc_valid: bool = False
    ) -> int:
        """Allocate memory based on chunk counts."""
        assert self.chunk_cnts is not None, "chunk_cnts must be allocated first"
        assert self.chunk_starts is None, "chunk_starts already allocated"

        cumsum = mint.cumsum(self.chunk_cnts, 0)
        n_edges = cumsum[-1].item()

        self.chunk_starts = cumsum - self.chunk_cnts
        self.memalloc_data(n_edges, alloc_masks, zero_init, alloc_valid)
        return 1

    def compute_chunk_start(self) -> int:
        """Compute chunk starts from chunk counts."""
        assert self.chunk_cnts is not None, "chunk_cnts must be allocated first"

        cumsum = mint.cumsum(self.chunk_cnts, 0)
        self.chunk_starts = cumsum - self.chunk_cnts
        return 1


@ms._no_grad()
def _ray_aabb_intersect(
    rays_o: Tensor,
    rays_d: Tensor,
    aabbs: Tensor,
    near_plane: float = -float("inf"),
    far_plane: float = float("inf"),
    miss_value: float = float("inf"),
) -> Tuple[Tensor, Tensor, Tensor]:
    """Ray-AABB intersection.

    Args:
        rays_o: (n_rays, 3) Ray origins.
        rays_d: (n_rays, 3) Normalized ray directions.
        aabbs: (m, 6) Axis-aligned bounding boxes {xmin, ymin, zmin, xmax, ymax, zmax}.
        near_plane: Optional. Near plane. Default to -infinity.
        far_plane: Optional. Far plane. Default to infinity.
        miss_value: Optional. Value to use for tmin and tmax when there is no intersection.
            Default to infinity.

    Returns:
        A tuple of {Tensor, Tensor, BoolTensor}:

        - **t_mins**: (n_rays, m) tmin for each ray-AABB pair.
        - **t_maxs**: (n_rays, m) tmax for each ray-AABB pair.
        - **hits**: (n_rays, m) whether each ray-AABB pair intersects.

    Functionally the same with `ray_aabb_intersect()`, but slower with pure mindspore.
    """

    # Compute the minimum and maximum bounds of the AABBs
    aabb_min = aabbs[:, :3]
    aabb_max = aabbs[:, 3:]

    # Compute the intersection distances between the ray and each of the six AABB planes
    t1 = (aabb_min[None, :, :] - rays_o[:, None, :]) / rays_d[:, None, :]
    t2 = (aabb_max[None, :, :] - rays_o[:, None, :]) / rays_d[:, None, :]

    # Compute the maximum tmin and minimum tmax for each AABB
    t_mins = mint.max(mint.minimum(t1, t2), dim=-1)[0]
    t_maxs = mint.min(mint.maximum(t1, t2), dim=-1)[0]

    # Compute whether each ray-AABB pair intersects
    hits = mint.logical_and(t_maxs > t_mins, t_maxs > 0)

    # Clip the tmin and tmax values to the near and far planes
    t_mins = mint.clamp(t_mins, min=near_plane, max=far_plane)
    t_maxs = mint.clamp(t_maxs, min=near_plane, max=far_plane)

    # Set the tmin and tmax values to miss_value if there is no intersection
    t_mins = mint.where(hits, t_mins, miss_value)
    t_maxs = mint.where(hits, t_maxs, miss_value)

    return t_mins, t_maxs, hits


@ms._no_grad()
def traverse_grids(
    # rays
    rays_o: Tensor,  # [n_rays, 3]
    rays_d: Tensor,  # [n_rays, 3]
    # grids
    binaries: Tensor,  # [m, resx, resy, resz]
    aabbs: Tensor,  # [m, 6]
    # options
    near_planes: Optional[Tensor] = None,  # [n_rays]
    far_planes: Optional[Tensor] = None,  # [n_rays]
    step_size: Optional[float] = 1e-3,
    cone_angle: Optional[float] = 0.0,
    traverse_steps_limit: Optional[int] = None,
    over_allocate: Optional[bool] = False,
    rays_mask: Optional[Tensor] = None,  # [n_rays]
    # pre-compute intersections
    t_sorted: Optional[Tensor] = None,  # [n_rays, n_grids * 2]
    t_indices: Optional[Tensor] = None,  # [n_rays, n_grids * 2]
    hits: Optional[Tensor] = None,  # [n_rays, n_grids]
) -> Tuple[RayIntervals, RaySamples, Tensor]:
    """Ray Traversal within Multiple Grids.

    Note:
        This function is not differentiable to any inputs.

    Args:
        rays_o: (n_rays, 3) Ray origins.
        rays_d: (n_rays, 3) Normalized ray directions.
        binary_grids: (m, resx, resy, resz) Multiple binary grids with the same resolution.
        aabbs: (m, 6) Axis-aligned bounding boxes {xmin, ymin, zmin, xmax, ymax, zmax}.
        near_planes: Optional. (n_rays,) Near planes for the traversal to start. Default to 0.
        far_planes: Optional. (n_rays,) Far planes for the traversal to end. Default to infinity.
        step_size: Optional. Step size for ray traversal. Default to 1e-3.
        cone_angle: Optional. Cone angle for linearly-increased step size. 0. means
            constant step size. Default: 0.0.
        traverse_steps_limit: Optional. Maximum number of samples per ray.
        over_allocate: Optional. Whether to over-allocate the memory for the outputs.
        rays_mask: Optional. (n_rays,) Skip some rays if given.
        t_sorted: Optional. (n_rays, n_grids * 2) Pre-computed sorted t values for each ray-grid pair. Default to None.
        t_indices: Optional. (n_rays, n_grids * 2) Pre-computed sorted t indices for each ray-grid pair. Default to None.
        hits: Optional. (n_rays, n_grids) Pre-computed hit flags for each ray-grid pair. Default to None.

    Returns:
        A :class:`RayIntervals` object containing the intervals of the ray traversal, and
        a :class:`RaySamples` object containing the samples within each interval.
        t :class:`Tensor` of shape (n_rays,) containing the terminated t values for each ray.
    """

    if near_planes is None:
        near_planes = mint.zeros_like(rays_o[:, 0])
    if far_planes is None:
        far_planes = ops.full_like(rays_o[:, 0], float("inf"))

    if rays_mask is None:
        rays_mask = mint.ones_like(rays_o[:, 0], dtype=ms.bool_)
    if traverse_steps_limit is None:
        traverse_steps_limit = -1
    if over_allocate:
        assert traverse_steps_limit > 0, "traverse_steps_limit must be set if over_allocate is True."

    if t_sorted is None or t_indices is None or hits is None:
        # Compute ray aabb intersection for all levels of grid. [n_rays, m]
        t_mins, t_maxs, hits = _ray_aabb_intersect(rays_o, rays_d, aabbs)
        # Sort the t values for each ray. [n_rays, m]
        t_sorted, t_indices = mint.sort(mint.cat([t_mins, t_maxs], dim=-1), dim=-1)

    # Traverse the grids. Original in as cpp extensions, now as operators
    intervals, samples, termination_planes = _traverse_grids(
        # rays
        rays_o.contiguous(),  # [n_rays, 3]
        rays_d.contiguous(),  # [n_rays, 3]
        rays_mask.contiguous(),  # [n_rays]
        # grids
        binaries.contiguous(),  # [m, resx, resy, resz]
        aabbs.contiguous(),  # [m, 6]
        # intersections
        t_sorted.contiguous(),  # [n_rays, m * 2]
        t_indices.contiguous(),  # [n_rays, m * 2]
        hits.contiguous(),  # [n_rays, m]
        # options
        near_planes.contiguous(),  # [n_rays]
        far_planes.contiguous(),  # [n_rays]
        step_size,
        cone_angle,
        True,
        True,
        True,
        # traverse_steps_limit=4e3,  # FIXME overallocate for now seems faster, but needs to turn it back
        # over_allocate=True,
    )
    return (
        RayIntervals._from(intervals),
        RaySamples._from(samples),
        termination_planes,
    )


def _enlarge_aabb(aabb, factor: float) -> Tensor:
    center = (aabb[:3] + aabb[3:]) / 2
    extent = (aabb[3:] - aabb[:3]) / 2
    return mint.cat([center - extent * factor, center + extent * factor])


def _query(x: Tensor, data: Tensor, base_aabb: Tensor) -> Tensor:
    """
    Query the grid values at the given points.

    This function assumes the aabbs of multiple grids are 2x scaled.

    Args:
        x: (N, 3) tensor of points to query.
        data: (m, resx, resy, resz) tensor of grid values
        base_aabb: (6,) aabb of base level grid.
    """
    # normalize so that the base_aabb is [0, 1]^3
    aabb_min, aabb_max = mint.split(base_aabb, 3, dim=0)
    x_norm = (x - aabb_min) / (aabb_max - aabb_min)

    # if maxval is almost zero, it will trigger frexpf to output 0
    # for exponent, which is not what we want.
    maxval = (x_norm - 0.5).abs().max(dim=-1).values
    maxval = mint.clamp(maxval, min=0.1)

    # compute the mip level
    exponent = np.frexp(maxval)[1].astype(ms.int32)
    mip = mint.clamp(exponent + 1, min=0)
    selector = mip < data.shape[0]

    # use the mip to re-normalize all points to [0, 1].
    scale = 2**mip
    x_unit = (x_norm - 0.5) / scale[:, None] + 0.5

    # map to the grid index
    resolution = Tensor(data.shape[1])
    ix = (x_unit * resolution).astype(ms.int32)

    ix = mint.clamp(ix, max=resolution - 1)
    mip = mint.clamp(mip, max=data.shape[0] - 1)

    return data[mip, ix[:, 0], ix[:, 1], ix[:, 2]] * selector, selector


def _calc_dt(t: float, cone_angle: float, dt_min: float, dt_max: float) -> float:
    """Calculate step size based on distance from origin."""
    return mint.clamp(t * cone_angle, dt_min, dt_max)


def _traverse_rays(
    rays_o: Tensor,  # [n_rays, 3]
    rays_d: Tensor,  # [n_rays, 3]
    rays_mask: Optional[Tensor],  # [n_rays]
    binaries: Tensor,  # [n_grids, resx, resy, resz]
    aabbs: Tensor,  # [n_grids, 6]
    resolution: Tuple[int, int, int],
    hits: Tensor,  # [n_rays, n_grids]
    t_sorted: Tensor,  # [n_rays, n_grids * 2]
    t_indices: Tensor,  # [n_rays, n_grids * 2]
    near_planes: Tensor,  # [n_rays]
    far_planes: Tensor,  # [n_rays]
    step_size: float,
    cone_angle: float,
    traverse_steps_limit: int,
    intervals: RaySegmentsSpec,
    samples: RaySegmentsSpec,
    terminate_planes: Optional[Tensor],
    first_pass: bool,
) -> None:
    """Serial implementation of traverse_grids_kernel."""
    eps = 1e-6
    n_rays = rays_o.shape[0]
    n_grids = binaries.shape[0]

    # Process each ray sequentially
    for ray_id in range(n_rays):
        start_time = time.time()

        if rays_mask is not None and not rays_mask[ray_id]:
            continue

        # Skip empty rays in second pass
        if not first_pass:
            if (intervals.chunk_cnts is not None and intervals.chunk_cnts[ray_id] == 0) or (
                samples.chunk_cnts is not None and samples.chunk_cnts[ray_id] == 0
            ):
                continue

        # Get chunk starts for second pass
        if not first_pass:
            chunk_start = intervals.chunk_starts[ray_id] if intervals.chunk_cnts is not None else None
            chunk_start_bin = samples.chunk_starts[ray_id] if samples.chunk_cnts is not None else None

        near_plane = near_planes[ray_id]
        far_plane = far_planes[ray_id]
        ray_o = rays_o[ray_id]
        ray_d = rays_d[ray_id]

        # Track intervals and samples for this ray
        n_intervals = 0
        n_samples = 0
        t_last = near_plane
        continuous = False

        # Loop over all intersections along the ray
        # no need to have base offset as doing 2D looping now
        base_hits = 0

        for i in range(n_grids * 2 - 1):
            # Check if entering or leaving grid
            is_entering = t_indices[ray_id, i] < n_grids
            level = t_indices[ray_id, i] % n_grids

            if not hits[ray_id, base_hits + level]:
                continue  # Grid not hit

            if not is_entering:
                # Check if we're entering next grid
                next_is_entering = t_indices[ray_id, i + 1] < n_grids
                if next_is_entering:
                    continue
                level = t_indices[ray_id, i + 1] % n_grids
                if not hits[ray_id, base_hits + level]:
                    continue

            # Calculate intersection interval
            this_tmin = mint.maximum(t_sorted[ray_id, i], near_plane)
            this_tmax = mint.minimum(t_sorted[ray_id, i + 1], far_plane)
            if this_tmin >= this_tmax:
                continue

            if not continuous:
                if step_size <= 0.0:
                    t_last = this_tmin
                else:
                    while True:
                        dt = _calc_dt(t_last, cone_angle, step_size, 1e10)
                        if t_last + dt * 0.5 >= this_tmin:
                            break
                        t_last += dt

            # Get AABB for this grid level
            aabb = aabbs[level]

            # Setup traversal variables
            tdist, delta, step_index, current_index, final_index = setup_traversal(
                ray_o, ray_d, this_tmin, this_tmax, eps, aabb, resolution
            )

            overflow_index = final_index + step_index

            # Traverse grid cells
            while traverse_steps_limit <= 0 or n_samples < traverse_steps_limit:
                t_traverse = min(tdist[0], min(tdist[1], tdist[2]))
                t_traverse = min(t_traverse, this_tmax)

                # Calculate cell index
                cell_id = (
                    current_index[0] * resolution[1] * resolution[2]
                    + current_index[1] * resolution[2]
                    + current_index[2]
                    + level * resolution[0] * resolution[1] * resolution[2]
                )

                if not binaries.flatten()[cell_id]:
                    # Skip empty cells
                    if step_size <= 0.0:
                        t_last = t_traverse
                    else:
                        while True:
                            dt = _calc_dt(t_last, cone_angle, step_size, 1e10)
                            if t_last + dt * 0.5 >= t_traverse:
                                break
                            t_last += dt
                    continuous = False
                else:
                    # Traverse occupied cell
                    while traverse_steps_limit <= 0 or n_samples < traverse_steps_limit:
                        if step_size <= 0.0:
                            t_next = t_traverse
                        else:
                            dt = _calc_dt(t_last, cone_angle, step_size, 1e10)
                            if t_last + dt * 0.5 >= t_traverse:
                                break  # march until t_mid is right after t_traverse.
                            t_next = t_last + dt

                        # Record intervals
                        if intervals.chunk_cnts is not None:
                            if not continuous:
                                if not first_pass:
                                    idx = chunk_start + n_intervals
                                    intervals.vals[idx] = t_last
                                    intervals.ray_indices[idx] = ray_id
                                    intervals.is_left[idx] = True
                                n_intervals += 1
                                if not first_pass:
                                    idx = chunk_start + n_intervals
                                    intervals.vals[idx] = t_next
                                    intervals.ray_indices[idx] = ray_id
                                    intervals.is_right[idx] = True
                                n_intervals += 1
                            else:
                                if (
                                    not first_pass
                                ):  # right side of the intervel, as it's continuous and the left side has been done?
                                    idx = chunk_start + n_intervals
                                    intervals.vals[idx] = t_next
                                    intervals.ray_indices[idx] = ray_id
                                    intervals.is_left[idx - 1] = True
                                    # intervals.is_right[idx - 1] = False
                                    intervals.is_right[idx] = True
                                n_intervals += 1

                        # Record samples
                        if samples.chunk_cnts is not None:
                            if not first_pass:
                                idx = chunk_start_bin + n_samples
                                samples.vals[idx] = (t_next + t_last) * 0.5
                                samples.ray_indices[idx] = ray_id
                                samples.is_valid[idx] = True

                        n_samples += 1
                        continuous = True
                        t_last = t_next
                        if t_next >= t_traverse:
                            break  # next travel reaches end

                if not single_traversal(tdist, current_index, overflow_index, step_index, delta):
                    break

        # Record final results
        if terminate_planes is not None:
            terminate_planes[ray_id] = t_last

        if intervals.chunk_cnts is not None:
            intervals.chunk_cnts[ray_id] = n_intervals
        if samples.chunk_cnts is not None:
            samples.chunk_cnts[ray_id] = n_samples

        ray_batch_time = time.time() - start_time
        print(f"ray batch time cost: {ray_batch_time:.3f}s.")


def setup_traversal(
    ray_o: Tensor,  # [3]
    ray_d: Tensor,  # [3]
    t_min: float,
    t_max: float,
    eps: float,
    aabb: Tensor,  # [6]
    resolution: Tuple[int, int, int],
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Setup variables needed for grid traversal."""
    # Calculate entry point
    ray_start = ray_o + ray_d * (t_min + eps)

    # Calculate cell size
    cell_size = (aabb[3:] - aabb[:3]) / Tensor(resolution)

    # Calculate initial cell indices
    current_index = mint.floor((ray_start - aabb[:3]) / cell_size).astype(ms.int32)
    current_index = mint.clamp(current_index, 0, Tensor(resolution)[0].item() - 1)

    # Calculate step direction
    step_index = mint.sign(ray_d).astype(ms.int32)

    # Calculate distance between cell boundaries
    delta = mint.abs(cell_size / (ray_d + eps))

    # Calculate distance to next cell boundary
    next_boundary = (current_index + mint.maximum(step_index, mint.zeros_like(step_index))) * cell_size + aabb[:3]
    tdist = (next_boundary - ray_start) / (ray_d + eps)

    # Calculate final cell index
    ray_end = ray_o + ray_d * (t_max - eps)
    final_index = mint.floor((ray_end - aabb[:3]) / cell_size).astype(ms.int32)
    final_index = mint.clamp(final_index, 0, Tensor(resolution)[0].item() - 1)

    return tdist, delta, step_index, current_index, final_index


def single_traversal(
    tdist: Tensor,  # [3]
    current_index: Tensor,  # [3]
    overflow_index: Tensor,  # [3]
    step_index: Tensor,  # [3]
    delta: Tensor,  # [3]
) -> bool:
    """Advance one step in grid traversal."""
    # Find minimum distance
    min_dist, min_axis = mint.min(tdist, dim=0)

    # Update current index
    current_index[min_axis] += step_index[min_axis]

    # Check if we've gone past the final index
    if current_index[min_axis] == overflow_index[min_axis]:
        return False

    # Update tdist
    tdist[min_axis] += delta[min_axis]

    return True


def _traverse_grids(
    # rays
    rays_o: Tensor,  # [n_rays, 3]
    rays_d: Tensor,  # [n_rays, 3]
    rays_mask: Tensor,  # [n_rays]
    # grids
    binaries: Tensor,  # [n_grids, resx, resy, resz]
    aabbs: Tensor,  # [n_grids, 6]
    # intersections
    t_sorted: Tensor,  # [n_rays, n_grids * 2]
    t_indices: Tensor,  # [n_rays, n_grids * 2]
    hits: Tensor,  # [n_rays, n_grids]
    # options
    near_planes: Tensor,  # [n_rays]
    far_planes: Tensor,  # [n_rays]
    step_size: float,
    cone_angle: float,
    compute_intervals: bool,
    compute_samples: bool,
    compute_terminate_planes: bool,
    traverse_steps_limit: int = 0,  # <= 0 means no limit
    over_allocate: bool = False,
) -> Tuple[Type[RaySegmentsSpec], Type[RaySegmentsSpec], Optional[Tensor]]:
    """Serial ms implementation of grid traversal.

    Key differences from CUDA version:
    1. Uses pure ms operations instead of CUDA kernels
    2. Processes rays sequentially rather than in parallel
    3. Uses Python classes/datastructures instead of C++ structs

    Traverses grids to compute ray intersections and samples. Adapts form the cuda kernel func. to ms serial.

    Args:
        rays_o: Tensor [n_rays, 3], ray origins.
        rays_d: Tensor [n_rays, 3], ray directions.
        binaries: Tensor [n_grids, resx, resy, resz], binary occupancy grids.
        aabbs: Tensor [n_grids, 6], AABBs for each grid.
        t_mins: Tensor [n_rays, n_grids], min intersection distances per ray and grid.
        t_maxs: Tensor [n_rays, n_grids], max intersection distances per ray and grid.
        hits: Tensor [n_rays, n_grids], whether each ray hits each grid.
        near_planes: Tensor [n_rays], near clipping planes.
        far_planes: Tensor [n_rays], far clipping planes.
        step_size: Scalar, step size for marching.
        cone_angle: Scalar, cone angle for adaptive step size.
        compute_intervals: Whether to compute intervals.
        compute_samples: Whether to compute samples.

    Returns:
        intervals: Dict containing interval data (if compute_intervals=True).
        samples: Dict containing sample data (if compute_samples=True).
    """
    n_rays = rays_o.shape[0]
    # n_grids = binaries.shape[0]
    resolution = binaries.shape[1:]  # (resx, resy, resz)

    # Initialize outputs
    intervals = RaySegmentsSpec()
    samples = RaySegmentsSpec()
    terminate_planes = mint.zeros(n_rays) if compute_terminate_planes else None

    if over_allocate:
        assert traverse_steps_limit > 0, "traverse_steps_limit must be > 0 when over_allocate is true"
        # Pre-allocate maximum possible size
        if compute_intervals:
            intervals.chunk_cnts = mint.full((n_rays,), traverse_steps_limit * 2, dtype=ms.int32)
            if rays_mask is not None:
                intervals.chunk_cnts *= rays_mask
            intervals.memalloc_data_from_chunk(True, True)

        if compute_samples:
            samples.chunk_cnts = mint.full((n_rays,), traverse_steps_limit, dtype=ms.int32)
            if rays_mask is not None:
                samples.chunk_cnts *= rays_mask
            samples.memalloc_data_from_chunk(False, True, True)

        # Process rays
        _traverse_rays(
            rays_o,
            rays_d,
            rays_mask,
            binaries,
            aabbs,
            resolution,
            hits,
            t_sorted,
            t_indices,
            near_planes,
            far_planes,
            step_size,
            cone_angle,
            traverse_steps_limit,
            intervals,
            samples,
            terminate_planes,
            first_pass=False,
        )

        # Update chunk starts
        intervals.compute_chunk_start()
        samples.compute_chunk_start()

    else:
        # Two-pass approach for exact memory allocation
        # FIXME in cuda memalloc needs to be done but in python, this makes it not None and then during the first pass the logic is incorrect?
        if compute_intervals:
            intervals.chunk_cnts = mint.zeros(n_rays, dtype=ms.int32)
        if compute_samples:
            samples.chunk_cnts = mint.zeros(n_rays, dtype=ms.int32)

        # First pass - count segments: get chunk_cnts
        _traverse_rays(
            rays_o,
            rays_d,
            None,  # No mask in first pass
            binaries,
            aabbs,
            resolution,
            hits,
            t_sorted,
            t_indices,
            near_planes,
            far_planes,
            step_size,
            cone_angle,
            traverse_steps_limit,
            intervals,
            samples,
            None,
            first_pass=True,
        )

        # Allocate exact memory needed, cal chunk_starts for both ray segs from chunk_cnts, init as 0 isleft/right
        if compute_intervals:
            intervals.memalloc_data_from_chunk(True, True)
        if compute_samples:
            samples.memalloc_data_from_chunk(False, False, True)

        # Second pass - record segments
        _traverse_rays(
            rays_o,
            rays_d,
            None,
            binaries,
            aabbs,
            resolution,
            hits,
            t_sorted,
            t_indices,
            near_planes,
            far_planes,
            step_size,
            cone_angle,
            traverse_steps_limit,
            intervals,
            samples,
            terminate_planes,
            first_pass=False,
        )

    return intervals, samples, terminate_planes
