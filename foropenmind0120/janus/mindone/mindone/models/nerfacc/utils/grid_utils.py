from mindspore import Tensor, mint, ops

from ..data_specs import RaySegmentsSpec
from ..setup_traverse import setup_traversal


# Helper function to compute dt
def calc_dt(t, cone_angle, dt_min, dt_max):
    return mint.clamp(t * cone_angle, min=dt_min, max=dt_max)


# Ray-box intersection (vectorized version)
def ray_aabb_intersect(rays_o, rays_d, aabbs, near_plane, far_plane, miss_value):
    """
    Computes ray-AABB intersections.
    Args:
        rays_o: Tensor of shape [n_rays, 3], ray origins.
        rays_d: Tensor of shape [n_rays, 3], ray directions.
        aabbs: Tensor of shape [n_aabbs, 6], axis-aligned bounding boxes (min and max coordinates).
        near_plane: Scalar, near clipping plane.
        far_plane: Scalar, far clipping plane.
        miss_value: Value to assign if the ray misses the AABB.

    Returns:
        t_mins: Tensor of shape [n_rays, n_aabbs], minimum intersection distances.
        t_maxs: Tensor of shape [n_rays, n_aabbs], maximum intersection distances.
        hits: Tensor of shape [n_rays, n_aabbs], boolean mask for hits.
    """
    n_rays = rays_o.shape[0]
    n_aabbs = aabbs.shape[0]

    # Expand rays and AABBs for pairwise intersection
    rays_o = rays_o.unsqueeze(1).expand((-1, n_aabbs, -1))  # [n_rays, n_aabbs, 3]
    rays_d = rays_d.unsqueeze(1).expand((-1, n_aabbs, -1))  # [n_rays, n_aabbs, 3]
    aabb_min = aabbs[:, :3].unsqueeze(0).expand((n_rays, -1, -1))  # [n_rays, n_aabbs, 3]
    aabb_max = aabbs[:, 3:].unsqueeze(0).expand((n_rays, -1, -1))  # [n_rays, n_aabbs, 3]

    # Compute inverse direction and handle division by zero
    inv_d = 1.0 / (rays_d + 1e-8)  # Avoid division by zero
    t_min = (aabb_min - rays_o) * inv_d
    t_max = (aabb_max - rays_o) * inv_d

    # Ensure t_min <= t_max
    t_min, t_max = mint.minimum(t_min, t_max), mint.maximum(t_min, t_max)

    # Compute intersection intervals
    t_min = mint.max(t_min, dim=-1)[0]  # [n_rays, n_aabbs]
    t_max = mint.min(t_max, dim=-1)[0]  # [n_rays, n_aabbs]

    # Check for valid intersections
    hits = mint.logical_and(mint.logical_and((t_min <= t_max), (t_max >= near_plane)), (t_min <= far_plane))
    t_mins = mint.where(hits, t_min, ops.full_like(t_min, miss_value))
    t_maxs = mint.where(hits, t_max, ops.full_like(t_max, miss_value))

    return t_mins, t_maxs, hits


class AABBSpec:
    def __init__(self, aabb):
        self.min = Tensor(aabb[:3])
        self.max = Tensor(aabb[3:])

    # def __init__(self, min_point, max_point):
    #     self.min = Tensor(min_point)
    #     self.max = Tensor(max_point)


class RaysSpec:
    def __init__(self):
        self.origins: Tensor
        self.dirs: Tensor


class PackedRaysSpec:
    def __init__(self, spec: RaysSpec):
        self.origins = spec.origins
        self.dirs = spec.dirs
        self.N = spec.origins.shape[0]


class SingleRaySpec:
    def __init__(self, *args):
        if isinstance(args[0], Tensor):
            rays_o, rays_d, tmin, tmax = args
            self.origin
        elif isinstance(args[0], PackedRaysSpec):
            rays_o, rays_d, tmin, tmax
        else:
            raise ValueError


# Traverse grids
def _traverse_grids(
    rays_o,
    rays_d,
    binaries,
    aabbs,
    t_mins,
    t_maxs,
    hits,
    near_planes,
    far_planes,
    step_size,
    cone_angle,
    compute_intervals=True,
    compute_samples=True,
    compute_terminate_planes=True,
    traverse_steps_limit=0,
    over_allocate=True,
    first_pass=False,
):
    """
    Traverses grids to compute ray intersections and samples. Adapts form the cuda kernel func. Serial.

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
    n_grids = binaries.shape[0]
    resolution = binaries.shape[1:]  # (resx, resy, resz)

    # Sort intersections
    t_sorted, t_indices = mint.sort(mint.cat([t_mins, t_maxs], dim=-1), dim=-1)

    # Outputs
    intervals = [] if compute_intervals else None
    samples = [] if compute_samples else None

    # parallel over rays
    for ray_idx in range(n_rays):
        near_plane = near_planes[ray_idx]
        far_plane = far_planes[ray_idx]
        if not first_pass:
            chunk_start = intervals.chunk_starts[ray_idx]
            chunk_start_bin = samples.chunk_starts[ray_idx]

        t_last = near_plane
        continuous = False
        base_hits = 0  # put it as 0 as now consider this a serial kernel
        n_intervals = 0
        n_samples = 0

        this_tmin = max(t_sorted[ray_idx, ray_idx], near_plane)
        this_tmax = min(t_sorted[ray_idx, ray_idx + 1], far_plane)
        if this_tmin >= this_tmax:
            continue  # Invalid interval

        # loop over intersections
        for i in range(0, n_grids * 2 - 1):
            is_entering = t_indices[ray_idx, i] < n_grids
            level = t_indices[ray_idx, i] % n_grids

            if not hits[ray_idx, level]:
                continue  # Skip grids that are not hit
            if not is_entering:
                next_is_entering = t_indices[i + 1] < n_grids
                if next_is_entering:
                    continue
                level = t_indices[i + 1] % n_grids
                if not hits[base_hits + level]:
                    continue  # skip grid not hitted

            # March through the grid
            if not continuous:
                if step_size <= 0.0:
                    t_last = this_tmin
                else:
                    dt = calc_dt(t_last, cone_angle, step_size, 1e10)
                    while t_last + dt * 0.5 < this_tmin:
                        t_last += dt

            # init prepare for traversal
            aabb = AABBSpec(aabbs[level * 6 : level * 6 + 6])
            ray = SingleRaySpec(rays_o + ray_idx * 3, rays_d + ray_idx * 3, near_plane, far_plane)
            eps = 1e-6

            delta, tdist, step_index, current_index, final_index = setup_traversal(
                ray, this_tmin, this_tmax, eps, aabb, resolution
            )
            while traverse_steps_limit <= 0 or n_samples < traverse_steps_limit:
                t_traverse = mint.min(tdist.x, mint.min(tdist.y, tdist.z))
                cell_id = (
                    current_index.x * resolution.y * resolution.z
                    + current_index.y * resolution.z
                    + current_index.z
                    + level * resolution.x * resolution.y * resolution.z
                )

                if not binaries[cell_id]:
                    # Skip the cell that is empty.
                    if step_size <= 0.0:  # March to t_traverse.
                        t_last = t_traverse
                    else:
                        dt = calc_dt(t_last, cone_angle, step_size, 1e10)
                        while True:  # March until t_mid is right after t_traverse.
                            if t_last + dt * 0.5 >= t_traverse:
                                break
                            t_last += dt
                    continuous = False
                else:
                    # This cell is not empty, so we need to traverse it.
                    while True:
                        if step_size <= 0.0:
                            t_next = t_traverse
                        else:  # March until t_mid is right after t_traverse.
                            dt = calc_dt(t_last, cone_angle, step_size, 1e10)
                            if t_last + dt * 0.5 >= t_traverse:
                                break
                            t_next = t_last + dt

                        # Write out the interval.
                        if intervals["chunk_cnts"] is not None:
                            if not continuous:
                                if not first_pass:  # Left side of the interval
                                    idx = chunk_start + n_intervals
                                    intervals["vals"][idx] = t_last
                                    intervals["ray_indices"][idx] = ray_idx
                                    intervals["is_left"][idx] = True
                                n_intervals += 1
                                if not first_pass:  # Right side of the interval
                                    idx = chunk_start + n_intervals
                                    intervals["vals"][idx] = t_next
                                    intervals["ray_indices"][idx] = ray_idx
                                    intervals["is_right"][idx] = True
                                n_intervals += 1
                            else:
                                if not first_pass:  # Right side of the interval, with continuity
                                    idx = chunk_start + n_intervals
                                    intervals["vals"][idx] = t_next
                                    intervals["ray_indices"][idx] = ray_idx
                                    intervals["is_left"][idx - 1] = True
                                    intervals["is_right"][idx] = True
                                n_intervals += 1

                        # Write out the sample.
                        if samples["chunk_cnts"] is not None:
                            if not first_pass:
                                idx = chunk_start_bin + n_samples
                                samples["vals"][idx] = (t_next + t_last) * 0.5
                                samples["ray_indices"][idx] = ray_idx
                            n_samples += 1

                        continuous = True
                        t_last = t_next
                        if t_next >= t_traverse:
                            break

            # Record intervals
            if compute_intervals:
                intervals.append((t_last, this_tmax))

            # Record samples
            if compute_samples:
                while t_last < this_tmax:
                    t_next = t_last + calc_dt(t_last, cone_angle, step_size, 1e10)
                    samples.append((t_last + t_next) * 0.5)
                    t_last = t_next

            continuous = True

    intervals = RaySegmentsSpec()
    samples = RaySegmentsSpec()

    return intervals, samples


# Example usage
if __name__ == "__main__":
    # Example inputs
    n_rays = 10
    n_grids = 5
    resx, resy, resz = 8, 8, 8

    rays_o = mint.rand(n_rays, 3)
    rays_d = mint.rand(n_rays, 3)
    binaries = ops.randint(0, 2, (n_grids, resx, resy, resz)).bool()
    aabbs = mint.rand(n_grids, 6)
    near_planes = mint.full((n_rays,), 0.1)
    far_planes = mint.full((n_rays,), 10.0)

    t_mins, t_maxs, hits = ray_aabb_intersect(rays_o, rays_d, aabbs, 0.0, 10.0, -1.0)
    intervals, samples = _traverse_grids(
        rays_o, rays_d, binaries, aabbs, t_mins, t_maxs, hits, near_planes, far_planes, step_size=0.1, cone_angle=0.01
    )

    print("Intervals:", intervals)
    print("Samples:", samples)
