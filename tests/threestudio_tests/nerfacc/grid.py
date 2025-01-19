import numpy as np

import mindspore as ms
from mindspore import Tensor, mint


def test_ray_aabb_intersect():
    from mindone.models.nerfacc.grid import _ray_aabb_intersect, ray_aabb_intersect

    ms.manual_seed(42)
    n_rays = 1000
    n_aabbs = 100

    rays_o = mint.rand((n_rays, 3))
    rays_d = mint.normal(size=(n_rays, 3))
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
    aabb_min = mint.rand((n_aabbs, 3))
    aabb_max = aabb_min + mint.rand_like(aabb_min)
    aabbs = mint.cat([aabb_min, aabb_max], dim=-1)

    # [n_rays, n_aabbs]
    tmins, tmaxs, hits = ray_aabb_intersect(rays_o, rays_d, aabbs)
    print(hits)
    _tmins, _tmaxs, _hits = _ray_aabb_intersect(rays_o, rays_d, aabbs)
    assert np.allclose(tmins, _tmins), (tmins - _tmins).abs().max()
    assert np.allclose(tmaxs, _tmaxs), (tmaxs - _tmaxs).abs().max()
    assert (hits == _hits).all(), (hits == _hits).float().mean()

    # whether mid points are inside aabbs
    tmids = mint.clamp((tmins + tmaxs) / 2, min=0.0)
    points = tmids[:, :, None] * rays_d[:, None, :] + rays_o[:, None, :]
    _hits = (mint.logical_and(points >= aabb_min[None, ...], points <= aabb_max[None, ...])).all(dim=-1)
    assert np.allclose(hits, _hits)


def test_traverse_grids():
    from mindone.models.nerfacc.grid import _enlarge_aabb, traverse_grids

    ms.set_context(mode=1, pynative_synchronize=False)

    ms.manual_seed(42)
    n_rays = 8 * 64 * 64
    n_aabbs = 1

    rays_o = mint.normal(size=(n_rays, 3))
    rays_d = mint.normal(size=(n_rays, 3))
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)

    base_aabb = Tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
    aabbs = mint.stack([_enlarge_aabb(base_aabb, 2**i) for i in range(n_aabbs)])

    binaries = mint.rand(n_aabbs, 32, 32, 32) > 0.5

    intervals, samples, _ = traverse_grids(rays_o, rays_d, binaries, aabbs)

    ray_indices = samples.ray_indices
    t_starts = intervals.vals[intervals.is_left]
    t_ends = intervals.vals[intervals.is_right]
    positions = rays_o[ray_indices] + rays_d[ray_indices] * (t_starts + t_ends)[:, None] / 2.0
    print(f"\ntest traverse grid done: pos shape: {positions.shape}")
    # occs, selector = _query(positions, binaries, base_aabb)
    # assert occs.all(), occs.float().mean()
    # assert selector.all(), selector.float().mean()


def test_traverse_grids_test_mode():
    from mindone.models.nerfacc.grid import _enlarge_aabb, traverse_grids
    from mindone.models.nerfacc.volrend import accumulate_along_rays

    ms.set_context(mode=1, pynative_synchronize=True)
    ms.manual_seed(42)
    n_rays = 10
    n_aabbs = 4
    rays_o = mint.normal(size=(n_rays, 3))
    rays_d = mint.normal(size=(n_rays, 3))
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)

    base_aabb = Tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
    aabbs = mint.stack([_enlarge_aabb(base_aabb, 2**i) for i in range(n_aabbs)])

    binaries = mint.rand(n_aabbs, 32, 32, 32) > 0.5

    # ref results: train mode
    intervals, samples, _ = traverse_grids(rays_o, rays_d, binaries, aabbs)
    ray_indices = samples.ray_indices
    t_starts = intervals.vals[intervals.is_left]
    t_ends = intervals.vals[intervals.is_right]
    accum_t_starts = accumulate_along_rays(t_starts, None, ray_indices, n_rays)
    accum_t_ends = accumulate_along_rays(t_ends, None, ray_indices, n_rays)

    # test mode
    _accum_t_starts, _accum_t_ends = 0.0, 0.0
    _terminate_planes = None
    _rays_mask = None
    for run in range(2):
        _intervals, _samples, _terminate_planes = traverse_grids(
            rays_o,
            rays_d,
            binaries,
            aabbs,
            near_planes=_terminate_planes,
            traverse_steps_limit=4000,
            over_allocate=True,
            rays_mask=_rays_mask,
        )
        # only keep rays that are not terminated (i.e. reach the limit)
        _rays_mask = _samples.packed_info[:, 1] == 4000
        _ray_indices = _samples.ray_indices[_samples.is_valid]
        _t_starts = _intervals.vals[_intervals.is_left]
        _t_ends = _intervals.vals[_intervals.is_right]
        _accum_t_starts += accumulate_along_rays(_t_starts, None, _ray_indices, n_rays)
        _accum_t_ends += accumulate_along_rays(_t_ends, None, _ray_indices, n_rays)
        print(f"for run {run}, ishape: {_intervals.vals.shape}, sshape: {_samples.vals.shape}")
    # there shouldn't be any rays that are not terminated
    assert (~_rays_mask).all()
    # TODO: figure out where this small diff comes from
    assert np.allclose(_accum_t_starts, accum_t_starts, atol=1e-1)
    assert np.allclose(accum_t_ends, _accum_t_ends, atol=1e-1)


def test_traverse_grids_with_near_far_planes():
    from mindone.models.nerfacc.grid import traverse_grids

    rays_o = Tensor([[-1.0, 0.0, 0.0]])
    rays_d = Tensor([[1.0, 0.01, 0.01]])
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)

    binaries = mint.ones((1, 1, 1, 1), dtype=ms.bool_)
    aabbs = Tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])

    near_planes = Tensor([1.2])
    far_planes = Tensor([1.5])
    step_size = 0.05

    intervals, samples, _ = traverse_grids(
        rays_o=rays_o,
        rays_d=rays_d,
        binaries=binaries,
        aabbs=aabbs,
        step_size=step_size,
        near_planes=near_planes,
        far_planes=far_planes,
    )
    assert (intervals.vals >= (near_planes - step_size / 2)).all()
    assert (intervals.vals <= (far_planes + step_size / 2)).all()


def test_sampling_with_min_max_distances():
    from nerfacc_ms import OccGridEstimator

    ms.manual_seed(42)
    n_rays = 64
    levels = 4
    resolution = 32
    render_step_size = 0.01
    near_plane = 0.15
    far_plane = 0.85

    rays_o = mint.rand((n_rays, 3)) * 2 - 1.0
    rays_d = mint.rand((n_rays, 3))
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)

    aabb = Tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
    binaries = mint.rand((levels, resolution, resolution, resolution)) > 0.5
    t_min = mint.rand((n_rays,))
    t_max = t_min + mint.rand((n_rays,))

    grid_estimator = OccGridEstimator(roi_aabb=aabb, resolution=resolution, levels=levels)

    grid_estimator.binaries = binaries

    ray_indices, t_starts, t_ends = grid_estimator.sampling(
        rays_o=rays_o,
        rays_d=rays_d,
        near_plane=near_plane,
        far_plane=far_plane,
        t_min=t_min,
        t_max=t_max,
        render_step_size=render_step_size,
    )

    assert (t_starts >= (t_min[ray_indices] - render_step_size / 2)).all()
    assert (t_ends <= (t_max[ray_indices] + render_step_size / 2)).all()


def test_mark_invisible_cells():
    from nerfacc_ms import OccGridEstimator

    levels = 4
    resolution = 32
    width = 100
    height = 100
    fx, fy = width, height
    cx, cy = width / 2, height / 2

    aabb = Tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])

    grid_estimator = OccGridEstimator(roi_aabb=aabb, resolution=resolution, levels=levels)

    K = Tensor([[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]])

    pose = Tensor(
        [[[-1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 2.5]]],
    )

    grid_estimator.mark_invisible_cells(K, pose, width, height)

    assert (grid_estimator.occs == -1).sum() == 77660
    assert (grid_estimator.occs == 0).sum() == 53412


if __name__ == "__main__":
    # test_ray_aabb_intersect()
    test_traverse_grids()
    # test_traverse_grids_with_near_far_planes()
    # test_sampling_with_min_max_distances()
    # test_mark_invisible_cells()
    # test_traverse_grids_test_mode()

    # t_starts = Tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    # t_ends = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    # sigmas = Tensor([0.4, 0.8, 0.1, 0.8, 0.1, 0.0, 0.9])
    # ray_indices = Tensor([0, 0, 0, 1, 1, 2, 2], dtype=ms.int32)

    # from mindone.models.nerfacc.volrend import (
    #     render_weight_from_density,
    # )
    # weights, transmittance, alphas = render_weight_from_density(
    #     t_starts, t_ends, sigmas, ray_indices=ray_indices)
    # # weights: [0.33, 0.37, 0.03, 0.55, 0.04, 0.00, 0.59]
    # # transmittance: [1.00, 0.67, 0.30, 1.00, 0.45, 1.00, 1.00]
    # # alphas: [0.33, 0.55, 0.095, 0.55, 0.095, 0.00, 0.59]
    # print(transmittance)
    # print(alphas)
