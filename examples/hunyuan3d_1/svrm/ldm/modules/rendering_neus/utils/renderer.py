# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
# Modified by Zexin He
# The modifications are subject to the same license as the original.


"""
The renderer is a module that takes in rays, decides where to sample along each
ray, and computes pixel colors using the volume rendering equation.
"""

import mindspore as ms
from mindspore import mint, nn, ops

from ..utils import no_grad
from . import math_utils
from .ray_marcher import MipRayMarcher2


def generate_planes():
    """
    Defines planes by the three vectors that form the "axes" of the
    plane. Should work with arbitrary number of planes and planes of
    arbitrary orientation.

    Bugfix reference: https://github.com/NVlabs/eg3d/issues/67
    """
    return ms.Tensor(
        [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 0, 1], [0, 1, 0]], [[0, 0, 1], [0, 1, 0], [1, 0, 0]]],
        dtype=ms.float32,
    )


def project_onto_planes(planes, coordinates):
    """
    Does a projection of a 3D point onto a batch of 2D planes,
    returning 2D plane coordinates.

    Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    """
    if planes.dtype != ms.float32:
        planes = planes.float()
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).broadcast_to((-1, n_planes, -1, -1)).reshape(N * n_planes, M, 3)
    inv_planes = (
        mint.linalg.inv(planes).unsqueeze(0).broadcast_to((N, -1, -1, -1)).reshape(N * n_planes, 3, 3)
    )  # MatrixInverseExt only supports float32, planes should be float32
    projections = mint.bmm(coordinates.to(planes.dtype), inv_planes)
    return projections[..., :2].to(coordinates.dtype)


def sample_from_planes(plane_axes, plane_features, coordinates, mode="bilinear", padding_mode="zeros", box_warp=None):
    assert padding_mode == "zeros"
    N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape
    plane_features = plane_features.view((N * n_planes, C, H, W))

    coordinates = (2 / box_warp) * coordinates  # add specific box bounds

    projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1)
    output_features = (
        mint.nn.functional.grid_sample(
            plane_features.float(),
            projected_coordinates.float(),
            mode=mode,
            padding_mode=padding_mode,
            align_corners=False,
        )
        .permute((0, 3, 2, 1))
        .reshape(N, n_planes, M, C)
    )
    return output_features


def sample_from_3dgrid(grid, coordinates):
    """
    Expects coordinates in shape (batch_size, num_points_per_batch, 3)
    Expects grid in shape (1, channels, H, W, D)
    (Also works if grid has batch size)
    Returns sampled features of shape (batch_size, num_points_per_batch, feature_channels)
    """
    batch_size, n_coords, n_dims = coordinates.shape
    sampled_features = mint.nn.functional.grid_sample(
        grid.broadcast_to((batch_size, -1, -1, -1, -1)),
        coordinates.reshape(batch_size, 1, 1, -1, n_dims),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    N, C, H, W, D = sampled_features.shape
    sampled_features = sampled_features.permute((0, 4, 3, 2, 1)).reshape(N, H * W * D, C)
    return sampled_features


class ImportanceRenderer(nn.Cell):
    """
    Modified original version to filter out-of-box samples as TensoRF does.

    Reference:
    TensoRF: https://github.com/apchenstu/TensoRF/blob/main/models/tensorBase.py#L277
    """

    def __init__(self):
        super().__init__()
        self.activation_factory = self._build_activation_factory()
        self.ray_marcher = MipRayMarcher2(self.activation_factory)
        self.plane_axes = generate_planes()
        self.max_pool1d = nn.MaxPool1d(kernel_size=2, stride=1, padding=1, pad_mode="pad")

    def _build_activation_factory(self):
        def activation_factory(options: dict):
            if options["clamp_mode"] == "softplus":
                return lambda x: mint.nn.functional.softplus(
                    x - 1
                )  # activation bias of -1 makes things initialize better
            else:
                assert False, "Renderer only supports `clamp_mode`=`softplus`!"

        return activation_factory

    def _forward_pass(
        self,
        depths: ms.Tensor,
        ray_directions: ms.Tensor,
        ray_origins: ms.Tensor,
        planes: ms.Tensor,
        decoder: nn.Cell,
        rendering_options: dict,
    ):
        """
        Additional filtering is applied to filter out-of-box samples.
        Modifications made by Zexin He.
        """

        # context related variables
        batch_size, num_rays, samples_per_ray, _ = depths.shape

        # define sample points with depths
        sample_directions = (
            ray_directions.unsqueeze(-2).broadcast_to((-1, -1, samples_per_ray, -1)).reshape(batch_size, -1, 3)
        )
        sample_coordinates = (ray_origins.unsqueeze(-2) + depths * ray_directions.unsqueeze(-2)).reshape(
            batch_size, -1, 3
        )
        # print(f'min bbox: {sample_coordinates.min()}, max bbox: {sample_coordinates.max()}')

        # filter out-of-box samples
        mask_inbox = (rendering_options["sampler_bbox_min"] <= sample_coordinates) & (
            sample_coordinates <= rendering_options["sampler_bbox_max"]
        )
        mask_inbox = mask_inbox.all(-1)

        # forward model according to all samples
        _out = self.run_model(planes, decoder, sample_coordinates, sample_directions, rendering_options)

        # set out-of-box samples to zeros(rgb) & -inf(sigma)
        SAFE_GUARD = 3
        DATA_TYPE = _out["sdf"].dtype
        colors_pass = mint.zeros((batch_size, num_rays * samples_per_ray, 3), dtype=DATA_TYPE)
        normals_pass = mint.zeros((batch_size, num_rays * samples_per_ray, 3), dtype=DATA_TYPE)
        sdfs_pass = (
            mint.nan_to_num(mint.full((batch_size, num_rays * samples_per_ray, 1), -float("inf"), dtype=DATA_TYPE))
            / SAFE_GUARD
        )

        # colors_pass[mask_inbox], sdfs_pass[mask_inbox] = _out['rgb'][mask_inbox], _out['sdf'][mask_inbox]
        colors_pass[mask_inbox], sdfs_pass = _out["rgb"][mask_inbox], _out["sdf"]
        normals_pass = _out["normal"]

        # reshape back
        colors_pass = colors_pass.reshape(batch_size, num_rays, samples_per_ray, colors_pass.shape[-1])
        sdfs_pass = sdfs_pass.reshape(batch_size, num_rays, samples_per_ray, sdfs_pass.shape[-1])
        normals_pass = normals_pass.reshape(batch_size, num_rays, samples_per_ray, normals_pass.shape[-1])

        return colors_pass, sdfs_pass, normals_pass, _out["sdf_grad"]

    def construct(self, planes, decoder, ray_origins, ray_directions, rendering_options, bgcolor=None):
        if rendering_options["ray_start"] == "auto" == rendering_options["ray_end"]:
            ray_start, ray_end = math_utils.get_ray_limits_box(
                ray_origins, ray_directions, box_side_length=rendering_options["box_warp"]
            )  # [1, N_ray, 1]
            is_ray_valid = ray_end > ray_start
            if mint.any(is_ray_valid).item():
                ray_start[~is_ray_valid] = ray_start[is_ray_valid].min()
                ray_end[~is_ray_valid] = ray_start[is_ray_valid].max()
            depths_coarse = self.sample_stratified(
                ray_origins,
                ray_start,
                ray_end,
                rendering_options["depth_resolution"],
                rendering_options["disparity_space_sampling"],
            )  # [1, N_ray, N_sample, 1]】
        else:
            # Create stratified depth samples
            depths_coarse = self.sample_stratified(
                ray_origins,
                rendering_options["ray_start"],
                rendering_options["ray_end"],
                rendering_options["depth_resolution"],
                rendering_options["disparity_space_sampling"],
            )

        # Coarse Pass
        colors_coarse, sdfs_coarse, normals_coarse, sdf_grad = self._forward_pass(
            depths=depths_coarse,
            ray_directions=ray_directions,
            ray_origins=ray_origins,
            planes=planes,
            decoder=decoder,
            rendering_options=rendering_options,
        )

        # Fine Pass
        N_importance = rendering_options["depth_resolution_importance"]

        if N_importance > 0:
            _, _, weights = self.ray_marcher(
                colors_coarse,
                sdfs_coarse,
                depths_coarse,
                sdf_grad.reshape(*normals_coarse.shape),
                ray_directions,
                rendering_options,
                bgcolor,
            )

            depths_fine = self.sample_importance(depths_coarse, weights, N_importance)

            colors_fine, densities_fine = self._forward_pass(
                depths=depths_fine,
                ray_directions=ray_directions,
                ray_origins=ray_origins,
                planes=planes,
                decoder=decoder,
                rendering_options=rendering_options,
            )
            densities_coarse = None  # unused yet
            all_depths, all_colors, all_densities = self.unify_samples(
                depths_coarse, colors_coarse, densities_coarse, depths_fine, colors_fine, densities_fine
            )
            ####
            # dists = depths_coarse[:, :, 1:, :] - depths_coarse[:, :, :-1, :]
            # inter =  (ray_end - ray_start) / ( rendering_options['depth_resolution'] + rendering_options['depth_resolution_importance'] - 1) # [1, N_ray, 1]
            # dists = mint.cat([dists, inter.unsqueeze(2), 2])
            ####

            # Aggregate
            rgb_final, depth_final, weights = self.ray_marcher(
                all_colors, all_densities, all_depths, rendering_options, bgcolor
            )
        else:
            # dists = depths_coarse[:, :, 1:, :] - depths_coarse[:, :, :-1, :]
            # inter =  (ray_end - ray_start) / ( rendering_options['depth_resolution'] - 1) # [1, N_ray, 1]
            # dists = mint.cat([dists, inter.unsqueeze(2)], 2)

            rgb_final, depth_final, weights, normal_final = self.ray_marcher(
                colors_coarse,
                sdfs_coarse,
                depths_coarse,
                sdf_grad.reshape(*normals_coarse.shape),
                ray_directions,
                rendering_options,
                bgcolor,
                normals_coarse,
            )

        return rgb_final, depth_final, weights.sum(2), sdf_grad, normal_final

    def run_model(self, planes, decoder, sample_coordinates, sample_directions, options):
        plane_axes = self.plane_axes
        out = decoder(sample_directions, sample_coordinates, plane_axes, planes, options)
        # if options.get('density_noise', 0) > 0:
        #     out['sigma'] += ops.randn_like(out['sigma']) * options['density_noise']
        return out

    def run_model_activated(self, planes, decoder, sample_coordinates, sample_directions, options):
        out = self.run_model(planes, decoder, sample_coordinates, sample_directions, options)
        out["sigma"] = self.activation_factory(options)(out["sigma"])
        return out

    def sort_samples(self, all_depths, all_colors, all_densities):
        _, indices = mint.sort(all_depths, dim=-2)
        all_depths = mint.gather(all_depths, -2, indices)
        all_colors = mint.gather(all_colors, -2, indices.broadcast_to((-1, -1, -1, all_colors.shape[-1])))
        all_densities = mint.gather(all_densities, -2, indices.broadcast_to((-1, -1, -1, 1)))
        return all_depths, all_colors, all_densities

    def unify_samples(self, depths1, colors1, densities1, depths2, colors2, densities2):
        all_depths = mint.cat([depths1, depths2], dim=-2)
        all_colors = mint.cat([colors1, colors2], dim=-2)
        all_densities = mint.cat([densities1, densities2], dim=-2)

        _, indices = mint.sort(all_depths, dim=-2)
        all_depths = mint.gather(all_depths, -2, indices)
        all_colors = mint.gather(all_colors, -2, indices.broadcast_to((-1, -1, -1, all_colors.shape[-1])))
        all_densities = mint.gather(all_densities, -2, indices.broadcast_to((-1, -1, -1, 1)))

        return all_depths, all_colors, all_densities

    def sample_stratified(self, ray_origins, ray_start, ray_end, depth_resolution, disparity_space_sampling=False):
        """
        Return depths of approximately uniformly spaced samples along rays.
        """
        N, M, _ = ray_origins.shape
        if disparity_space_sampling:
            depths_coarse = (
                ops.linspace(
                    0,
                    1,
                    depth_resolution,
                )
                .reshape(1, 1, depth_resolution, 1)
                .tile((N, M, 1, 1))
            )
            depth_delta = 1 / (depth_resolution - 1)
            depths_coarse += mint.rand_like(depths_coarse) * depth_delta
            depths_coarse = 1.0 / (1.0 / ray_start * (1.0 - depths_coarse) + 1.0 / ray_end * depths_coarse)
        else:
            if type(ray_start) == ms.Tensor:
                depths_coarse = math_utils.linspace(ray_start, ray_end, depth_resolution).permute((1, 2, 0, 3))
                depth_delta = (ray_end - ray_start) / (depth_resolution - 1)
                depths_coarse += mint.rand_like(depths_coarse) * depth_delta[..., None]
            else:
                depths_coarse = (
                    ops.linspace(ray_start, ray_end, depth_resolution)
                    .reshape(1, 1, depth_resolution, 1)
                    .tile((N, M, 1, 1))
                )
                depth_delta = (ray_end - ray_start) / (depth_resolution - 1)
                depths_coarse += mint.rand_like(depths_coarse) * depth_delta

        return depths_coarse

    def sample_importance(self, z_vals, weights, N_importance):
        """
        Return depths of importance sampled points along rays. See NeRF importance sampling for more.
        """
        with no_grad():  # pynative only
            batch_size, num_rays, samples_per_ray, _ = z_vals.shape

            z_vals = z_vals.reshape(batch_size * num_rays, samples_per_ray)
            weights = weights.reshape(batch_size * num_rays, -1)  # -1 to account for loss of 1 sample in MipRayMarcher

            # smooth weights
            weights = self.max_pool1d(weights.unsqueeze(1).float())
            weights = ops.avg_pool1d(weights, 2, 1).squeeze()
            weights = weights + 0.01

            z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
            importance_z_vals = self.sample_pdf(z_vals_mid, weights[:, 1:-1], N_importance).reshape(
                batch_size, num_rays, N_importance, 1
            )
        return importance_z_vals

    def sample_pdf(self, bins, weights, N_importance, det=False, eps=1e-5):
        """
        Sample @N_importance samples from @bins with distribution defined by @weights.
        Inputs:
            bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
            weights: (N_rays, N_samples_)
            N_importance: the number of samples to draw from the distribution
            det: deterministic or not
            eps: a small number to prevent division by zero
        Outputs:
            samples: the sampled samples
        """
        N_rays, N_samples_ = weights.shape
        weights = weights + eps  # prevent division by zero (don't do inplace op!)
        pdf = weights / mint.sum(weights, -1, keepdim=True)  # (N_rays, N_samples_)
        cdf = mint.cumsum(pdf, -1)  # (N_rays, N_samples), cumulative distribution function
        cdf = mint.cat([mint.zeros_like(cdf[:, :1]), cdf], -1)  # (N_rays, N_samples_+1)
        # padded to 0~1 inclusive

        if det:
            u = ops.linspace(0, 1, N_importance)
            u = u.broadcast_to((N_rays, N_importance))
        else:
            u = mint.rand((N_rays, N_importance))
        u = u.contiguous()

        inds = mint.searchsorted(cdf, u, right=True)
        below = mint.clamp(inds - 1, min=0)
        above = mint.clamp(inds, max=N_samples_)

        inds_sampled = mint.stack([below, above], -1).view((N_rays, 2 * N_importance))
        cdf_g = mint.gather(cdf, 1, inds_sampled).view((N_rays, N_importance, 2))
        bins_g = mint.gather(bins, 1, inds_sampled).view((N_rays, N_importance, 2))

        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom[denom < eps] = 1  # denom equals 0 means a bin has weight 0, in which case it will not be sampled
        # anyway, therefore any value for it is fine (set to 1 here)

        samples = bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (bins_g[..., 1] - bins_g[..., 0])
        return samples
