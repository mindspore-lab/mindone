""" The renderer is a module that takes in rays, decides where to sample along each
ray, and computes pixel colors using the volume rendering equation.
"""
import logging
from typing import Dict, Optional

import mindspore as ms
import mindspore.nn as nn
from mindspore import _no_grad, mint, ops

from .math_utils import get_ray_limits_box, linspace
from .ray_marcher import MipRayMarcher2

logger = logging.getLogger(__name__)


class ImportanceRenderer(nn.Cell):
    """Modified version of the filtering the out-of-box sampels as TensorRF does."""

    def __init__(self, opts: Dict, dtype: ms.dtype, debug: bool = False, decoder: Optional[nn.Cell] = None):
        super().__init__()
        self.rendering_options = opts

        # ms graph mode paradigm: NOT passing hyperparam (that are not Tensors) as the construct args, make it class attr instead
        self.disparity_space_sampling = opts["disparity_space_sampling"]
        self.depth_resolution = opts["depth_resolution"]
        self.N_importance = opts["depth_resolution_importance"]
        self.ray_marcher = MipRayMarcher2(opts)

        self.plane_axes = generate_planes().astype(dtype)
        self.max_pool1d_layer = nn.MaxPool1d(2, 1, pad_mode="pad", padding=1)
        self.avg_pool1d_layer = nn.AvgPool1d(2, 1)

        self.decoder = decoder
        self.debug_logging = debug

    def project_onto_planes(
        self,
        planes: ms.Tensor,  # when calling this here from outside, particually it's sampling on the unit plane axes
        coordinates: ms.Tensor,
    ) -> ms.Tensor:
        """
        Does a projection of a 3D point onto a batch of 2D planes,
        returning 2D plane coordinates.

        Takes plane axes of shape n_planes, 3, 3
        # Takes coordinates of shape N, M, 3
        # returns projections of shape N*n_planes, M, 2
        """
        N, M, C = coordinates.shape
        n_planes, _, _ = planes.shape
        coordinates = coordinates.unsqueeze(1)
        coordinates = coordinates.broadcast_to((-1, n_planes, -1, -1)).reshape(N * n_planes, M, 3)

        inv_planes = mint.inverse(planes.to(ms.float32)).unsqueeze(0)
        inv_planes = inv_planes.broadcast_to((N, -1, -1, -1)).reshape(N * n_planes, 3, 3)

        projections = mint.bmm(coordinates, inv_planes.to(planes.dtype))
        return projections[..., :2]

    def sample_from_planes(self, plane_features, coordinates):
        mode = "bilinear"
        padding_mode = "zeros"
        assert padding_mode == "zeros"
        N, n_planes, C, H, W = plane_features.shape
        _, M, _ = coordinates.shape
        plane_features = plane_features.view(N * n_planes, C, H, W)

        coordinates = (2 / self.rendering_options["box_warp"]) * coordinates  # add specific box bounds
        projected_coordinates = self.project_onto_planes(self.plane_axes, coordinates).unsqueeze(1)

        output_features = (
            mint.nn.functional.grid_sample(
                plane_features,
                projected_coordinates,
                mode=mode,
                padding_mode=padding_mode,
                align_corners=False,
            )
            .permute(0, 3, 2, 1)
            .reshape(N, n_planes, M, C)
        )

        return output_features

    def run_model(
        self,
        planes,
        sample_coordinates,
    ):
        """Run triplane sampler & nerf decoder model"""
        sampled_features = self.sample_from_planes(planes, sample_coordinates)
        out = self.decoder(sampled_features)
        return out

    def _forward_pass(
        self,
        depths: ms.Tensor,
        ray_directions: ms.Tensor,
        ray_origins: ms.Tensor,
        planes: ms.Tensor,
    ):
        """
        Additional filtering is applied to filter out-of-box samples.
        """

        # context related variables
        batch_size, num_rays, samples_per_ray, _ = depths.shape

        # define sample points
        sample_coordinates = (ray_origins.unsqueeze(-2) + depths * ray_directions.unsqueeze(-2)).reshape(
            batch_size, -1, 3
        )

        # filter out-of-box samples
        mask_inbox = mint.logical_and(
            self.rendering_options["sampler_bbox_min"] <= sample_coordinates,
            sample_coordinates <= self.rendering_options["sampler_bbox_max"],
        )
        mask_inbox = mask_inbox.all(-1)

        # forward model according to all samples
        _rgb, _sigma = self.run_model(planes, sample_coordinates)

        # set out-of-box samples to zeros(rgb) & -inf(sigma)
        SAFE_GUARD = 3
        DATA_TYPE = _sigma.dtype
        colors_pass = mint.zeros((batch_size, num_rays * samples_per_ray, 3), dtype=DATA_TYPE)
        densities_pass = (
            ops.nan_to_num(mint.full((batch_size, num_rays * samples_per_ray, 1), -float("inf"), dtype=DATA_TYPE))
            / SAFE_GUARD
        )

        if self.debug_logging:
            logger.info(
                f"shape] depths: {mask_inbox.shape}, rd: {_rgb.shape}, ro: {colors_pass.shape}, planes: {planes.shape}"
            )
            logger.info(f"shape] mi: {mask_inbox.shape}, rgb: {_rgb.shape}, colorpass: {colors_pass.shape}")

        # colors_pass[mask_inbox] = _rgb[mask_inbox]
        mask_inbox = mask_inbox[..., None]
        colors_pass = mint.where(mask_inbox, _rgb, colors_pass)  # Tensor indexing assignment in G mode
        # densities_pass[mask_inbox] = _sigma[mask_inbox]  # GRAPH MODE: index val assignment using tensor cannot be mul dims
        densities_pass = mint.where(mask_inbox, _sigma, densities_pass)

        # reshape back
        colors_pass = colors_pass.reshape(batch_size, num_rays, samples_per_ray, colors_pass.shape[-1])
        densities_pass = densities_pass.reshape(batch_size, num_rays, samples_per_ray, densities_pass.shape[-1])

        return colors_pass, densities_pass

    def sort_samples(self, all_depths, all_colors, all_densities):
        _, indices = mint.sort(all_depths, dim=-2)
        all_depths = mint.gather(all_depths, -2, indices)
        all_colors = mint.gather(all_colors, -2, indices.broadcast_to((-1, -1, -1, all_colors.shape[-1])))
        all_densities = mint.gather(all_densities, -2, indices.broadcast_to((-1, -1, -1, 1)))
        return all_depths, all_colors, all_densities

    def unify_samples(self, depths1, colors1, densities1, depths2, colors2, densities2, normals1=None, normals2=None):
        all_depths = mint.cat([depths1, depths2], dim=-2)
        all_colors = mint.cat([colors1, colors2], dim=-2)
        all_densities = mint.cat([densities1, densities2], dim=-2)

        if normals1 is not None and normals2 is not None:
            all_normals = mint.cat([normals1, normals2], dim=-2)
        else:
            all_normals = None

        _, indices = mint.sort(all_depths, dim=-2)
        all_depths = mint.gather(all_depths, -2, indices)
        all_colors = mint.gather(all_colors, -2, indices.broadcast_to((-1, -1, -1, all_colors.shape[-1])))
        all_densities = mint.gather(all_densities, -2, indices.broadcast_to((-1, -1, -1, 1)))

        if all_normals is not None:
            all_normals = mint.gather(all_normals, -2, indices.broadcast_to((-1, -1, -1, all_normals.shape[-1])))
            return all_depths, all_colors, all_normals, all_densities

        return all_depths, all_colors, all_densities

    def sample_stratified(
        self,
        ray_origins: ms.Tensor,  # b n 3
        ray_start: ms.Tensor,  # b n 1
        ray_end: ms.Tensor,  # b n 1
    ):
        """
        Return depths of approximately uniformly spaced samples along rays.
        """
        N, M, _ = ray_origins.shape
        if self.disparity_space_sampling:
            depths_coarse = (
                mint.linspace(0, 1, self.depth_resolution).reshape(1, 1, self.depth_resolution, 1).tile((N, M, 1, 1))
            )
            depth_delta = 1 / (self.depth_resolution - 1)
            depths_coarse += mint.rand_like(depths_coarse) * depth_delta
            depths_coarse = 1.0 / (1.0 / ray_start * (1.0 - depths_coarse) + 1.0 / ray_end * depths_coarse)
        else:
            # print(f'shape: ray start: {ray_start.shape}, ray end: {ray_end.shape}')
            depths_coarse = linspace(ray_start, ray_end, self.depth_resolution).permute(1, 2, 0, 3)
            depth_delta = (ray_end - ray_start) / (self.depth_resolution - 1)
            depths_coarse += mint.rand_like(depths_coarse) * depth_delta[..., None]

        return depths_coarse

    def sample_pdf(self, bins, weights):
        """
        Sample @N_importance samples from @bins with distribution defined by @weights.
        Inputs:
            bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
            weights: (N_rays, N_samples_)
            N_importance: the number of samples to draw from the distribution
        Not-tensor Inputs:
            det: deterministic or not
            eps: a small number to prevent division by zero
        Outputs:
            samples: the sampled samples
        """
        det = (False,)
        eps = 1e-5
        N_rays, N_samples_ = weights.shape
        weights = weights + eps  # prevent division by zero (don't do inplace op!)
        pdf = weights / mint.sum(weights, -1, keepdim=True)  # (N_rays, N_samples_)
        cdf = mint.cumsum(pdf, -1)  # (N_rays, N_samples), cumulative distribution function
        cdf = mint.cat([mint.zeros_like(cdf[:, :1]), cdf], -1)  # (N_rays, N_samples_+1)
        # padded to 0~1 inclusive

        if det:
            u = mint.linspace(0, 1, self.N_importance)
            u = u.broadcast_to((N_rays, self.N_importance))
        else:
            u = mint.rand(N_rays, self.N_importance)
        u = u.contiguous()

        inds = mint.searchsorted(cdf, u, right=True)
        below = mint.clamp(inds - 1, min=0)
        above = mint.clamp(inds, max=N_samples_)

        inds_sampled = mint.stack([below, above], -1).view(N_rays, 2 * self.N_importance)
        cdf_g = mint.gather(cdf, 1, inds_sampled).view(N_rays, self.N_importance, 2)
        bins_g = mint.gather(bins, 1, inds_sampled).view(N_rays, self.N_importance, 2)

        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom[denom < eps] = 1  # denom equals 0 means a bin has weight 0, in which case it will not be sampled
        # anyway, therefore any value for it is fine (set to 1 here)

        samples = bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (bins_g[..., 1] - bins_g[..., 0])
        return samples

    def sample_importance(
        self,
        z_vals,
        weights,
    ):
        """
        Return depths of importance sampled points along rays. See NeRF importance sampling for more.
        """
        with _no_grad():
            batch_size, num_rays, samples_per_ray, _ = z_vals.shape

            z_vals = z_vals.reshape(batch_size * num_rays, samples_per_ray)
            weights = weights.reshape(batch_size * num_rays, -1)  # -1 to account for loss of 1 sample in MipRayMarcher

            # smooth weights
            weights = self.max_pool1d_layer(weights.unsqueeze(1))
            weights = self.avg_pool1d_layer(weights).squeeze()
            weights = weights + 0.01

            z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
            importance_z_vals = self.sample_pdf(z_vals_mid, weights[:, 1:-1]).reshape(
                batch_size, num_rays, self.N_importance, 1
            )
        return importance_z_vals

    def construct(
        self,
        planes: ms.Tensor,
        ray_origins: ms.Tensor,
        ray_directions: ms.Tensor,
    ):
        ray_start, ray_end = get_ray_limits_box(ray_origins, ray_directions)
        is_ray_valid = ray_end > ray_start

        # FIXME below take item may degrade the shape, potentially into unknown errors...
        if mint.any(is_ray_valid).item():
            ray_start[~is_ray_valid] = ray_start[is_ray_valid].min()
            ray_end[~is_ray_valid] = ray_start[is_ray_valid].max()
        depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end)

        # Coarse Pass
        colors_coarse, densities_coarse = self._forward_pass(
            depths=depths_coarse,
            ray_directions=ray_directions,
            ray_origins=ray_origins,
            planes=planes,
        )

        # print(f'input below cc: {colors_coarse}\n weights: {densities_coarse}\n depth color: {densities_coarse}')
        # print(f'input below cc: {colors_coarse}\n weights: {densities_coarse}\n depth color: {densities_coarse}')
        # ops.print_('n importance is', self.N_importance)

        # Fine Pass
        _, _, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse)

        depths_fine = self.sample_importance(depths_coarse, weights)

        colors_fine, densities_fine = self._forward_pass(
            depths=depths_fine,
            ray_directions=ray_directions,
            ray_origins=ray_origins,
            planes=planes,
        )

        all_depths, all_colors, all_densities = self.unify_samples(
            depths_coarse, colors_coarse, densities_coarse, depths_fine, colors_fine, densities_fine
        )
        rgb_final, depth_final, weights = self.ray_marcher(all_colors, all_densities, all_depths)

        return rgb_final, depth_final, weights.sum(2)

    # [inference] for run meshing code, not trained
    def run_model_activated(self, planes, sample_coordinates, options=None):
        _rgb, _sigma = self.run_model(planes, sample_coordinates)
        _sigma = self.activation_factory(options)(_sigma)
        return _rgb, _sigma


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
