from dataclasses import dataclass
from typing import Optional, Tuple

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import Renderer

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, mint, ops

from . import math_utils
from .ray_marcher import MipRayMarcher2, RayOpts


@threestudio.register("nerf-volume-renderer")
class NeRFVolumeRenderer(Renderer):
    @dataclass
    class Config(Renderer.Config):
        num_samples_per_ray: int = 512
        grid_prune: bool = False
        disparity_space_sampling: bool = False
        sampler_bbox_min: int = -1
        sampler_bbox_max: int = 1
        depth_resolution: int = 256
        depth_resolution_importance: int = 256

    cfg: Config

    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure(geometry, material, background)
        self.cfg.depth_resolution = int(
            self.cfg.num_samples_per_ray / 2
        )  # needs to be cal according to cfg #spr, thus cannot put in the class def
        self.cfg.depth_resolution_importance = int(self.cfg.num_samples_per_ray / 2)
        self.render_step_size = 1.732 * 2 * self.cfg.radius / self.cfg.num_samples_per_ray
        # mip-style renderer
        self.ray_marcher = MipRayMarcher2(RayOpts)
        self.max_pool1d_layer = nn.MaxPool1d(2, 1, pad_mode="pad", padding=1)
        self.avg_pool1d_layer = nn.AvgPool1d(2, 1)

    @ms._no_grad()
    def sample_stratified(
        self,
        ray_origins: Tensor,  # b n 3
        ray_start: Tensor,  # b n 1
        ray_end: Tensor,  # b n 1
    ) -> Tensor:  # samples of rays as depths
        """
        Return depths of approximately uniformly spaced samples along rays.
        """
        N, M, _ = ray_origins.shape
        if self.cfg.disparity_space_sampling:
            depths_coarse = (
                mint.linspace(0, 1, self.cfg.depth_resolution)
                .reshape(1, 1, self.cfg.depth_resolution, 1)
                .tile((N, M, 1, 1))
            )
            depth_delta = 1 / (self.cfg.depth_resolution - 1)
            depths_coarse += mint.rand_like(depths_coarse) * depth_delta
            depths_coarse = 1.0 / (1.0 / ray_start * (1.0 - depths_coarse) + 1.0 / ray_end * depths_coarse)
        else:
            # print(f'shape: ray start: {ray_start.shape}, ray end: {ray_end.shape}')
            depths_coarse = math_utils.linspace(ray_start, ray_end, self.cfg.depth_resolution).permute(1, 2, 0, 3)
            depth_delta = (ray_end - ray_start) / (self.cfg.depth_resolution - 1)

        return depths_coarse

    def sample_importance(
        self,
        z_vals: Tensor,
        weights: Tensor,
    ) -> Tensor:  # samples of rays as depths
        """
        Return depths of importance sampled points along rays. See NeRF importance sampling for more.
        """
        with ms._no_grad():
            batch_size, num_rays, samples_per_ray, _ = z_vals.shape

            z_vals = z_vals.reshape(batch_size * num_rays, samples_per_ray)
            weights = weights.reshape(batch_size * num_rays, -1)  # -1 to account for loss of 1 sample in MipRayMarcher

            # smooth weights
            weights = self.max_pool1d_layer(weights.unsqueeze(1))
            weights = self.avg_pool1d_layer(weights).squeeze()
            weights = weights + 0.01

            z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
            importance_z_vals = self.sample_pdf(z_vals_mid, weights[:, 1:-1]).reshape(
                batch_size, num_rays, self.cfg.depth_resolution_importance, 1
            )
            return importance_z_vals

    @ms._no_grad()
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
            u = mint.linspace(0, 1, self.cfg.depth_resolution_importance)
            u = u.broadcast_to((N_rays, self.cfg.depth_resolution_importance))
        else:
            u = mint.rand(N_rays, self.cfg.depth_resolution_importance)

        inds = mint.searchsorted(cdf, u, right=True)
        below = mint.clamp(inds - 1, min=0)
        above = mint.clamp(inds, max=N_samples_)

        inds_sampled = mint.stack([below, above], -1).view(N_rays, 2 * self.cfg.depth_resolution_importance)
        cdf_g = mint.gather(cdf, 1, inds_sampled).view(N_rays, self.cfg.depth_resolution_importance, 2)
        bins_g = mint.gather(bins, 1, inds_sampled).view(N_rays, self.cfg.depth_resolution_importance, 2)

        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom[denom < eps] = 1  # denom equals 0 means a bin has weight 0, in which case it will not be sampled
        # anyway, therefore any value for it is fine (set to 1 here)

        samples = bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (bins_g[..., 1] - bins_g[..., 0])
        return samples

    def _forward_sampling_with_geo(
        self, depths: Tensor, ray_directions: Tensor, ray_origins: Tensor, **kwargs
    ) -> Tuple[Tensor, Tensor]:  # samples of rays as colors and densities
        """
        Additional filtering is applied to filter out-of-box samples.

        Model forwarding: runs geometry mlps.
        """

        # context related variables
        batch_size, num_rays, samples_per_ray, _ = depths.shape

        # define sample points
        sample_coordinates = (ray_origins.unsqueeze(-2) + depths * ray_directions.unsqueeze(-2)).reshape(
            batch_size, -1, 3
        )

        # filter out-of-box samples
        mask_inbox = mint.logical_and(
            self.cfg.sampler_bbox_min <= sample_coordinates, sample_coordinates <= self.cfg.sampler_bbox_max
        )
        mask_inbox = mask_inbox.all(-1)

        # rgb/feat decoded by the geometry decoder from each sampled coords
        # _rgb, _sigma = self.run_model(planes, sample_coordinates)
        geo_out = self.geometry(sample_coordinates, output_normal=self.material.requires_normal)
        _rgb = geo_out["features"]
        _sigma = geo_out["density"]

        # set out-of-box samples to zeros(rgb) & -inf(sigma)
        SAFE_GUARD = 3
        DATA_TYPE = _sigma.dtype
        colors_pass = mint.zeros((batch_size, num_rays * samples_per_ray, 3), dtype=DATA_TYPE)
        densities_pass = (
            ops.nan_to_num(mint.full((batch_size, num_rays * samples_per_ray, 1), -float("inf"), dtype=DATA_TYPE))
            / SAFE_GUARD
        )

        mask_inbox = mask_inbox[..., None]
        colors_pass = mint.where(mask_inbox, _rgb, colors_pass)  # Tensor indexing assignment ok for graph mode
        densities_pass = mint.where(mask_inbox, _sigma, densities_pass)

        # reshape back
        colors_pass = colors_pass.reshape(batch_size, num_rays, samples_per_ray, colors_pass.shape[-1])
        densities_pass = densities_pass.reshape(batch_size, num_rays, samples_per_ray, densities_pass.shape[-1])
        return colors_pass, densities_pass

    def construct(
        self,
        rays_o: Tensor,  # b h w 3
        rays_d: Tensor,  # b h w 3
        # light_positions: Tensor,  # b 3
        bg_color: Optional[Tensor] = None,
        # **kwargs
    ):
        """this an lrm-alike nerf renderer implementation."""
        # threestudio.info(f'within nerf renderer, the shape rays_o is {rays_o.shape}')
        bs, height, width = rays_o.shape[:3]
        with ms._no_grad():
            rays_o, rays_d = rays_o.reshape(bs, -1, 3), rays_d.reshape(bs, -1, 3)
            ray_start, ray_end = math_utils.sampling_get_ray_limits_box(rays_o, rays_d)

            is_ray_valid = ray_end > ray_start
            if mint.any(is_ray_valid).item():
                ray_start[~is_ray_valid] = ray_start[is_ray_valid].min()
                ray_end[~is_ray_valid] = ray_start[is_ray_valid].max()

            # stratified depths which are used to cal colors and densities below
            depths_coarse = self.sample_stratified(rays_o, ray_start, ray_end)

        # coarse pass for the sample weights
        colors_coarse, densities_coarse = self._forward_sampling_with_geo(
            depths=depths_coarse,
            ray_directions=rays_d,
            ray_origins=rays_o,
        )

        colors_coarse = self.material(colors_coarse)

        # counterparts for nerfacc.accumulate_along_rays()
        weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, True)
        depths_fine = self.sample_importance(z_vals=depths_coarse, weights=weights)

        # fine pass do the hierachical sampling
        colors_fine, densities_fine = self._forward_sampling_with_geo(
            depths=depths_fine,
            ray_directions=rays_d,
            ray_origins=rays_o,
        )
        # colors_fine = self.material(colors_fine)  # comment for now as there is only one material mlp inference in original repo
        comp_rgb_fg, comp_depth, weights = self.ray_marcher(colors_fine, densities_fine, depths_fine)
        comp_rgb_bg = self.background(dirs=rays_d)

        if bg_color is None:
            bg_color = comp_rgb_bg
        else:
            if bg_color.shape[:-1] == (bs,):
                # e.g. constant random color used for Zero123
                # [bs,3] -> [bs, 1, 1, 3]):
                bg_color = bg_color.unsqueeze(1).unsqueeze(1)
                #        -> [bs, height, width, 3]):
                bg_color = bg_color.broadcast_to(-1, height, width, -1)

        if bg_color.shape[:-1] == (bs, height, width):
            bg_color = bg_color.reshape(bs * height * width, -1)

        opacity = weights.sum(-2)
        composite_rgb = comp_rgb_fg + bg_color * (1.0 - opacity)
        # composite_rgb = comp_rgb_fg + bg_color * opacity

        out = {"comp_rgb": composite_rgb.view(bs, height, width, -1), "opacity": opacity.view(bs, height, width, 1)}

        return out

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False) -> None:
        if self.cfg.grid_prune:

            def occ_eval_fn(x):
                density = self.geometry.forward_density(x)
                return density * self.render_step_size

            if self.training and not on_load_weights:
                # self.estimator.update_every_n_steps(
                #     step=global_step, occ_eval_fn=occ_eval_fn
                # )
                threestudio.debug("updating step for renderer, but as we are not using nerfacc thus nothing done..")
