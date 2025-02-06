from typing import Dict, Optional

import mindspore.nn as nn
from mindspore import Tensor, mint, ops

RayOpts = {"white_back": False}  # the bg is determined by the bg module


class MipRayMarcher2(nn.Cell):
    def __init__(self, rendering_options: Optional[Dict] = None):
        super().__init__()
        self.white_back = rendering_options.get("white_back", False)

    def construct(self, colors: Tensor, densities: Tensor, depths: Tensor, get_weigths: bool = False):
        dtype = colors.dtype
        deltas = depths[:, :, 1:] - depths[:, :, :-1]

        # middle pt of a pair of samples
        colors_mid = (colors[:, :, :-1] + colors[:, :, 1:]) / 2
        densities_mid = (densities[:, :, :-1] + densities[:, :, 1:]) / 2
        depths_mid = (depths[:, :, :-1] + depths[:, :, 1:]) / 2

        # using factory mode for better usability
        densities_mid = mint.nn.functional.softplus(densities_mid - 1).to(dtype)
        density_delta = densities_mid * deltas

        alpha = 1 - mint.exp(-density_delta).to(dtype)

        alpha_shifted = mint.cat([mint.ones_like(alpha[:, :, :1]), 1 - alpha + 1e-10], -2)
        weights = alpha * ops.cumprod(alpha_shifted, -2)[:, :, :-1]
        weights = weights.to(dtype)

        if get_weigths:
            return weights

        composite_rgb = mint.sum(weights * colors_mid, -2)
        weight_total = weights.sum(2)
        composite_depth = mint.sum(weights * depths_mid, -2)

        # clip the composite to min/max range of depths
        composite_depth = ops.nan_to_num(composite_depth, float("inf")).to(dtype)

        min_val = mint.min(depths)
        max_val = mint.max(depths)
        composite_depth = mint.clamp(composite_depth, min_val, max_val)

        if self.white_back:
            composite_rgb = composite_rgb + 1 - weight_total

        # rendered value scale is 0-1, comment out original mipnerf scaling
        # composite_rgb = composite_rgb * 2 - 1 # Scale to (-1, 1)

        return composite_rgb, composite_depth, weights
