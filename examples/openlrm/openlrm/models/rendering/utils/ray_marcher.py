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
# Modified by Zexin He in 2023-2024.
# The modifications are subject to the same license as the original.


"""
The ray marcher takes the raw output of the implicit representation and uses the volume rendering equation to produce composited colors and depths.
Based off of the implementation in MipNeRF (this one doesn't do any cone tracing though!)
"""

import mindspore as ms
from mindspore import mint, nn, ops

from . import CumProd, NanToNum


class MipRayMarcher2(nn.Cell):
    def __init__(self, activation_factory):
        super().__init__()
        self.activation_factory = activation_factory
        self.cum_prod = CumProd()
        self.nan_to_num = NanToNum()

    def run_construct(self, colors, densities, depths, rendering_options, bg_colors=None):
        deltas = depths[:, :, 1:] - depths[:, :, :-1]
        colors_mid = (colors[:, :, :-1] + colors[:, :, 1:]) / 2
        densities_mid = (densities[:, :, :-1] + densities[:, :, 1:]) / 2
        depths_mid = (depths[:, :, :-1] + depths[:, :, 1:]) / 2

        # using factory mode for better usability
        densities_mid = self.activation_factory(rendering_options)(densities_mid)

        density_delta = densities_mid * deltas

        alpha = 1 - ops.exp(-density_delta)

        alpha_shifted = mint.cat([mint.ones_like(alpha[:, :, :1], dtype=ms.float32), 1 - alpha + 1e-10], -2)
        # weights = alpha * ops.cumprod(alpha_shifted, -2)[:, :, :-1]
        weights = alpha * self.cum_prod(alpha_shifted, -2)[:, :, :-1]

        composite_rgb = mint.sum(weights * colors_mid, -2)
        weight_total = mint.sum(weights, 2)
        composite_depth = mint.sum(weights * depths_mid, -2) / weight_total

        # clip the composite to min/max range of depths
        # composite_depth = ops.nan_to_num(composite_depth, float("inf"))
        composite_depth = self.nan_to_num(composite_depth, float("inf"))
        composite_depth = mint.clamp(composite_depth, mint.min(depths), mint.max(depths))

        if rendering_options.get("white_back", False):
            composite_rgb = composite_rgb + 1 - weight_total
        else:
            assert bg_colors is not None, "Must provide bg_colors if white_back is False"
            composite_rgb = composite_rgb + bg_colors.unsqueeze(-1) * (1 - weight_total)

        # rendered value scale is 0-1, comment out original mipnerf scaling
        # composite_rgb = composite_rgb * 2 - 1 # Scale to (-1, 1)

        return composite_rgb, composite_depth, weights

    def construct(self, colors, densities, depths, rendering_options, bg_colors=None):
        composite_rgb, composite_depth, weights = self.run_construct(
            colors, densities, depths, rendering_options, bg_colors=bg_colors
        )

        return composite_rgb, composite_depth, weights
