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
# Modified by Jiale Xu
# The modifications are subject to the same license as the original.
#
# This code is adapted from https://github.com/TencentARC/InstantMesh to work with MindSpore.


""" The ray marcher takes the raw output of the implicit representation and uses the volume rendering equation to produce composited colors and depths.
Based off of the implementation in MipNeRF (this one doesn't do any cone tracing though!)
"""
from typing import Dict, Optional

import mindspore as ms
import mindspore.nn as nn
from mindspore import mint, ops


class MipRayMarcher2(nn.Cell):
    def __init__(self, rendering_options: Optional[Dict] = None):
        super().__init__()
        self.white_back = rendering_options.get("white_back", False)

    def construct(
        self,
        colors: ms.Tensor,
        densities: ms.Tensor,
        depths: ms.Tensor,
    ):
        dtype = colors.dtype
        deltas = depths[:, :, 1:] - depths[:, :, :-1]
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

        composite_rgb = mint.sum(weights * colors_mid, -2)
        weight_total = weights.sum(2)
        # composite_depth = torch.sum(weights * depths_mid, -2) / weight_total
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
