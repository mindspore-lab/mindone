# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import mindspore as ms
from mindspore import mint, nn, ops


# only support fp16, fp32, not support bf16
class CumProd(nn.Cell):
    def construct(self, x, dim):
        return ops.cumprod(x, dim=dim)


# only support fp16, fp32, not support bf16
class NanToNum(nn.Cell):
    def construct(self, x, nan: float = 0.0):
        return ops.nan_to_num(x, nan)


# inv only support float32
class MatrixInv(nn.Cell):
    def construct(self, x):
        if x.dtype != ms.float32:
            x = x.float()
        return mint.linalg.inv(x)


# only support fp16, fp32, not support bf16
class GridSample(nn.Cell):
    def construct(
        self,
        plane_features,
        projected_coordinates,
        mode,
        padding_mode,
        align_corners,
    ):
        x = mint.nn.functional.grid_sample(
            plane_features,
            projected_coordinates,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )
        return x


# not support bf16
class MeshGrid(nn.Cell):
    def construct(self, x, y):
        uv = mint.stack(
            ops.meshgrid(
                mint.arange(x, dtype=ms.float32),
                mint.arange(y, dtype=ms.float32),
                indexing="ij",
            )
        )
        return uv


# not support bf16
class SearchSorted(nn.Cell):
    def construct(self, seq, values, right):
        inds = mint.searchsorted(seq, values, right=right)
        return inds
