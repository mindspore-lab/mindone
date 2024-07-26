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


"""
The renderer is a module that takes in rays, decides where to sample along each
ray, and computes pixel colors using the volume rendering equation.
"""
import mindspore as ms
from mindspore import ops


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
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).expand((-1, n_planes, -1, -1)).reshape(N * n_planes, M, 3)
    inv_planes = ops.inverse(planes).unsqueeze(0).expand((N, -1, -1, -1)).reshape(N * n_planes, 3, 3)
    projections = ops.bmm(coordinates, inv_planes)
    return projections[..., :2]


def sample_from_planes(plane_axes, plane_features, coordinates, mode="bilinear", padding_mode="zeros", box_warp=None):
    assert padding_mode == "zeros"
    N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape
    plane_features = plane_features.view(N * n_planes, C, H, W)
    dtype = plane_features.dtype

    coordinates = (2 / box_warp) * coordinates  # add specific box bounds

    projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1)
    output_features = (
        ops.grid_sample(
            plane_features,
            projected_coordinates.to(dtype),
            mode=mode,
            padding_mode=padding_mode,
            align_corners=False,
        )
        .permute(0, 3, 2, 1)
        .reshape(N, n_planes, M, C)
    )
    return output_features
