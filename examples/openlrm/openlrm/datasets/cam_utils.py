# Copyright (c) 2023-2024, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math

import mindspore as ms
from mindspore import mint, ops

"""
R: (N, 3, 3)
T: (N, 3)
E: (N, 4, 4)
vector: (N, 3)
"""


def compose_extrinsic_R_T(R: ms.Tensor, T: ms.Tensor):
    """
    Compose the standard form extrinsic matrix from R and T.
    Batched I/O.
    """
    RT = mint.cat((R, T.unsqueeze(-1)), dim=-1)
    return compose_extrinsic_RT(RT)


def compose_extrinsic_RT(RT: ms.Tensor):
    """
    Compose the standard form extrinsic matrix from RT.
    Batched I/O.
    """
    return mint.cat([RT, ms.Tensor([[[0, 0, 0, 1]]], dtype=RT.dtype).tile((RT.shape[0], 1, 1))], dim=1)


def decompose_extrinsic_R_T(E: ms.Tensor):
    """
    Decompose the standard extrinsic matrix into R and T.
    Batched I/O.
    """
    RT = decompose_extrinsic_RT(E)
    return RT[:, :, :3], RT[:, :, 3]


def decompose_extrinsic_RT(E: ms.Tensor):
    """
    Decompose the standard extrinsic matrix into RT.
    Batched I/O.
    """
    return E[:, :3, :]


def camera_normalization_objaverse(normed_dist_to_center, poses: ms.Tensor, ret_transform: bool = False):
    assert normed_dist_to_center is not None
    pivotal_pose = compose_extrinsic_RT(poses[:1])  # 1x4x4
    if normed_dist_to_center == "auto":
        dist_to_center = pivotal_pose[:, :3, 3]  # 1x3
        dist_to_center = ops.norm(dist_to_center, dim=-1).item()
    else:
        dist_to_center = normed_dist_to_center

    # compute camera norm (new version)
    canonical_camera_extrinsics = ms.Tensor(
        [
            [
                [1, 0, 0, 0],
                [0, 0, -1, -dist_to_center],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ]
        ],
        dtype=ms.float32,
    )
    pivotal_pose_inv = mint.inverse(pivotal_pose)
    camera_norm_matrix = mint.bmm(canonical_camera_extrinsics, pivotal_pose_inv)

    # normalize all views
    poses = compose_extrinsic_RT(poses)
    poses = mint.bmm(camera_norm_matrix.tile((poses.shape[0], 1, 1)), poses)
    poses = decompose_extrinsic_RT(poses)

    if ret_transform:
        return poses, camera_norm_matrix.squeeze(0)
    return poses


def get_normalized_camera_intrinsics(intrinsics: ms.Tensor):
    """
    intrinsics: (N, 3, 2), [[fx, fy], [cx, cy], [width, height]]
    Return batched fx, fy, cx, cy
    """
    fx, fy = intrinsics[:, 0, 0], intrinsics[:, 0, 1]
    cx, cy = intrinsics[:, 1, 0], intrinsics[:, 1, 1]
    width, height = intrinsics[:, 2, 0], intrinsics[:, 2, 1]
    fx, fy = fx / width, fy / height
    cx, cy = cx / width, cy / height
    return fx, fy, cx, cy


def build_camera_principle(RT: ms.Tensor, intrinsics: ms.Tensor):
    """
    RT: (N, 3, 4)
    intrinsics: (N, 3, 2), [[fx, fy], [cx, cy], [width, height]]
    """
    fx, fy, cx, cy = get_normalized_camera_intrinsics(intrinsics)
    return mint.cat(
        [
            RT.reshape(-1, 12),
            fx.unsqueeze(-1),
            fy.unsqueeze(-1),
            cx.unsqueeze(-1),
            cy.unsqueeze(-1),
        ],
        dim=-1,
    )


def build_camera_standard(RT: ms.Tensor, intrinsics: ms.Tensor):
    """
    RT: (N, 3, 4)
    intrinsics: (N, 3, 2), [[fx, fy], [cx, cy], [width, height]]
    If convert it to intrinsics K:
    K = [[fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]]
    """
    Extr_M = compose_extrinsic_RT(RT)
    fx, fy, cx, cy = get_normalized_camera_intrinsics(intrinsics)
    Intr_M = mint.stack(
        [
            mint.stack([fx, mint.zeros_like(fx, dtype=ms.float32), cx], dim=-1),
            mint.stack([mint.zeros_like(fy, dtype=ms.float32), fy, cy], dim=-1),
            ms.Tensor([[0, 0, 1]], dtype=ms.float32).tile((RT.shape[0], 1)),
        ],
        dim=1,
    )
    return mint.cat(
        [
            Extr_M.reshape(-1, 16),
            Intr_M.reshape(-1, 9),
        ],
        dim=-1,
    )


def center_looking_at_camera_pose(
    camera_position: ms.Tensor,
    look_at: ms.Tensor = None,
    up_world: ms.Tensor = None,
):
    """
    camera_position: (M, 3)
    look_at: (3)
    up_world: (3)
    return: (M, 3, 4)
    """
    # by default, looking at the origin and world up is pos-z
    if look_at is None:
        look_at = ms.Tensor([0, 0, 0], dtype=ms.float32)
    if up_world is None:
        up_world = ms.Tensor([0, 0, 1], dtype=ms.float32)
    look_at = look_at.unsqueeze(0).tile((camera_position.shape[0], 1))
    up_world = up_world.unsqueeze(0).tile((camera_position.shape[0], 1))

    z_axis = camera_position - look_at
    z_axis = z_axis / z_axis.norm(dim=-1, keepdim=True)
    x_axis = ops.cross(up_world, z_axis)
    x_axis = x_axis / x_axis.norm(dim=-1, keepdim=True)
    y_axis = ops.cross(z_axis, x_axis)
    y_axis = y_axis / y_axis.norm(dim=-1, keepdim=True)
    extrinsics = mint.stack([x_axis, y_axis, z_axis, camera_position], dim=-1)
    return extrinsics


def surrounding_views_linspace(n_views: int, radius: float = 2.0, height: float = 0.8):
    """
    n_views: number of surrounding views
    radius: camera dist to center
    height: height of the camera
    return: (M, 3, 4)
    """
    assert n_views > 0
    assert radius > 0

    theta = ops.linspace(-ms.numpy.pi / 2, 3 * ms.numpy.pi / 2, n_views)
    projected_radius = math.sqrt(radius**2 - height**2)
    x = ops.cos(theta) * projected_radius
    y = ops.sin(theta) * projected_radius
    z = ops.full((n_views,), height, dtype=ms.float32)

    camera_positions = mint.stack([x, y, z], dim=1)
    extrinsics = center_looking_at_camera_pose(camera_positions)

    return extrinsics


def create_intrinsics(
    f: float,
    c: float = None,
    cx: float = None,
    cy: float = None,
    w: float = 1.0,
    h: float = 1.0,
    dtype: ms.dtype = ms.float32,
):
    """
    return: (3, 2)
    """
    fx = fy = f
    if c is not None:
        assert cx is None and cy is None, "c and cx/cy cannot be used together"
        cx = cy = c
    else:
        assert cx is not None and cy is not None, "cx/cy must be provided when c is not provided"
    fx, fy, cx, cy, w, h = fx / w, fy / h, cx / w, cy / h, 1.0, 1.0
    intrinsics = ms.Tensor(
        [
            [fx, fy],
            [cx, cy],
            [w, h],
        ],
        dtype=dtype,
    )
    return intrinsics
