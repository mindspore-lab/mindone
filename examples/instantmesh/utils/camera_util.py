import numpy as np

import mindspore as ms
from mindspore import Tensor, mint, ops


def pad_camera_extrinsics_4x4(extrinsics):
    if extrinsics.shape[-2] == 4:
        return extrinsics
    padding = Tensor([[0, 0, 0, 1]]).to(extrinsics.dtype)
    if extrinsics.ndim == 3:
        padding = padding.unsqueeze(0).tile((extrinsics.shape[0], 1, 1))
    extrinsics = mint.cat([extrinsics, padding], dim=-2)
    return extrinsics


def center_looking_at_camera_pose(camera_position: Tensor, look_at=None, up_world=None):
    """
    Create OpenGL camera extrinsics from camera locations and look-at position.

    camera_position: (M, 3) or (3,)
    look_at: (3)
    up_world: (3)
    return: Tensor, (M, 3, 4) or (3, 4)
    """
    # by default, looking at the origin and world up is z-axis
    if look_at is None:
        look_at = Tensor([0, 0, 0], dtype=ms.float32)
    if up_world is None:
        up_world = Tensor([0, 0, 1], dtype=ms.float32)
    if camera_position.ndim == 2:
        look_at = look_at.unsqueeze(0).tile((camera_position.shape[0], 1))
        up_world = up_world.unsqueeze(0).tile((camera_position.shape[0], 1))

    # OpenGL camera: z-backward, x-right, y-up
    z_axis = camera_position - look_at
    norm = ops.L2Normalize(axis=-1)
    z_axis = norm(z_axis)
    x_axis = mint.cross(up_world, z_axis, dim=-1)
    x_axis = norm(x_axis)
    y_axis = mint.cross(z_axis, x_axis, dim=-1)
    y_axis = norm(y_axis)
    print(f"zshape: {z_axis.shape}, xshape: {x_axis.shape}, yshape: {y_axis.shape}")

    extrinsics = mint.stack([x_axis, y_axis, z_axis, camera_position], dim=-1)
    print(f"fred: the extrinsics shape of {extrinsics.shape}")
    extrinsics = pad_camera_extrinsics_4x4(extrinsics)
    return extrinsics


def spherical_camera_pose(azimuths: np.ndarray, elevations: np.ndarray, radius=2.5):
    azimuths = np.deg2rad(azimuths)
    elevations = np.deg2rad(elevations)

    xs = radius * np.cos(elevations) * np.cos(azimuths)
    ys = radius * np.cos(elevations) * np.sin(azimuths)
    zs = radius * np.sin(elevations)

    cam_locations = np.stack([xs, ys, zs], axis=-1)
    cam_locations = Tensor.from_numpy(cam_locations).float()
    print(f"fred: camloc shape {cam_locations.shape}")

    c2ws = center_looking_at_camera_pose(cam_locations)
    return c2ws


def get_circular_camera_poses(M=120, radius=2.5, elevation=30.0):
    # M: number of circular views
    # radius: camera dist to center
    # elevation: elevation degrees of the camera
    # return: (M, 4, 4)
    assert M > 0 and radius > 0

    elevation = np.deg2rad(elevation)

    camera_positions = []
    for i in range(M):
        azimuth = 2 * np.pi * i / M
        x = radius * np.cos(elevation) * np.cos(azimuth)
        y = radius * np.cos(elevation) * np.sin(azimuth)
        z = radius * np.sin(elevation)
        camera_positions.append([x, y, z])
    camera_positions = np.array(camera_positions)
    camera_positions = Tensor.from_numpy(camera_positions).float()
    extrinsics = center_looking_at_camera_pose(camera_positions)
    return extrinsics


def FOV_to_intrinsics(fov):
    """
    Creates a 3x3 camera intrinsics matrix from the camera field of view, specified in degrees.
    Note the intrinsics are returned as normalized by image size, rather than in pixel units.
    Assumes principal point is at image center.
    """
    focal_length = 0.5 / np.tan(np.deg2rad(fov) * 0.5)
    intrinsics = np.array([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]])
    return intrinsics


def get_sv3d_input_cameras(bs=1, radius=4.0, fov=30.0):
    """
    Get the input camera parameters.
    """
    azimuths = np.array([360 / 21 * i for i in range(21)])
    elevations = np.array([0] * 21).astype(float)

    # tensor
    pose_cam2world = spherical_camera_pose(azimuths, elevations, radius)
    pose_cam2world = pose_cam2world.float().flatten(start_dim=-2)

    Ks = Tensor(FOV_to_intrinsics(fov)).unsqueeze(0).tile((21, 1, 1)).float().flatten(start_dim=-2)

    extrinsics = pose_cam2world[:, :12]
    intrinsics = mint.stack([Ks[:, 0], Ks[:, 4], Ks[:, 2], Ks[:, 5]], dim=-1)
    cameras = mint.cat([extrinsics, intrinsics], dim=-1)

    print(f"cameras dtype is {cameras.dtype}")

    return cameras.unsqueeze(0).tile((int(bs), 1, 1))
