import mindspore as ms
import mindspore.nn as nn
from mindspore import mint, ops


class RaySampler(nn.Cell):
    def __init__(self):
        super().__init__()
        self.ray_origins_h, self.ray_directions, self.depths, self.image_coords, self.rendering_options = (
            None,
            None,
            None,
            None,
            None,
        )

    def construct(self, cam2world_matrix: ms.Tensor, intrinsics: ms.Tensor, render_size: int):
        """
        Create batches of rays and return origins and directions.

        cam2world_matrix: (N, 4, 4)
        intrinsics: (N, 3, 3)

        ray_origins: (N, M, 3)
        ray_dirs: (N, M, 2)
        """
        dtype = cam2world_matrix.dtype
        N, M = cam2world_matrix.shape[0], render_size**2
        cam_locs_world = cam2world_matrix[:, :3, 3]
        fx = intrinsics[:, 0, 0]
        fy = intrinsics[:, 1, 1]
        cx = intrinsics[:, 0, 2]
        cy = intrinsics[:, 1, 2]
        sk = intrinsics[:, 0, 1]

        uv = mint.stack(
            ops.meshgrid(
                mint.arange(render_size, dtype=dtype),
                mint.arange(render_size, dtype=dtype),
                indexing="ij",
            )
        )  # FIXME mint.stack() builds graph mode fail, bypass with ops.stack first
        uv = mint.flip(uv, dims=(0,)).reshape((2, -1)).swapaxes(1, 0)
        uv = uv.unsqueeze(0).tile((cam2world_matrix.shape[0], 1, 1))

        x_cam = uv[:, :, 0].view(N, -1) * (1.0 / render_size) + (0.5 / render_size)
        y_cam = uv[:, :, 1].view(N, -1) * (1.0 / render_size) + (0.5 / render_size)
        z_cam = mint.ones((N, M), dtype=dtype)

        x_lift = (
            (
                x_cam
                - cx.unsqueeze(-1)
                + cy.unsqueeze(-1) * sk.unsqueeze(-1) / fy.unsqueeze(-1)
                - sk.unsqueeze(-1) * y_cam / fy.unsqueeze(-1)
            )
            / fx.unsqueeze(-1)
            * z_cam
        )
        y_lift = (y_cam - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z_cam

        cam_rel_points = mint.stack((x_lift, y_lift, z_cam, mint.ones_like(z_cam)), dim=-1).to(dtype)

        _opencv2blender = (
            ms.tensor(
                [
                    [1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1],
                ],
                dtype=dtype,
            )
            .unsqueeze(0)
            .tile((N, 1, 1))
        )

        cam2world_matrix = mint.bmm(cam2world_matrix, _opencv2blender)

        world_rel_points = mint.bmm(cam2world_matrix, cam_rel_points.permute(0, 2, 1)).permute(0, 2, 1)[:, :, :3]

        ray_dirs = world_rel_points - cam_locs_world[:, None, :]
        # ray_dirs = torch.nn.functional.normalize(ray_dirs, dim=2).to(dtype)
        # l2 norm
        ray_dirs_denom = ray_dirs.norm(2.0, 2, keepdim=True).clip(min=1e-12).broadcast_to(ray_dirs.shape)
        ray_dirs = ray_dirs / ray_dirs_denom

        ray_origins = cam_locs_world.unsqueeze(1).tile((1, ray_dirs.shape[1], 1))

        return ray_origins, ray_dirs
