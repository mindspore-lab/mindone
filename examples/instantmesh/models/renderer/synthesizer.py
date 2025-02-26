import itertools

import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import mint

from .utils.ray_sampler import RaySampler
from .utils.renderer import ImportanceRenderer


class OSGDecoder(nn.Cell):
    """
    Triplane decoder that gives RGB and sigma values from sampled features.
    Using ReLU here instead of Softplus in the original implementation.

    Reference:
    EG3D: https://github.com/NVlabs/eg3d/blob/main/eg3d/training/triplane.py#L112
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 64,
        num_layers: int = 4,
        activation: nn.Cell = nn.ReLU,
    ):
        super().__init__()
        self.net = nn.SequentialCell(
            nn.Dense(3 * n_features, hidden_dim),
            activation(),
            *itertools.chain(
                *[
                    [
                        nn.Dense(hidden_dim, hidden_dim),
                        activation(),
                    ]
                    for _ in range(num_layers - 2)
                ]
            ),
            nn.Dense(hidden_dim, 1 + 3),
        )
        # bias init as zero by default, can refer to ~/examples/stable_diffusion_v2/tests/test_lora.py & lora_torch.py for evidence

    # @ms.jit  # now has the error: Exceed function call depth limit 1000, (function call depth: 1001, simulate call depth: 508).
    def construct(self, sampled_features):
        # Aggregate features by mean
        # sampled_features = sampled_features.mean(1)
        # Aggregate features by concatenation
        _N, n_planes, _M, _C = sampled_features.shape
        sampled_features = sampled_features.permute(0, 2, 1, 3).reshape(_N, _M, n_planes * _C)
        x = sampled_features

        N, M, C = x.shape
        x = x.contiguous().view(N * M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        rgb = mint.sigmoid(x[..., 1:]) * (1 + 2 * 0.001) - 0.001  # Uses sigmoid clamping from MipNeRF
        sigma = x[..., 0:1]

        return rgb, sigma


class TriplaneSynthesizer(nn.Cell):
    """
    Synthesizer that renders a triplane volume with planes and a camera.

    Reference:
    EG3D: https://github.com/NVlabs/eg3d/blob/main/eg3d/training/triplane.py#L19
    """

    DEFAULT_RENDERING_KWARGS = {
        "ray_start": "auto",
        "ray_end": "auto",
        "box_warp": 2.0,
        "white_back": True,
        "disparity_space_sampling": False,
        "clamp_mode": "softplus",
        "sampler_bbox_min": -1.0,
        "sampler_bbox_max": 1.0,
    }

    def __init__(
        self, triplane_dim: int, samples_per_ray: int, dtype: ms.dtype = ms.float32, use_recompute: bool = False
    ):
        super().__init__()

        # attributes
        self.triplane_dim = triplane_dim
        dep_res = int(np.divmod(samples_per_ray, 2)[0])
        dep_res_imp = int(np.divmod(samples_per_ray, 2)[0])
        self.rendering_kwargs = {
            **self.DEFAULT_RENDERING_KWARGS,
            "depth_resolution": dep_res,
            "depth_resolution_importance": dep_res_imp,
        }

        # nerf decoder
        self.decoder = OSGDecoder(n_features=triplane_dim)

        # renderings
        self.renderer = ImportanceRenderer(self.rendering_kwargs, dtype=dtype, decoder=self.decoder)
        self.ray_sampler = RaySampler()

        if use_recompute:
            self.renderer.recompute()

    # @ms.jit  # now has the error in the renderer: Exceed function call depth limit 1000, (function call depth: 1001, simulate call depth: 508).
    def construct(self, planes, cameras, render_size, crop_params):
        # planes: (N, 3, D', H', W')
        # cameras: (N, M, D_cam)
        # render_size: int
        assert planes.shape[0] == cameras.shape[0], "Batch size mismatch for planes and cameras"
        N, M = cameras.shape[:2]

        cam2world_matrix = cameras[..., :16].view(N, M, 4, 4)
        intrinsics = cameras[..., 16:25].view(N, M, 3, 3)

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(
            cam2world_matrix=cam2world_matrix.reshape(-1, 4, 4),
            intrinsics=intrinsics.reshape(-1, 3, 3),
            render_size=render_size,
        )
        assert N * M == ray_origins.shape[0], "Batch size mismatch for ray_origins"
        assert ray_origins.dim() == 3, "ray_origins should be 3-dimensional"

        # Crop rays if crop_params is available
        if crop_params is not None:
            ray_origins = ray_origins.reshape(N * M, render_size, render_size, 3)
            ray_directions = ray_directions.reshape(N * M, render_size, render_size, 3)
            i, j, h, w = crop_params.tolist()[0]
            ray_origins = ray_origins[:, i : i + h, j : j + w, :].reshape(N * M, -1, 3)
            ray_directions = ray_directions[:, i : i + h, j : j + w, :].reshape(N * M, -1, 3)

        # Perform volume rendering
        rgb_samples, depth_samples, weights_samples = self.renderer(
            planes.repeat_interleave(M, dim=0), ray_origins, ray_directions
        )

        # Reshape into 'raw' neural-rendered image
        if crop_params is not None:
            Himg, Wimg = crop_params.tolist()[0][2:]
        else:
            Himg = Wimg = render_size
        rgb_images = (
            rgb_samples.permute(0, 2, 1).reshape(N, M, rgb_samples.shape[-1], Himg, Wimg).contiguous()
        )  # b n c h w
        depth_images = depth_samples.permute(0, 2, 1).reshape(N, M, 1, Himg, Wimg)
        weight_images = weights_samples.permute(0, 2, 1).reshape(N, M, 1, Himg, Wimg)

        # out = {
        #     'images_rgb': rgb_images,
        #     'images_depth': depth_images,
        #     'images_weight': weight_images,
        # }
        return rgb_images, depth_images, weight_images

    # [inference only] below two func shortcuts, not used in graph training: get_texture_prediction() & extract_mesh()
    # for run meshing code, not trained
    def forward_points(self, planes, points: ms.Tensor, chunk_size: int = 2**20):
        # planes: (N, 3, D', H', W')
        # points: (N, P, 3)
        N, P = points.shape[:2]

        # query triplane in chunks
        outs = []
        for i in range(0, points.shape[1], chunk_size):
            chunk_points = points[:, i : i + chunk_size]

            # query triplane
            chunk_out = self.renderer.run_model_activated(
                planes=planes,
                sample_coordinates=chunk_points,
            )
            outs.append(chunk_out)

        # concatenate the outputs
        point_features = {k: mint.cat([out[k] for out in outs], dim=1) for k in outs[0].keys()}
        return point_features

    def forward_grid(self, planes, grid_size: int, aabb: ms.Tensor = None):
        # planes: (N, 3, D', H', W')
        # grid_size: int
        # aabb: (N, 2, 3)
        if aabb is None:
            aabb = (
                ms.tensor(
                    [
                        [self.rendering_kwargs["sampler_bbox_min"]] * 3,
                        [self.rendering_kwargs["sampler_bbox_max"]] * 3,
                    ],
                    dtype=planes.dtype,
                )
                .unsqueeze(0)
                .repeat(planes.shape[0], 1, 1)
            )
        assert planes.shape[0] == aabb.shape[0], "Batch size mismatch for planes and aabb"
        N = planes.shape[0]

        # create grid points for triplane query
        grid_points = []
        for i in range(N):
            grid_points.append(
                mint.stack(
                    mint.meshgrid(
                        mint.linspace(aabb[i, 0, 0], aabb[i, 1, 0], grid_size),
                        mint.linspace(aabb[i, 0, 1], aabb[i, 1, 1], grid_size),
                        mint.linspace(aabb[i, 0, 2], aabb[i, 1, 2], grid_size),
                        indexing="ij",
                    ),
                    dim=-1,
                ).reshape(-1, 3)
            )
        cube_grid = mint.stack(grid_points, dim=0)

        features = self.forward_points(planes, cube_grid)

        # reshape into grid
        features = {k: v.reshape(N, grid_size, grid_size, grid_size, -1) for k, v in features.items()}
        return features
