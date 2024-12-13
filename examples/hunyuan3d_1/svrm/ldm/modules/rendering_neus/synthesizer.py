# ORIGINAL LICENSE
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Modified by Zexin He
# The modifications are subject to the same license as the original.


import itertools

# from ...utils.ops import get_rank
from typing import Optional

import mindspore as ms
from mindspore import mint, nn, ops
from mindspore.common.initializer import Zero, initializer

from .utils.ray_sampler import RaySampler
from .utils.renderer import ImportanceRenderer, sample_from_planes


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
        num_layers: int = 2,
        activation: nn.Cell = nn.ReLU,
        sdf_bias="sphere",
        sdf_bias_params=0.5,
        output_normal=True,
        normal_type="finite_difference",
        dtype=ms.float16,
    ):
        super().__init__()
        self.sdf_bias = sdf_bias
        self.sdf_bias_params = sdf_bias_params
        self.output_normal = output_normal
        self.normal_type = normal_type
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
        ).to_float(dtype)
        # init all bias to zero
        for m in self.cells():
            if isinstance(m, nn.Dense):
                m.bias.set_data(initializer(Zero(), m.bias.shape, m.bias.dtype))

    def construct(self, ray_directions, sample_coordinates, plane_axes, planes, options):
        # Aggregate features by mean
        # sampled_features = sampled_features.mean(1)
        # Aggregate features by concatenation
        # self.set_train(True)
        # sample_coordinates.requires_grad = True

        sampled_features = sample_from_planes(
            plane_axes, planes, sample_coordinates, padding_mode="zeros", box_warp=options["box_warp"]
        )

        _N, n_planes, _M, _C = sampled_features.shape
        sampled_features = sampled_features.permute((0, 2, 1, 3)).reshape(_N, _M, n_planes * _C)
        x = sampled_features

        N, M, C = x.shape
        # x = x.contiguous().view(N*M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        rgb = mint.sigmoid(x[..., 1:]) * (1 + 2 * 0.001) - 0.001  # Uses sigmoid clamping from MipNeRF

        sdf = x[..., 0:1]
        # import ipdb; ipdb.set_trace()
        # print(f'sample_coordinates shape: {sample_coordinates.shape}')
        # sdf = self.get_shifted_sdf(sample_coordinates, sdf)

        # calculate normal
        eps = 0.01
        offsets = ms.Tensor([[eps, 0.0, 0.0], [0.0, eps, 0.0], [0.0, 0.0, eps]]).to(sample_coordinates.dtype)
        points_offset = (sample_coordinates[..., None, :] + offsets).clamp(  # Float[Tensor, "... 3 3"]
            options["sampler_bbox_min"], options["sampler_bbox_max"]
        )

        sdf_offset_list = [
            self.forward_sdf(plane_axes, planes, points_offset[:, :, i, :], options).unsqueeze(-2)
            for i in range(points_offset.shape[-2])
        ]  # Float[Tensor, "... 3 1"]

        sdf_offset = mint.cat(sdf_offset_list, -2)
        sdf_grad = (sdf_offset[..., 0::1, 0] - sdf) / eps

        normal = (sdf_grad / ops.norm(sdf_grad, dim=-1, keepdim=True)).to(sdf.dtype)
        return {"rgb": rgb, "sdf": sdf, "normal": normal, "sdf_grad": sdf_grad}

    def forward_sdf(self, plane_axes, planes, points_offset, options):
        sampled_features = sample_from_planes(
            plane_axes, planes, points_offset, padding_mode="zeros", box_warp=options["box_warp"]
        )
        _N, n_planes, _M, _C = sampled_features.shape
        sampled_features = sampled_features.permute((0, 2, 1, 3)).reshape(_N, _M, n_planes * _C)
        x = sampled_features

        N, M, C = x.shape
        # x = x.contiguous().view(N*M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        sdf = x[..., 0:1]
        # sdf = self.get_shifted_sdf(points_offset, sdf)
        return sdf

    def get_shifted_sdf(self, points, sdf):
        if self.sdf_bias == "sphere":
            assert isinstance(self.sdf_bias_params, float)
            radius = self.sdf_bias_params
            sdf_bias = (points**2).sum(dim=-1, keepdim=True).sqrt() - radius
        else:
            raise ValueError(f"Unknown sdf bias {self.cfg.sdf_bias}")
        return sdf + sdf_bias.to(sdf.dtype)


class TriplaneSynthesizer(nn.Cell):
    """
    Synthesizer that renders a triplane volume with planes and a camera.

    Reference:
    EG3D: https://github.com/NVlabs/eg3d/blob/main/eg3d/training/triplane.py#L19
    """

    DEFAULT_RENDERING_KWARGS = {
        "ray_start": "auto",
        "ray_end": "auto",
        "box_warp": 1.2,
        # 'box_warp': 1.,
        "white_back": True,
        "disparity_space_sampling": False,
        "clamp_mode": "softplus",
        # 'sampler_bbox_min': -1,
        # 'sampler_bbox_max': 1.,
        "sampler_bbox_min": -0.6,
        "sampler_bbox_max": 0.6,
    }
    print("DEFAULT_RENDERING_KWARGS")
    print(DEFAULT_RENDERING_KWARGS)

    def __init__(self, triplane_dim: int, samples_per_ray: int, osg_decoder="default"):
        super().__init__()

        # attributes
        self.triplane_dim = triplane_dim
        self.rendering_kwargs = {
            **self.DEFAULT_RENDERING_KWARGS,
            "depth_resolution": samples_per_ray,
            "depth_resolution_importance": 0
            # 'depth_resolution': samples_per_ray // 2,
            # 'depth_resolution_importance': samples_per_ray // 2,
        }

        # renderings
        self.renderer = ImportanceRenderer()
        self.ray_sampler = RaySampler()
        # modules
        if osg_decoder == "default":
            self.decoder = OSGDecoder(n_features=triplane_dim)
        else:
            raise NotImplementedError

    def to(self, dtype: Optional[ms.Type] = None):
        for p in self.get_parameters():
            p.set_dtype(dtype)
        return self

    def construct(self, planes, ray_origins, ray_directions, render_size, bgcolor=None):
        # planes: (N, 3, D', H', W')
        # render_size: int
        assert ray_origins.ndim == 3, "ray_origins should be 3-dimensional"

        # Perform volume rendering
        rgb_samples, depth_samples, weights_samples, sdf_grad, normal_samples = self.renderer(
            planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs, bgcolor
        )
        N = planes.shape[0]

        # zhaohx : add for normals
        # normal_samples = F.normalize(normal_samples, dim=-1)
        normal_samples = normal_samples / ops.norm(normal_samples, dim=-1)
        normal_samples = (normal_samples + 1.0) / 2.0  # for visualization
        normal_samples = ops.lerp(mint.zeros_like(normal_samples), normal_samples, weights_samples)

        # Reshape into 'raw' neural-rendered image
        Himg = Wimg = render_size
        rgb_images = rgb_samples.permute((0, 2, 1)).reshape(N, rgb_samples.shape[-1], Himg, Wimg).contiguous()
        depth_images = depth_samples.permute((0, 2, 1)).reshape(N, 1, Himg, Wimg)
        weight_images = weights_samples.permute((0, 2, 1)).reshape(N, 1, Himg, Wimg)

        # zhaohx : add for normals
        normal_images = normal_samples.permute((0, 2, 1)).reshape(N, normal_samples.shape[-1], Himg, Wimg).contiguous()

        # return {
        #     'images_rgb': rgb_images,
        #     'images_depth': depth_images,
        #     'images_weight': weight_images,
        # }

        return {
            "comp_rgb": rgb_images,
            "comp_depth": depth_images,
            "opacity": weight_images,
            "sdf_grad": sdf_grad,
            "comp_normal": normal_images,
        }
        # 输出normal的话在这个return里加

    def forward_grid(self, planes, grid_size: int, aabb: ms.Tensor = None):
        # planes: (N, 3, D', H', W')
        # grid_size: int
        # aabb: (N, 2, 3)
        if aabb is None:
            aabb = (
                ms.Tensor(
                    [
                        [self.rendering_kwargs["sampler_bbox_min"]] * 3,
                        [self.rendering_kwargs["sampler_bbox_max"]] * 3,
                    ]
                )
                .unsqueeze(0)
                .tile((planes.shape[0], 1, 1))
            )
        assert planes.shape[0] == aabb.shape[0], "Batch size mismatch for planes and aabb"
        aabb = aabb.float()  # convert to float32 since ops.linspace only support float32/float64
        N = planes.shape[0]

        # create grid points for triplane query
        grid_points = []
        for i in range(N):
            grid_points.append(
                mint.stack(
                    ops.meshgrid(
                        ops.linspace(aabb[i, 0, 0], aabb[i, 1, 0], grid_size),
                        ops.linspace(aabb[i, 0, 1], aabb[i, 1, 1], grid_size),
                        ops.linspace(aabb[i, 0, 2], aabb[i, 1, 2], grid_size),
                        indexing="ij",
                    ),
                    dim=-1,
                ).reshape(-1, 3)
            )
        cube_grid = mint.stack(grid_points, dim=0).to(planes.dtype)

        features = self.forward_points(planes, cube_grid)

        # reshape into grid
        features = {k: v.reshape(N, grid_size, grid_size, grid_size, -1) for k, v in features.items()}
        return features

    def forward_points(self, planes, points: ms.Tensor, chunk_size: int = 2**20):
        # planes: (N, 3, D', H', W')
        # points: (N, P, 3)
        N, P = points.shape[:2]

        # query triplane in chunks
        outs = []
        for i in range(0, points.shape[1], chunk_size):
            chunk_points = points[:, i : i + chunk_size]

            # query triplane
            # chunk_out = self.renderer.run_model_activated(
            chunk_out = self.renderer.run_model(
                planes=planes,
                decoder=self.decoder,
                sample_coordinates=chunk_points,
                sample_directions=mint.zeros_like(chunk_points),
                options=self.rendering_kwargs,
            )
            outs.append(chunk_out)

        # concatenate the outputs
        point_features = {k: mint.cat([out[k] for out in outs], dim=1) for k in outs[0].keys()}
        return point_features
