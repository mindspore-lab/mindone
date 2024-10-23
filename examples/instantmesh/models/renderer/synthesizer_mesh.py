import itertools

import mindspore.nn as nn
from mindspore import ops

from .utils.renderer import generate_planes, sample_from_planes


class OSGDecoder(nn.Cell):
    """
    Triplane decoder that gives RGB and sigma values from sampled features.
    Using ReLU here instead of Softplus in the original implementation.

    Reference:
    EG3D: https://github.com/NVlabs/eg3d/blob/main/eg3d/training/triplane.py#L112
    """

    def __init__(self, n_features: int, hidden_dim: int = 64, num_layers: int = 4, activation: nn.Cell = nn.ReLU):
        super().__init__()

        self.net_sdf = nn.SequentialCell(
            nn.Dense(3 * n_features, hidden_dim, bias_init="zeros"),
            activation(),
            *itertools.chain(
                *[
                    [
                        nn.Dense(hidden_dim, hidden_dim, bias_init="zeros"),
                        activation(),
                    ]
                    for _ in range(num_layers - 2)
                ]
            ),
            nn.Dense(hidden_dim, 1, bias_init="zeros"),
        )
        self.net_rgb = nn.SequentialCell(
            nn.Dense(3 * n_features, hidden_dim, bias_init="zeros"),
            activation(),
            *itertools.chain(
                *[
                    [
                        nn.Dense(hidden_dim, hidden_dim, bias_init="zeros"),
                        activation(),
                    ]
                    for _ in range(num_layers - 2)
                ]
            ),
            nn.Dense(hidden_dim, 3, bias_init="zeros"),
        )
        self.net_deformation = nn.SequentialCell(
            nn.Dense(3 * n_features, hidden_dim, bias_init="zeros"),
            activation(),
            *itertools.chain(
                *[
                    [
                        nn.Dense(hidden_dim, hidden_dim, bias_init="zeros"),
                        activation(),
                    ]
                    for _ in range(num_layers - 2)
                ]
            ),
            nn.Dense(hidden_dim, 3, bias_init="zeros"),
        )
        self.net_weight = nn.SequentialCell(
            nn.Dense(8 * 3 * n_features, hidden_dim, bias_init="zeros"),
            activation(),
            *itertools.chain(
                *[
                    [
                        nn.Dense(hidden_dim, hidden_dim, bias_init="zeros"),
                        activation(),
                    ]
                    for _ in range(num_layers - 2)
                ]
            ),
            nn.Dense(hidden_dim, 21, bias_init="zeros"),
        )

    def get_geometry_prediction(self, sampled_features, flexicubes_indices):
        _N, n_planes, _M, _C = sampled_features.shape
        sampled_features = sampled_features.permute(0, 2, 1, 3).reshape(_N, _M, n_planes * _C)

        sdf = self.net_sdf(sampled_features)
        deformation = self.net_deformation(sampled_features)

        grid_features = ops.index_select(input=sampled_features, index=flexicubes_indices.reshape(-1), axis=1)
        grid_features = grid_features.reshape(
            sampled_features.shape[0],
            flexicubes_indices.shape[0],
            flexicubes_indices.shape[1] * sampled_features.shape[-1],
        )
        weight = self.net_weight(grid_features) * 0.1

        return sdf, deformation, weight

    def get_texture_prediction(self, sampled_features):
        _N, n_planes, _M, _C = sampled_features.shape
        sampled_features = sampled_features.permute(0, 2, 1, 3).reshape(_N, _M, n_planes * _C)

        rgb = self.net_rgb(sampled_features)
        rgb = ops.sigmoid(rgb) * (1 + 2 * 0.001) - 0.001  # Uses sigmoid clamping from MipNeRF

        return rgb


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

    def __init__(self, triplane_dim: int, samples_per_ray: int):
        super().__init__()

        # attributes
        self.triplane_dim = triplane_dim
        self.rendering_kwargs = {
            **self.DEFAULT_RENDERING_KWARGS,
            "depth_resolution": samples_per_ray // 2,
            "depth_resolution_importance": samples_per_ray // 2,
        }

        # modules
        self.plane_axes = generate_planes()
        self.decoder = OSGDecoder(n_features=triplane_dim)

    def get_geometry_prediction(self, planes, sample_coordinates, flexicubes_indices):
        plane_axes = self.plane_axes
        sampled_features = sample_from_planes(
            plane_axes, planes, sample_coordinates, padding_mode="zeros", box_warp=self.rendering_kwargs["box_warp"]
        )

        sdf, deformation, weight = self.decoder.get_geometry_prediction(sampled_features, flexicubes_indices)
        return sdf, deformation, weight

    def get_texture_prediction(self, planes, sample_coordinates):
        plane_axes = self.plane_axes
        sampled_features = sample_from_planes(
            plane_axes, planes, sample_coordinates, padding_mode="zeros", box_warp=self.rendering_kwargs["box_warp"]
        )

        rgb = self.decoder.get_texture_prediction(sampled_features)
        return rgb
