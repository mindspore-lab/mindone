from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import threestudio
from omegaconf import DictConfig
from threestudio.models.geometry.base import BaseGeometry, BaseImplicitGeometry, contract_to_unisphere
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.ops import get_activation

import mindspore as ms
from mindspore import Tensor, mint, ops


@threestudio.register("implicit-volume")
class ImplicitVolume(BaseImplicitGeometry):
    @dataclass
    class Config(BaseImplicitGeometry.Config):
        n_input_dims: int = 3
        n_feature_dims: int = 3
        density_activation: Optional[str] = "softplus"
        density_bias: Union[float, str] = "blob_magic3d"
        density_blob_scale: float = 10.0
        density_blob_std: float = 0.5
        pos_encoding_config: dict = field(
            default_factory=lambda: {
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.447269237440378,
            }
        )
        mlp_network_config: dict = field(
            default_factory=lambda: {
                "otype": "VanillaMLP",
                "activation": "ReLU",
                "output_activation": "none",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            }
        )
        normal_type: Optional[
            str
        ] = "finite_difference"  # in ['pred', 'finite_difference', 'finite_difference_laplacian']
        finite_difference_normal_eps: float = 0.01

        # automatically determine the threshold
        isosurface_threshold: Union[float, str] = 25.0

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.encoding = get_encoding(self.cfg.n_input_dims, self.cfg.pos_encoding_config)
        self.density_network = get_mlp(self.encoding.n_output_dims, 1, self.cfg.mlp_network_config)
        if self.cfg.n_feature_dims > 0:
            self.feature_network = get_mlp(
                self.encoding.n_output_dims,
                self.cfg.n_feature_dims,
                self.cfg.mlp_network_config,
            )
        if self.cfg.normal_type == "pred":
            self.normal_network = get_mlp(self.encoding.n_output_dims, 3, self.cfg.mlp_network_config)

    def get_activated_density(
        self, points: Tensor, density: Tensor  # N Di  # N 1
    ) -> Tuple[Tensor, Tensor]:  # N 1, N 1
        density_bias: Tensor  # N 1
        # psum = (points**2).sum(axis=-1)
        if self.cfg.density_bias == "blob_dreamfusion":
            # pre-activation density bias
            density_bias = (
                self.cfg.density_blob_scale
                * mint.exp(-0.5 * (points**2).sum(axis=-1) / self.cfg.density_blob_std**2)[..., None]
            )
        elif self.cfg.density_bias == "blob_magic3d":
            # pre-activation density bias
            # density_bias = mint.ones_like(psum) - mint.sqrt(psum) / self.cfg.density_blob_std
            # density_bias = self.cfg.density_blob_scale * density_bias
            # density_bias = density_bias[..., None]
            density_bias = (
                self.cfg.density_blob_scale
                * (1 - mint.sqrt((points**2).sum(axis=-1)) / self.cfg.density_blob_std)[..., None]
            )
            # density_bias = density_bias.unsqueeze(-1)
        elif isinstance(self.cfg.density_bias, float):
            density_bias = self.cfg.density_bias
        else:
            raise ValueError(f"Unknown density bias {self.cfg.density_bias}")
        raw_density: Tensor = density + density_bias
        density = get_activation(self.cfg.density_activation)(raw_density)
        return raw_density, density

    def construct(self, points: Tensor, output_normal: bool = False) -> Dict:  # N Di
        if output_normal and self.cfg.normal_type == "analytic":
            self.requires_grad(True)
            points.requires_grad = True

        points_unscaled = points  # points in the original scale
        points = contract_to_unisphere(points, self.bbox, self.unbounded)  # points normalized to (0, 1)

        enc = self.encoding(points.view(-1, self.cfg.n_input_dims))
        density = self.density_network(enc).view(*points.shape[:-1], 1)
        raw_density, density = self.get_activated_density(points_unscaled, density)

        output = {
            "density": density,
        }

        if self.cfg.n_feature_dims > 0:
            features = self.feature_network(enc).view(*points.shape[:-1], self.cfg.n_feature_dims)
            output.update({"features": features})

        if output_normal:
            if self.cfg.normal_type == "finite_difference" or self.cfg.normal_type == "finite_difference_laplacian":
                # TODO: use raw density
                eps = self.cfg.finite_difference_normal_eps
                if self.cfg.normal_type == "finite_difference_laplacian":
                    offsets = Tensor(  # 6 3
                        [
                            [eps, 0.0, 0.0],
                            [-eps, 0.0, 0.0],
                            [0.0, eps, 0.0],
                            [0.0, -eps, 0.0],
                            [0.0, 0.0, eps],
                            [0.0, 0.0, -eps],
                        ]
                    ).to(points_unscaled)
                    points_offset: Tensor = (points_unscaled[..., None, :] + offsets).clamp(
                        -self.cfg.radius, self.cfg.radius
                    )
                    density_offset: Tensor = self.forward_density(points_offset)
                    normal = -0.5 * (density_offset[..., 0::2, 0] - density_offset[..., 1::2, 0]) / eps
                else:
                    offsets = Tensor([[eps, 0.0, 0.0], [0.0, eps, 0.0], [0.0, 0.0, eps]]).to(points_unscaled)
                    points_offset: Tensor = (points_unscaled[..., None, :] + offsets).clamp(
                        -self.cfg.radius, self.cfg.radius
                    )
                    density_offset: Tensor = self.forward_density(points_offset)
                    normal = (density_offset[..., 0::1, 0] - density) / eps
                normal = ops.L2Normalize(normal)
            elif self.cfg.normal_type == "pred":
                normal = self.normal_network(enc).view(*points.shape[:-1], 3)
                normal = ops.L2Normalize(normal)
            elif self.cfg.normal_type == "analytic":
                normal = -ops.grad(
                    density,
                    points_unscaled,
                    grad_outputs=mint.ones_like(density),
                    create_graph=True,
                )[0]
                normal = ops.L2Normalize(normal)
            else:
                raise AttributeError(f"Unknown normal type {self.cfg.normal_type}")
            output.update({"normal": normal, "shading_normal": normal})

        return output

    def forward_density(self, points: Tensor) -> Tensor:
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox, self.unbounded)

        density = self.density_network(self.encoding(points.reshape(-1, self.cfg.n_input_dims))).reshape(
            *points.shape[:-1], 1
        )

        _, density = self.get_activated_density(points_unscaled, density)
        return density

    def forward_field(self, points: Tensor) -> Tuple[Tensor, None]:  # N Di
        if self.cfg.isosurface_deformable_grid:
            threestudio.info(f"{self.__class__.__name__} does not support isosurface_deformable_grid. Ignoring.")
        density = self.forward_density(points)
        return density, None

    def forward_level(self, field: Tensor, threshold: float) -> Tensor:  # Float[Tensor, "*N 1"]
        return -(field - threshold)

    def export(self, points: Tensor, **kwargs) -> Dict[str, Any]:  # N Di
        out: Dict[str, Any] = {}
        if self.cfg.n_feature_dims == 0:
            return out
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox, self.unbounded)
        enc = self.encoding(points.reshape(-1, self.cfg.n_input_dims))
        features = self.feature_network(enc).view(*points.shape[:-1], self.cfg.n_feature_dims)
        out.update(
            {
                "features": features,
            }
        )
        return out

    @staticmethod
    @ms._no_grad()
    def create_from(
        other: BaseGeometry,
        cfg: Optional[Union[dict, DictConfig]] = None,
        copy_net: bool = True,
        **kwargs,
    ) -> "ImplicitVolume":
        if isinstance(other, ImplicitVolume):
            instance = ImplicitVolume(cfg, **kwargs)
            instance.encoding.load_state_dict(other.encoding.state_dict())
            instance.density_network.load_state_dict(other.density_network.state_dict())
            if copy_net:
                if instance.cfg.n_feature_dims > 0 and other.cfg.n_feature_dims == instance.cfg.n_feature_dims:
                    instance.feature_network.load_state_dict(other.feature_network.state_dict())
                if instance.cfg.normal_type == "pred" and other.cfg.normal_type == "pred":
                    instance.normal_network.load_state_dict(other.normal_network.state_dict())
            return instance
        else:
            raise TypeError(f"Cannot create {ImplicitVolume.__name__} from {other.__class__.__name__}")
