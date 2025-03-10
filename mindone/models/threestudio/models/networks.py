import math

from nerfstudio_ms.encoding import HashEncoding, SHEncoding
from threestudio.utils.base import Updateable
from threestudio.utils.config import config_to_primitive
from threestudio.utils.ops import get_activation

from mindspore import mint, nn


class CompositeEncoding(nn.Cell, Updateable):
    def __init__(self, encoding, n_input_dims, n_output_dims, include_xyz=False, xyz_scale=2.0, xyz_offset=-1.0):
        super(CompositeEncoding, self).__init__()
        self.encoding = encoding
        self.include_xyz, self.xyz_scale, self.xyz_offset = (
            include_xyz,
            xyz_scale,
            xyz_offset,
        )
        self.n_output_dims = int(self.include_xyz) * n_input_dims + n_output_dims

    def construct(self, x, *args):
        return (
            self.encoding(x, *args)
            if not self.include_xyz
            else mint.cat([x * self.xyz_scale + self.xyz_offset, self.encoding(x, *args)], dim=-1)
        )


def get_encoding(n_input_dims: int, config) -> nn.Cell:
    # input suppose to be range [0, 1]
    encoding: nn.Cell
    cfg = config_to_primitive(config)
    if config.otype == "HashGrid":
        encoding = HashEncoding(
            num_levels=cfg["n_levels"],
            features_per_level=cfg["n_features_per_level"],
            log2_hashmap_size=cfg["log2_hashmap_size"],
            min_res=cfg["base_resolution"],
            hash_init_scale=cfg["per_level_scale"],
        )
        n_output_dims = 32
    elif config.otype == "SphericalHarmonics":
        encoding = SHEncoding(levels=cfg["degree"])
        n_output_dims = 9
    else:
        raise ValueError

    encoding = CompositeEncoding(
        encoding,
        n_input_dims,
        n_output_dims=n_output_dims,  # from tcnn
        include_xyz=config.get("include_xyz", False),
        xyz_scale=2.0,
        xyz_offset=-1.0,
    )  # FIXME: hard coded
    return encoding


class VanillaMLP(nn.Cell):
    def __init__(self, dim_in: int, dim_out: int, config: dict):
        super().__init__()
        self.n_neurons, self.n_hidden_layers = (
            config["n_neurons"],
            config["n_hidden_layers"],
        )
        layers = [
            self.make_linear(dim_in, self.n_neurons, is_first=True, is_last=False),
            self.make_activation(),
        ]
        for i in range(self.n_hidden_layers - 1):
            layers += [
                self.make_linear(self.n_neurons, self.n_neurons, is_first=False, is_last=False),
                self.make_activation(),
            ]
        layers += [self.make_linear(self.n_neurons, dim_out, is_first=False, is_last=True)]
        self.layers = nn.SequentialCell(*layers)
        self.output_activation = get_activation(config.get("output_activation", None))

    def construct(self, x):
        x = self.layers(x)
        x = self.output_activation(x)
        return x

    def make_linear(self, dim_in, dim_out, is_first, is_last):
        layer = nn.Linear(dim_in, dim_out, bias=False)
        return layer

    def make_activation(self):
        return nn.ReLU()


class SphereInitVanillaMLP(nn.Cell):
    def __init__(self, dim_in, dim_out, config):
        super().__init__()
        self.n_neurons, self.n_hidden_layers = (
            config["n_neurons"],
            config["n_hidden_layers"],
        )
        self.sphere_init, self.weight_norm = True, True
        self.sphere_init_radius = config["sphere_init_radius"]
        self.sphere_init_inside_out = config["inside_out"]

        self.layers = [
            self.make_linear(dim_in, self.n_neurons, is_first=True, is_last=False),
            self.make_activation(),
        ]
        for i in range(self.n_hidden_layers - 1):
            self.layers += [
                self.make_linear(self.n_neurons, self.n_neurons, is_first=False, is_last=False),
                self.make_activation(),
            ]
        self.layers += [self.make_linear(self.n_neurons, dim_out, is_first=False, is_last=True)]
        self.layers = nn.SequentialCell(*self.layers)
        self.output_activation = get_activation(config.get("output_activation", None))

    def construct(self, x):
        x = self.layers(x)
        x = self.output_activation(x)
        return x

    def make_linear(self, dim_in, dim_out, is_first, is_last):
        layer = nn.Linear(dim_in, dim_out, bias=True)

        if is_last:
            if not self.sphere_init_inside_out:
                mint.nn.init.constant_(layer.bias, -self.sphere_init_radius)
                mint.nn.init.normal_(
                    layer.weight,
                    mean=math.sqrt(math.pi) / math.sqrt(dim_in),
                    std=0.0001,
                )
            else:
                mint.nn.init.constant_(layer.bias, self.sphere_init_radius)
                mint.nn.init.normal_(
                    layer.weight,
                    mean=-math.sqrt(math.pi) / math.sqrt(dim_in),
                    std=0.0001,
                )
        elif is_first:
            mint.nn.init.constant_(layer.bias, 0.0)
            mint.nn.init.constant_(layer.weight[:, 3:], 0.0)
            mint.nn.init.normal_(layer.weight[:, :3], 0.0, math.sqrt(2) / math.sqrt(dim_out))
        else:
            mint.nn.init.constant_(layer.bias, 0.0)
            mint.nn.init.normal_(layer.weight, 0.0, math.sqrt(2) / math.sqrt(dim_out))

        if self.weight_norm:
            layer = nn.utils.weight_norm(layer)
        return layer

    def make_activation(self):
        return mint.nn.functional.Softplus(beta=100)


def get_mlp(n_input_dims, n_output_dims, config) -> nn.Cell:
    network: nn.Cell
    if config.otype == "VanillaMLP":
        network = VanillaMLP(n_input_dims, n_output_dims, config_to_primitive(config))
    elif config.otype == "SphereInitVanillaMLP":
        network = SphereInitVanillaMLP(n_input_dims, n_output_dims, config_to_primitive(config))
    else:
        assert config.get("sphere_init", False) is False, "sphere_init=True only supported by VanillaMLP"
    return network
