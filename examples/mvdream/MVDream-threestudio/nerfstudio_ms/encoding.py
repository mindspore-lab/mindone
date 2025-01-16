from abc import abstractmethod
from typing import Literal, Optional

import numpy as np
from nerfstudio_ms.base_field import FieldComponent
from nerfstudio_ms.spherical_harmonics import MAX_SH_DEGREE, components_from_spherical_harmonics

import mindspore as ms
from mindspore import Tensor, mint, ops


class Encoding(FieldComponent):
    """Encode an input tensor. Intended to be subclassed

    Args:
        in_dim: Input dimension of tensor
    """

    def __init__(self, in_dim: int) -> None:
        if in_dim <= 0:
            raise ValueError("Input dimension should be greater than zero")
        super().__init__(in_dim=in_dim)

    @classmethod
    def get_tcnn_encoding_config(cls) -> dict:
        """Get the encoding configuration for tcnn if implemented"""
        raise NotImplementedError("Encoding does not have a TCNN implementation")

    @abstractmethod
    def construct(self, in_tensor: Tensor) -> Tensor:
        """Call forward and returns and processed tensor

        Args:
            in_tensor: the input tensor to process
        """
        raise NotImplementedError


class HashEncoding(Encoding):
    """Hash encoding

    Args:
        num_levels: Number of feature grids.
        min_res: Resolution of smallest feature grid.
        max_res: Resolution of largest feature grid.
        log2_hashmap_size: Size of hash map is 2^log2_hashmap_size.
        features_per_level: Number of features per level.
        hash_init_scale: Value to initialize hash grid.
        implementation: Implementation of hash encoding. Fallback to ms if tcnn not available.
        interpolation: Interpolation override for tcnn hashgrid. Not supported for ms unless linear.
    """

    def __init__(
        self,
        num_levels: int = 16,
        min_res: int = 16,
        max_res: int = 1024,
        log2_hashmap_size: int = 19,
        features_per_level: int = 2,
        hash_init_scale: float = 0.001,
        implementation: Literal["ms"] = "ms",
        interpolation: Optional[Literal["Nearest", "Linear", "Smoothstep"]] = None,
    ) -> None:
        super().__init__(in_dim=3)
        self.num_levels = num_levels
        self.min_res = min_res
        self.features_per_level = features_per_level
        self.hash_init_scale = hash_init_scale
        self.log2_hashmap_size = log2_hashmap_size
        self.hash_table_size = 2**log2_hashmap_size

        levels = np.arange(num_levels)
        growth_factor = np.exp((np.log(max_res) - np.log(min_res)) / (num_levels - 1)) if num_levels > 1 else 1
        self.scalings = np.floor(min_res * growth_factor**levels).reshape(-1, 1)

        self.hash_offset = levels * self.hash_table_size

        self.tcnn_encoding = None
        # self.hash_table = np.empty(0)
        if implementation == "ms":
            self.build_nn_modules()
        else:
            raise ValueError

        if self.tcnn_encoding is None:
            assert (
                interpolation is None or interpolation == "Linear"
            ), f"interpolation '{interpolation}' is not supported for ms encoding backend"

    def build_nn_modules(self) -> None:
        """Initialize the ms version of the hash encoding."""
        hash_table = mint.rand(self.hash_table_size * self.num_levels, self.features_per_level) * 2 - 1
        hash_table *= self.hash_init_scale
        self.hash_table = ms.Parameter(hash_table)

    @classmethod
    def get_tcnn_encoding_config(
        cls, num_levels, features_per_level, log2_hashmap_size, min_res, growth_factor, interpolation=None
    ) -> dict:
        """Get the encoding configuration for tcnn if implemented"""
        encoding_config = {
            "otype": "HashGrid",
            "n_levels": num_levels,
            "n_features_per_level": features_per_level,
            "log2_hashmap_size": log2_hashmap_size,
            "base_resolution": min_res,
            "per_level_scale": growth_factor,
        }
        if interpolation is not None:
            encoding_config["interpolation"] = interpolation
        return encoding_config

    def get_out_dim(self) -> int:
        return self.num_levels * self.features_per_level

    def hash_fn(self, in_tensor: Tensor) -> Tensor:
        """Returns hash tensor using method described in Instant-NGP

        Args:
            in_tensor: Tensor to be hashed
        """

        # min_val = mint.min(in_tensor)
        # max_val = mint.max(in_tensor)
        # assert min_val >= 0.0
        # assert max_val <= 1.0

        in_tensor = in_tensor * Tensor([1, 2654435761, 805459861], dtype=ms.int32)
        x = mint.bitwise_xor(in_tensor[..., 0], in_tensor[..., 1])
        x = mint.bitwise_xor(x, in_tensor[..., 2])
        x %= self.hash_table_size
        x += Tensor(self.hash_offset, dtype=ms.int32)
        return x

    def construct(self, in_tensor: Tensor) -> Tensor:
        """ms implementation of hash encoding. Not as fast as the tcnn implementation."""

        assert in_tensor.shape[-1] == 3
        in_tensor = in_tensor[..., None, :]  # [..., 1, 3]
        scaled = in_tensor * Tensor(self.scalings, dtype=ms.int32)  # [..., L, 3]
        scaled_c = mint.ceil(scaled).to(ms.int32)
        scaled_f = mint.floor(scaled).to(ms.int32)

        offset = scaled - scaled_f

        hashed_0 = self.hash_fn(scaled_c)  # [..., num_levels]
        hashed_1 = self.hash_fn(mint.cat([scaled_c[..., 0:1], scaled_f[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_2 = self.hash_fn(mint.cat([scaled_f[..., 0:1], scaled_f[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_3 = self.hash_fn(mint.cat([scaled_f[..., 0:1], scaled_c[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_4 = self.hash_fn(mint.cat([scaled_c[..., 0:1], scaled_c[..., 1:2], scaled_f[..., 2:3]], dim=-1))
        hashed_5 = self.hash_fn(mint.cat([scaled_c[..., 0:1], scaled_f[..., 1:2], scaled_f[..., 2:3]], dim=-1))
        hashed_6 = self.hash_fn(scaled_f)
        hashed_7 = self.hash_fn(mint.cat([scaled_f[..., 0:1], scaled_c[..., 1:2], scaled_f[..., 2:3]], dim=-1))

        f_0 = self.hash_table[hashed_0]  # [..., num_levels, features_per_level]
        f_1 = self.hash_table[hashed_1]
        f_2 = self.hash_table[hashed_2]
        f_3 = self.hash_table[hashed_3]
        f_4 = self.hash_table[hashed_4]
        f_5 = self.hash_table[hashed_5]
        f_6 = self.hash_table[hashed_6]
        f_7 = self.hash_table[hashed_7]

        f_03 = f_0 * offset[..., 0:1] + f_3 * (1 - offset[..., 0:1])
        f_12 = f_1 * offset[..., 0:1] + f_2 * (1 - offset[..., 0:1])
        f_56 = f_5 * offset[..., 0:1] + f_6 * (1 - offset[..., 0:1])
        f_47 = f_4 * offset[..., 0:1] + f_7 * (1 - offset[..., 0:1])

        f0312 = f_03 * offset[..., 1:2] + f_12 * (1 - offset[..., 1:2])
        f4756 = f_47 * offset[..., 1:2] + f_56 * (1 - offset[..., 1:2])

        encoded_value = f0312 * offset[..., 2:3] + f4756 * (
            1 - offset[..., 2:3]
        )  # [..., num_levels, features_per_level]

        return mint.flatten(encoded_value, start_dim=-2, end_dim=-1)  # [..., num_levels * features_per_level]

    # def construct(self, in_tensor: Tensor) -> Tensor:
    #     return self.forward(in_tensor)


class NSOEncoding(HashEncoding):
    """Adapted from NerfStudio in ms"""

    def __init__(self, in_channels, config, dtype=ms.float32) -> None:
        super().__init__()
        self.n_input_dims = in_channels
        # self.encoding = tcnn.Encoding(in_channels, config, dtype=dtype)
        # self.n_output_dims = self.encoding.n_output_dims

    def construct(self, x):
        return self.encoding(x)

    def encoding(
        self,
        in_tensor: Tensor,
        covs: Tensor = None,
    ) -> Tensor:
        """Calculates NeRF encoding. If covariances are provided the encodings will be integrated as proposed
            in mip-NeRF.

        Args:
            in_tensor: For best performance, the input tensor should be between 0 and 1.
            covs: Covariances of input points.
        Returns:
            Output values will be between -1 and 1
        """
        scaled_in_tensor = 2 * Tensor(np.pi, dtype=ms.int32) * in_tensor  # scale to [0, 2pi]
        freqs = 2 ** mint.linspace(self.min_freq, self.max_freq, self.num_frequencies)
        scaled_inputs = scaled_in_tensor[..., None] * freqs  # [..., "input_dim", "num_scales"]
        scaled_inputs = scaled_inputs.view(*scaled_inputs.shape[:-2], -1)  # [..., "input_dim" * "num_scales"]

        if covs is None:
            encoded_inputs = mint.sin(
                mint.cat([scaled_inputs, scaled_inputs + Tensor(np.pi, dtype=ms.int32) / 2.0], dim=-1)
            )
        else:
            input_var = ops.diagonal(covs, dim1=-2, dim2=-1)[..., :, None] * freqs[None, :] ** 2
            input_var = input_var.reshape((*input_var.shape[:-2], -1))
            encoded_inputs = expected_sin(
                mint.cat([scaled_inputs, scaled_inputs + Tensor(np.pi, dtype=ms.int32) / 2.0], dim=-1),
                mint.cat(2 * [input_var], dim=-1),
            )

        return encoded_inputs


class SHEncoding(Encoding):
    """Spherical harmonic encoding

    Args:
        levels: Number of spherical harmonic levels to encode. (level = sh degree + 1)
    """

    def __init__(self, levels: int = 4, implementation: Literal["tcnn", "ms"] = "ms") -> None:
        super().__init__(in_dim=3)

        if levels <= 0 or levels > MAX_SH_DEGREE + 1:
            raise ValueError(
                f"Spherical harmonic encoding only supports 1 to {MAX_SH_DEGREE + 1} levels, requested {levels}"
            )

        self.levels = levels

        self.tcnn_encoding = None

    @classmethod
    def get_tcnn_encoding_config(cls, levels: int) -> dict:
        """Get the encoding configuration for tcnn if implemented"""
        encoding_config = {
            "otype": "SphericalHarmonics",
            "degree": levels,
        }
        return encoding_config

    def get_out_dim(self) -> int:
        return self.levels**2

    @ms._no_grad()
    def construct(self, in_tensor: Tensor) -> Tensor:
        return components_from_spherical_harmonics(degree=self.levels - 1, directions=in_tensor)


def expected_sin(x_means: Tensor, x_vars: Tensor) -> Tensor:
    """Computes the expected value of sin(y) where y ~ N(x_means, x_vars)

    Args:
        x_means: Mean values.
        x_vars: Variance of values.

    Returns:
        mint.Tensor: The expected value of sin.
    """
    return mint.exp(-0.5 * x_vars) * mint.sin(x_means)
