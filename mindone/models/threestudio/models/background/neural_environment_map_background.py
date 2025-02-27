import random
from dataclasses import dataclass, field
from typing import Optional, Tuple

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.ops import get_activation

from mindspore import Tensor, mint


@threestudio.register("neural-environment-map-background")
class NeuralEnvironmentMapBackground(BaseBackground):
    @dataclass
    class Config(BaseBackground.Config):
        n_output_dims: int = 3
        color_activation: str = "sigmoid"
        dir_encoding_config: dict = field(default_factory=lambda: {"otype": "SphericalHarmonics", "degree": 3})
        mlp_network_config: dict = field(
            default_factory=lambda: {
                "otype": "VanillaMLP",
                "activation": "ReLU",
                "n_neurons": 16,
                "n_hidden_layers": 2,
            }
        )
        random_aug: bool = False
        random_aug_prob: float = 0.5
        share_aug_bg: bool = False
        eval_color: Optional[Tuple[float, float, float]] = None

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"within neural bg, the enc cfg: {self.cfg.dir_encoding_config}")
        self.encoding = get_encoding(3, self.cfg.dir_encoding_config)
        self.network = get_mlp(
            self.encoding.n_output_dims,
            self.cfg.n_output_dims,
            self.cfg.mlp_network_config,
        )

    def construct(self, dirs: Tensor) -> Tensor:  # B H*W 3 -> B H*W Nc
        if not self.training and self.cfg.eval_color is not None:
            return mint.ones(*dirs.shape[:-1], self.cfg.n_output_dims).to(dirs) * Tensor(self.cfg.eval_color).to(dirs)
        # viewdirs must be normalized before passing to this function
        dirs = (dirs + 1.0) / 2.0  # (-1, 1) => (0, 1)
        dirs_embd = self.encoding(dirs.view(-1, 3))
        color = self.network(dirs_embd).view(*dirs.shape[:-1], self.cfg.n_output_dims)
        color = get_activation(self.cfg.color_activation)(color)
        if self.training and self.cfg.random_aug and random.random() < self.cfg.random_aug_prob:
            # use random background color with probability random_aug_prob
            n_color = 1 if self.cfg.share_aug_bg else dirs.shape[0]
            color = color * 0 + (  # prevent checking for unused parameters in DDP
                mint.rand(n_color, 1, self.cfg.n_output_dims).to(dirs.dtype).broadcast_to((*dirs.shape[:-1], -1))
            )
        return color
