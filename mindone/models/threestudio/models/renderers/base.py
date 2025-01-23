from dataclasses import dataclass
from typing import Any, Dict

import mindspore as ms
from mindspore import Tensor, nn


class Renderer(nn.Cell):
    @dataclass
    class Config:
        radius: float = 1.0

    cfg: Config

    def __init__(
        self,
        radius: float,
    ) -> None:
        # original renderer base cannot get recompute correct, nram leakage, replace the parent class with Cell

        # set up bounding box
        self.bbox = Tensor(
            [
                [-self.cfg.radius, -self.cfg.radius, -self.cfg.radius],
                [self.cfg.radius, self.cfg.radius, self.cfg.radius],
            ],
            dtype=ms.float32,
        )

    def construct(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError


class VolumeRenderer(Renderer):
    pass


class Rasterizer(Renderer):
    pass
