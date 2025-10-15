from dataclasses import dataclass

from mindspore import Tensor

from mindone.models.threestudio.utils.base import BaseModule


class BaseBackground(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pass

    cfg: Config

    def configure(self):
        pass

    def construct(self, dirs: Tensor) -> Tensor:  # B H W 3  # B H W Nc
        raise NotImplementedError
