from dataclasses import dataclass

from threestudio.utils.base import BaseModule

from mindspore import Tensor


class BaseBackground(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pass

    cfg: Config

    def configure(self):
        pass

    def construct(self, dirs: Tensor) -> Tensor:  # B H W 3  # B H W Nc
        raise NotImplementedError
