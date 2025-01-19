from dataclasses import dataclass
from typing import Any, Optional, Union

from omegaconf import DictConfig
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import load_module_weights

from mindspore import Parameter, Tensor, nn


class Configurable:
    @dataclass
    class Config:
        pass

    def __init__(self, cfg: Optional[dict] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(self.Config, cfg)


class Updateable:
    system_mod_list = [
        "do_update_step",  # this to update uncond dataset
        "geometry",
        "material",
        "renderer",
        "background",
        "guidance",
        "prompt_processor",
        "exporter",
    ]

    def do_update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # for attr in self.__dir__():
        for attr in self.system_mod_list:
            if attr.startswith("_"):
                continue
            try:
                module = getattr(self, attr)
            except Exception:
                print("get attr fail, try next module_name")
                continue  # ignore attributes like property, which can't be retrived using getattr?
            if isinstance(module, Updateable):
                # if attr in self.system_mod_list:
                module.do_update_step(epoch, global_step, on_load_weights=on_load_weights)
        self.update_step(epoch, global_step, on_load_weights=on_load_weights)

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # override this method to implement custom update logic
        # if on_load_weights is True, you should be careful doing things related to model evaluations,
        # as the models and tensors are not guarenteed to be on the same device
        pass


def update_if_possible(module: Any, epoch: int, global_step: int) -> None:
    if isinstance(module, Updateable):
        module.do_update_step(epoch, global_step)


class BaseObject(Updateable):
    @dataclass
    class Config:
        pass

    cfg: Config  # add this to every subclass of BaseObject to enable static type checking

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None, *args, **kwargs) -> None:
        super().__init__()
        self.cfg = parse_structured(self.Config, cfg)
        self.configure(*args, **kwargs)

    def configure(self, *args, **kwargs) -> None:
        pass


class BaseModule(nn.Cell, Updateable):
    @dataclass
    class Config:
        weights: Optional[str] = None

    cfg: Config  # add this to every subclass of BaseModule to enable static type checking

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None, *args, **kwargs) -> None:
        super().__init__()
        self.cfg = parse_structured(self.Config, cfg)
        self.configure(*args, **kwargs)
        if self.cfg.weights is not None:
            # format: path/to/weights:module_name
            weights_path, module_name = self.cfg.weights.split(":")
            state_dict, epoch, global_step = load_module_weights(
                weights_path, module_name=module_name, map_location="cpu"
            )
            self.load_state_dict(state_dict)
            self.do_update_step(epoch, global_step, on_load_weights=True)  # restore states
        # dummy tensor to indicate model state
        self._dummy: Tensor

    def register_buffer_ms(self, name: str, tensor: Tensor):
        return setattr(self, name, Parameter(default_input=tensor, requires_grad=False))

    def configure(self, *args, **kwargs) -> None:
        pass

    def construct(self):
        raise NotImplementedError
