from dataclasses import dataclass
from typing import Any, Dict

import mindspore as ms
from mindspore import Tensor, nn

import mindone.models.threestudio as threestudio
from mindone.models.threestudio.models.geometry.implicit_volume import ImplicitVolume
from mindone.models.threestudio.utils.config import load_config, parse_structured
from mindone.models.threestudio.utils.misc import load_module_weights


class Renderer(nn.Cell):
    @dataclass
    class Config:
        radius: float = 1.0

    def __init__(self, cfg: Dict, cfg_for_highres: Dict) -> None:
        super().__init__()
        # original renderer base cannot get recompute correct, nram leakage, replace the parent class with Cell
        self.cfg = self.Config(cfg.renderer.radius)
        self.material = threestudio.find(cfg.material_type)(cfg.material)
        self.background = threestudio.find(cfg.background_type)(cfg.background)
        if (
            cfg.geometry_convert_from  # from_coarse must be specified
            and cfg_for_highres["train_highres"]
            and cfg_for_highres["resumed"]
        ):
            threestudio.info("Initializing geometry from a given checkpoint ...")

            prev_cfg = load_config("configs/mvdream-sd21.yaml")  # TODO: hard-coded relative path
            prev_system_cfg = parse_structured(ImplicitVolume.Config, prev_cfg.system)
            prev_geometry_cfg = prev_system_cfg.geometry
            prev_geometry_cfg.update(self.cfg.geometry_convert_override)
            prev_geometry = threestudio.find(prev_system_cfg.geometry_type)(prev_geometry_cfg)
            state_dict, epoch, global_step = load_module_weights(
                self.cfg.geometry_convert_from,
                module_name="geometry",
                map_location="cpu",
            )

            # prev_geometry.load_state_dict(state_dict, strict=False)
            m, u = ms.load_param_into_net(prev_geometry, state_dict, strict_load=False)

            # convert from coarse stage geometry
            self.geometry = threestudio.find(self.cfg.geometry_type).create_from(
                prev_geometry,
                self.cfg.geometry,
                copy_net=self.cfg.geometry_convert_inherit_texture,
            )
            del prev_geometry
        else:
            self.geometry = threestudio.find(self.cfg.geometry_type)(self.cfg.geometry)
        self.cfg = self.Config(**cfg.renderer)

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
