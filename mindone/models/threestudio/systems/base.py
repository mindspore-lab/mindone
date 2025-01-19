import os
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

import threestudio
from threestudio.models.exporters.base import ExporterOutput
from threestudio.utils.base import Updateable, update_if_possible
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import C, load_module_weights
from threestudio.utils.saving import SaverMixin

import mindspore as ms
import mindspore.mint.nn.functional as F
from mindspore import nn


class BaseSystem(nn.Cell, Updateable, SaverMixin):
    @dataclass
    class Config:
        loggers: dict = field(default_factory=dict)
        loss: dict = field(default_factory=dict)
        optimizer: dict = field(default_factory=dict)
        scheduler: Optional[dict] = None
        weights: Optional[str] = None
        weights_ignore_modules: Optional[List[str]] = None
        cleanup_after_validation_step: bool = False
        cleanup_after_test_step: bool = False

    cfg: Config

    def __init__(self, cfg, resumed=False) -> None:
        super().__init__()
        self.cfg = parse_structured(self.Config, cfg)
        self._save_dir: Optional[str] = None
        self._resumed: bool = resumed
        self._resumed_eval: bool = False
        self._resumed_eval_status: dict = {"global_step": 0, "current_epoch": 0}
        # if "loggers" in cfg:
        #     self.create_loggers(cfg.loggers)

        self.configure()
        if self.cfg.weights is not None:
            self.load_weights(self.cfg.weights, self.cfg.weights_ignore_modules)
        self.post_configure()

        # fred time mea
        self.prev_time = time.time()
        self.delta_buf = []

    def load_weights(self, weights: str, ignore_modules: Optional[List[str]] = None):
        state_dict, epoch, global_step = load_module_weights(weights, ignore_modules=ignore_modules)
        m, u = ms.load_param_into_net(self, state_dict, strict_load=False)

    def set_resume_status_eval(self, current_epoch: int, global_step: int):
        # restore correct epoch and global step in eval
        self._resumed_eval = True
        self._resumed_eval_status["current_epoch"] = current_epoch
        self._resumed_eval_status["global_step"] = global_step

    @property
    def resumed(self):
        # whether from resumed checkpoint
        return self._resumed

    @property
    def true_global_step(self):
        if self._resumed_eval:
            return self._resumed_eval_status["global_step"]
        else:
            return self.global_step

    @property
    def true_current_epoch(self):
        if self._resumed_eval:
            return self._resumed_eval_status["current_epoch"]
        else:
            return self.current_epoch

    def configure(self) -> None:
        self.global_step = 0
        self.current_epoch = 0

    def post_configure(self) -> None:
        """
        executed after weights are loaded
        """
        pass

    def C(self, value: Any) -> float:
        return C(value, self.true_current_epoch, self.true_global_step)

    def construct(self):
        pass

    def preprocess_data(self, batch, stage):
        pass

    def on_test_epoch_end(self):
        pass

    """
    Implementing on_after_batch_transfer of DataModule does the same.
    But on_after_batch_transfer does not support DP.
    """

    def on_train_batch_start(self, batch, batch_idx, dataset, unused=0):
        # threestudio.info(f'batch_idx: {batch_idx}, fred log while in "on_train_batch_start"')
        self.batch_start_time = time.time()
        delta_t = self.batch_start_time - self.prev_time
        # update prev time stamp
        self.prev_time = time.time()
        self.delta_buf.append(delta_t)
        self.delta_buf = self.delta_buf[-5:]  # only save 5 runs in the delta time buf
        avg_step_speed = len(self.delta_buf) / sum(self.delta_buf)

        threestudio.info(f"elapsed time this batch: {delta_t:.3f}s")
        threestudio.info(f"Running avg step speed: {avg_step_speed:.3f} step/s")

        self.preprocess_data(batch, "train")
        update_if_possible(dataset, self.true_current_epoch, self.true_global_step)
        self.do_update_step(self.true_current_epoch, self.true_global_step)

    def on_validation_batch_start(self, batch, batch_idx, dataset, dataloader_idx=0):
        self.preprocess_data(batch, "validation")
        update_if_possible(dataset, self.true_current_epoch, self.true_global_step)
        self.do_update_step(self.true_current_epoch, self.true_global_step)

    def on_test_batch_start(self, batch, batch_idx, dataset, dataloader_idx=0):
        self.preprocess_data(batch, "test")
        update_if_possible(dataset, self.true_current_epoch, self.true_global_step)
        self.do_update_step(self.true_current_epoch, self.true_global_step)

    def on_predict_batch_start(self, batch, batch_idx, dataset, dataloader_idx=0):
        self.preprocess_data(batch, "predict")
        update_if_possible(dataset, self.true_current_epoch, self.true_global_step)
        self.do_update_step(self.true_current_epoch, self.true_global_step)

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # threestudio.info(f'global_step: {global_step}, fred log while in "update step"')
        pass

    def on_before_optimizer_step(self, optimizer):
        """
        # some gradient-related debugging goes here, example:
        import grad_norm
        norms = grad_norm(self.geometry, norm_type=2)
        print(norms)
        """
        pass


class BaseLift3DSystem(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):
        geometry_type: str = ""
        geometry: dict = field(default_factory=dict)
        geometry_convert_from: Optional[str] = None
        geometry_convert_inherit_texture: bool = False
        # used to override configurations of the previous geometry being converted from,
        # for example isosurface_threshold
        geometry_convert_override: dict = field(default_factory=dict)

        material_type: str = ""
        material: dict = field(default_factory=dict)

        background_type: str = ""
        background: dict = field(default_factory=dict)

        renderer_type: str = ""
        renderer: dict = field(default_factory=dict)

        guidance_type: str = ""
        guidance: dict = field(default_factory=dict)

        prompt_processor_type: str = ""
        prompt_processor: dict = field(default_factory=dict)

        # geometry export configurations, no need to specify in training
        exporter_type: str = "mesh-exporter"
        exporter: dict = field(default_factory=dict)

    cfg: Config

    def configure(self) -> None:
        super().configure()
        if (
            self.cfg.geometry_convert_from  # from_coarse must be specified
            and not self.cfg.weights  # not initialized from coarse when weights are specified
            and not self.resumed  # not initialized from coarse when resumed from checkpoints
        ):
            threestudio.info("Initializing geometry from a given checkpoint ...")
            from threestudio.utils.config import load_config, parse_structured

            prev_cfg = load_config(
                os.path.join(
                    os.path.dirname(self.cfg.geometry_convert_from),
                    "../configs/parsed.yaml",
                )
            )  # TODO: hard-coded relative path
            prev_system_cfg: BaseLift3DSystem.Config = parse_structured(self.Config, prev_cfg.system)
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

        self.material = threestudio.find(self.cfg.material_type)(self.cfg.material)
        self.background = threestudio.find(self.cfg.background_type)(self.cfg.background)
        self.renderer = threestudio.find(self.cfg.renderer_type)(
            self.cfg.renderer,
            geometry=self.geometry,
            material=self.material,
            background=self.background,
        )

    def guidance_evaluation_save(self, comp_rgb, guidance_eval_out):
        """MView Diffusion Guidance"""
        B, size = comp_rgb.shape[:2]
        resize = lambda x: F.interpolate(
            x.permute(0, 3, 1, 2), (size, size), mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)
        filename = f"it{self.true_global_step}-train.png"

        def merge12(x):
            return x.reshape(-1, *x.shape[2:])

        self.save_image_grid(
            filename,
            [
                {
                    "type": "rgb",
                    "img": merge12(comp_rgb),
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_noisy"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_1step"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_1orig"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_final"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            ),
            name="train_step",
            step=self.true_global_step,
            texts=guidance_eval_out["texts"],
        )

    def on_predict_epoch_end(self) -> None:
        if self.exporter.cfg.save_video:
            self.on_test_epoch_end()
        exporter_output: List[ExporterOutput] = self.exporter()
        for out in exporter_output:
            save_func_name = f"save_{out.save_type}"
            if not hasattr(self, save_func_name):
                raise ValueError(f"{save_func_name} not supported by the SaverMixin")
            save_func = getattr(self, save_func_name)
            save_func(f"it{self.true_global_step}-export/{out.save_name}", **out.params)
