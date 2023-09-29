import logging
import os

import mindspore_lite as mslite

__all__ = ["LiteConverter"]

_logger = logging.getLogger(__name__)


class LiteConverter:
    def __init__(self, target: str = "ascend") -> None:
        self.optimize_dict = {"ascend": "ascend_oriented", "gpu": "gpu_oriented", "cpu": "general"}
        self.target = target
        self.config_file = "deploy/lite_configs/vc_lite.cfg"

    def __call__(self, name: str, model_save_path: str = "./models/lite", mindir_root: str = "./models/mindir") -> None:
        if not os.path.isdir(model_save_path):
            os.makedirs(model_save_path)

        mindir_path = os.path.join(mindir_root, f"{name}.mindir")
        if not os.path.isfile(mindir_path):
            mindir_path = os.path.join(mindir_root, f"{name}_graph.mindir")

        lite_file = os.path.join(model_save_path, f"{name}_lite")

        if os.path.isfile(f"{lite_file}.mindir"):
            _logger.warning(f"`{name}` lite mindir already exist, skip.")
            return

        if os.path.isfile(f"{lite_file}_graph.mindir"):
            _logger.warning(f"`{name}` lite mindir already exist, skip.")
            return

        # create converter when it is necessary
        converter = mslite.Converter()
        converter.save_type = mslite.ModelType.MINDIR
        converter.optimize = self.optimize_dict[self.target]

        converter.convert(
            fmk_type=mslite.FmkType.MINDIR,
            model_file=mindir_path,
            output_file=lite_file,
            config_file=self.config_file,
        )
        _logger.info(f"convert `{name}` mindspore lite done")
