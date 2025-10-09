"""Adapted from https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/sana/__init__.py."""

from typing import TYPE_CHECKING

from ...utils import _LazyModule

_dummy_objects = {}
_import_structure = {}


_import_structure["pipeline_sana"] = ["SanaPipeline"]
_import_structure["pipeline_sana_controlnet"] = ["SanaControlNetPipeline"]
_import_structure["pipeline_sana_sprint"] = ["SanaSprintPipeline"]
_import_structure["pipeline_sana_sprint_img2img"] = ["SanaSprintImg2ImgPipeline"]

if TYPE_CHECKING:
    from .pipeline_sana import SanaPipeline
    from .pipeline_sana_controlnet import SanaControlNetPipeline
    from .pipeline_sana_sprint import SanaSprintPipeline
    from .pipeline_sana_sprint_img2img import SanaSprintImg2ImgPipeline
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )

    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
