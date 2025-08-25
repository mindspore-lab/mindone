"""Adapted from https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/shap_e/__init__.py."""

from typing import TYPE_CHECKING

from ...utils import _LazyModule

_dummy_objects = {}
_import_structure = {}

_import_structure["camera"] = ["create_pan_cameras"]
_import_structure["pipeline_shap_e"] = ["ShapEPipeline"]
_import_structure["pipeline_shap_e_img2img"] = ["ShapEImg2ImgPipeline"]
_import_structure["renderer"] = [
    "BoundingBoxVolume",
    "ImportanceRaySampler",
    "MLPNeRFModelOutput",
    "MLPNeRSTFModel",
    "ShapEParamsProjModel",
    "ShapERenderer",
    "StratifiedRaySampler",
    "VoidNeRFModel",
]

if TYPE_CHECKING:
    from .camera import create_pan_cameras
    from .pipeline_shap_e import ShapEPipeline
    from .pipeline_shap_e_img2img import ShapEImg2ImgPipeline
    from .renderer import (
        BoundingBoxVolume,
        ImportanceRaySampler,
        MLPNeRFModelOutput,
        MLPNeRSTFModel,
        ShapEParamsProjModel,
        ShapERenderer,
        StratifiedRaySampler,
        VoidNeRFModel,
    )
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
