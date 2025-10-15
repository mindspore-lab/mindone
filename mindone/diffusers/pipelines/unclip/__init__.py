"""Adapted from https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/unclip/__init__.py."""

from typing import TYPE_CHECKING

from ...utils import _LazyModule

_dummy_objects = {}
_import_structure = {}

_import_structure["pipeline_unclip"] = ["UnCLIPPipeline"]
_import_structure["pipeline_unclip_image_variation"] = ["UnCLIPImageVariationPipeline"]
_import_structure["text_proj"] = ["UnCLIPTextProjModel"]


if TYPE_CHECKING:
    from .pipeline_unclip import UnCLIPPipeline
    from .pipeline_unclip_image_variation import UnCLIPImageVariationPipeline
    from .text_proj import UnCLIPTextProjModel
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
