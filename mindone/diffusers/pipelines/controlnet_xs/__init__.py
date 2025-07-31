"""Adapted from https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/controlnet_xs/__init__.py."""

from typing import TYPE_CHECKING

from ...utils import _LazyModule

_import_structure = {}
_import_structure["pipeline_controlnet_xs"] = ["StableDiffusionControlNetXSPipeline"]
_import_structure["pipeline_controlnet_xs_sd_xl"] = ["StableDiffusionXLControlNetXSPipeline"]


if TYPE_CHECKING:
    from .pipeline_controlnet_xs import StableDiffusionControlNetXSPipeline
    from .pipeline_controlnet_xs_sd_xl import StableDiffusionXLControlNetXSPipeline
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
