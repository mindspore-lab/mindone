"""Adapted from https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/t2i_adapter/__init__.py."""

from typing import TYPE_CHECKING

from ...utils import _LazyModule

_import_structure = {}
_import_structure["pipeline_stable_diffusion_adapter"] = ["StableDiffusionAdapterPipeline"]
_import_structure["pipeline_stable_diffusion_xl_adapter"] = ["StableDiffusionXLAdapterPipeline"]


if TYPE_CHECKING:
    from .pipeline_stable_diffusion_adapter import StableDiffusionAdapterPipeline
    from .pipeline_stable_diffusion_xl_adapter import StableDiffusionXLAdapterPipeline
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
