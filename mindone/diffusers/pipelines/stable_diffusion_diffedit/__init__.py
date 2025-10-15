"""Adapted from https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/stable_diffusion_diffedit/__init__.py."""

from typing import TYPE_CHECKING

from ...utils import _LazyModule

_dummy_objects = {}
_import_structure = {}


_import_structure["pipeline_stable_diffusion_diffedit"] = ["StableDiffusionDiffEditPipeline"]

if TYPE_CHECKING:
    from .pipeline_stable_diffusion_diffedit import StableDiffusionDiffEditPipeline

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
