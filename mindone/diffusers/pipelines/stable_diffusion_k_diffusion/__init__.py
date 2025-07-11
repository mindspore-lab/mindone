"""Adapted from https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/stable_diffusion_k_diffusion/__init__.py."""

from typing import TYPE_CHECKING

from ...utils import _LazyModule

_import_structure = {
    "pipeline_stable_diffusion_k_diffusion": ["StableDiffusionKDiffusionPipeline"],
    "pipeline_stable_diffusion_xl_k_diffusion": ["StableDiffusionXLKDiffusionPipeline"],
}

if TYPE_CHECKING:
    from .pipeline_stable_diffusion_k_diffusion import StableDiffusionKDiffusionPipeline
    from .pipeline_stable_diffusion_xl_k_diffusion import StableDiffusionXLKDiffusionPipeline

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
