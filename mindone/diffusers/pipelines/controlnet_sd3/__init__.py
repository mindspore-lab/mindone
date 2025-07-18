"""Adapted from https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/controlnet_sd3/__init__.py."""

from typing import TYPE_CHECKING

from ...utils import _LazyModule

_import_structure = {}
_import_structure["pipeline_stable_diffusion_3_controlnet"] = ["StableDiffusion3ControlNetPipeline"]
_import_structure["pipeline_stable_diffusion_3_controlnet_inpainting"] = [
    "StableDiffusion3ControlNetInpaintingPipeline"
]


if TYPE_CHECKING:
    from .pipeline_stable_diffusion_3_controlnet import StableDiffusion3ControlNetPipeline
    from .pipeline_stable_diffusion_3_controlnet_inpainting import StableDiffusion3ControlNetInpaintingPipeline
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
