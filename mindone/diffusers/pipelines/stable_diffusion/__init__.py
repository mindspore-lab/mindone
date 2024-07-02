from typing import TYPE_CHECKING

from ...utils import _LazyModule

_import_structure = {
    "clip_image_project_model": ["CLIPImageProjection"],
    "pipeline_output": ["StableDiffusionPipelineOutput"],
    "pipeline_stable_diffusion": ["StableDiffusionPipeline"],
    "pipeline_stable_diffusion_img2img": ["StableDiffusionImg2ImgPipeline"],
    "safety_checker": ["StableDiffusionSafetyChecker"],
}

if TYPE_CHECKING:
    from .clip_image_project_model import CLIPImageProjection
    from .pipeline_stable_diffusion import (
        StableDiffusionPipeline,
        StableDiffusionPipelineOutput,
        StableDiffusionSafetyChecker,
    )
    from .pipeline_stable_diffusion_img2img import StableDiffusionImg2ImgPipeline

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
