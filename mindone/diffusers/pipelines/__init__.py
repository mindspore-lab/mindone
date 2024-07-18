from typing import TYPE_CHECKING

from ..utils import _LazyModule

# These modules contain pipelines from multiple libraries/frameworks
_import_structure = {
    "ddim": ["DDIMPipeline"],
    "ddpm": ["DDPMPipeline"],
    "stable_diffusion": [
        "StableDiffusionPipeline",
        "StableDiffusionImg2ImgPipeline",
    ],
    "stable_diffusion_3": [
        "StableDiffusion3Pipeline",
    ],
    "dance_diffusion": [
        "DanceDiffusionPipeline",
    ],
    "stable_diffusion_xl": [
        "StableDiffusionXLPipeline",
        "StableDiffusionXLInpaintPipeline",
        "StableDiffusionXLImg2ImgPipeline",
    ],
    "pipeline_utils": [
        "DiffusionPipeline",
        "ImagePipelineOutput",
    ],
}

if TYPE_CHECKING:
    from .ddim import DDIMPipeline
    from .ddpm import DDPMPipeline
    from .pipeline_utils import DiffusionPipeline, ImagePipelineOutput
    from .stable_diffusion import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
    from .stable_diffusion_3 import StableDiffusion3Pipeline
    from .stable_diffusion_xl import (
        StableDiffusionXLImg2ImgPipeline,
        StableDiffusionXLInpaintPipeline,
        StableDiffusionXLPipeline,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
