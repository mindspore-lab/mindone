from typing import TYPE_CHECKING

from ..utils import _LazyModule

# These modules contain pipelines from multiple libraries/frameworks
_import_structure = {
    "ddim": ["DDIMPipeline"],
    "ddpm": ["DDPMPipeline"],
    "stable_diffusion": [
        "StableDiffusionPipeline",
        "StableDiffusionImg2ImgPipeline",
        "StableDiffusionInpaintPipeline",
        "StableDiffusionInstructPix2PixPipeline",
        "StableDiffusionDepth2ImgPipeline",
        "StableDiffusionImageVariationPipeline",
        "StableDiffusionLatentUpscalePipeline",
        "StableDiffusionUpscalePipeline",
    ],
    "stable_diffusion_3": [
        "StableDiffusion3Pipeline",
    ],
    "stable_diffusion_xl": [
        "StableDiffusionXLPipeline",
        "StableDiffusionXLInpaintPipeline",
        "StableDiffusionXLImg2ImgPipeline",
        "StableDiffusionXLInstructPix2PixPipeline",
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
    from .stable_diffusion import (
        StableDiffusionDepth2ImgPipeline,
        StableDiffusionImageVariationPipeline,
        StableDiffusionImg2ImgPipeline,
        StableDiffusionInpaintPipeline,
        StableDiffusionInstructPix2PixPipeline,
        StableDiffusionLatentUpscalePipeline,
        StableDiffusionPipeline,
        StableDiffusionUpscalePipeline,
    )
    from .stable_diffusion_xl import (
        StableDiffusionXLImg2ImgPipeline,
        StableDiffusionXLInpaintPipeline,
        StableDiffusionXLInstructPix2PixPipeline,
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
