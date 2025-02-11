from typing import TYPE_CHECKING

from ...utils import _LazyModule

_import_structure = {
    "clip_image_project_model": ["CLIPImageProjection"],
    "pipeline_output": ["StableDiffusionPipelineOutput"],
    "pipeline_stable_diffusion": ["StableDiffusionPipeline"],
    "pipeline_stable_diffusion_depth2img": ["StableDiffusionDepth2ImgPipeline"],
    "pipeline_stable_diffusion_img2img": ["StableDiffusionImg2ImgPipeline"],
    "pipeline_stable_diffusion_inpaint": ["StableDiffusionInpaintPipeline"],
    "pipeline_stable_diffusion_instruct_pix2pix": ["StableDiffusionInstructPix2PixPipeline"],
    "pipeline_stable_diffusion_image_variation": ["StableDiffusionImageVariationPipeline"],
    "pipeline_stable_diffusion_latent_upscale": ["StableDiffusionLatentUpscalePipeline"],
    "pipeline_stable_diffusion_upscale": ["StableDiffusionUpscalePipeline"],
    "safety_checker": ["StableDiffusionSafetyChecker"],
}

if TYPE_CHECKING:
    from .clip_image_project_model import CLIPImageProjection
    from .pipeline_stable_diffusion import (
        StableDiffusionPipeline,
        StableDiffusionPipelineOutput,
        StableDiffusionSafetyChecker,
    )
    from .pipeline_stable_diffusion_depth2img import StableDiffusionDepth2ImgPipeline
    from .pipeline_stable_diffusion_image_variation import StableDiffusionImageVariationPipeline
    from .pipeline_stable_diffusion_img2img import StableDiffusionImg2ImgPipeline
    from .pipeline_stable_diffusion_inpaint import StableDiffusionInpaintPipeline
    from .pipeline_stable_diffusion_instruct_pix2pix import StableDiffusionInstructPix2PixPipeline
    from .pipeline_stable_diffusion_latent_upscale import StableDiffusionLatentUpscalePipeline
    from .pipeline_stable_diffusion_upscale import StableDiffusionUpscalePipeline
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
