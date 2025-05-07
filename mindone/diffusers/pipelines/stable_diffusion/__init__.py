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
    "pipeline_stable_unclip": ["StableUnCLIPPipeline"],
    "pipeline_stable_unclip_img2img": ["StableUnCLIPImg2ImgPipeline"],
    "safety_checker": ["StableDiffusionSafetyChecker"],
    "stable_unclip_image_normalizer": ["StableUnCLIPImageNormalizer"],
}

if TYPE_CHECKING:
    from .clip_image_project_model import CLIPImageProjection
    from .pipeline_stable_diffusion import StableDiffusionPipeline, StableDiffusionPipelineOutput
    from .pipeline_stable_diffusion_depth2img import StableDiffusionDepth2ImgPipeline
    from .pipeline_stable_diffusion_image_variation import StableDiffusionImageVariationPipeline
    from .pipeline_stable_diffusion_img2img import StableDiffusionImg2ImgPipeline
    from .pipeline_stable_diffusion_inpaint import StableDiffusionInpaintPipeline
    from .pipeline_stable_diffusion_instruct_pix2pix import StableDiffusionInstructPix2PixPipeline
    from .pipeline_stable_diffusion_latent_upscale import StableDiffusionLatentUpscalePipeline
    from .pipeline_stable_diffusion_upscale import StableDiffusionUpscalePipeline
    from .pipeline_stable_unclip import StableUnCLIPPipeline
    from .pipeline_stable_unclip_img2img import StableUnCLIPImg2ImgPipeline
    from .safety_checker import StableDiffusionSafetyChecker
    from .stable_unclip_image_normalizer import StableUnCLIPImageNormalizer
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
