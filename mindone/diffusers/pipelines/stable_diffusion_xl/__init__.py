from typing import TYPE_CHECKING

from ...utils import _LazyModule

_import_structure = {
    "pipeline_output": ["StableDiffusionXLPipelineOutput"],
    "pipeline_stable_diffusion_xl": ["StableDiffusionXLPipeline"],
    "pipeline_stable_diffusion_xl_img2img": ["StableDiffusionXLImg2ImgPipeline"],
    "pipeline_stable_diffusion_xl_inpaint": ["StableDiffusionXLInpaintPipeline"],
}

if TYPE_CHECKING:
    from .pipeline_stable_diffusion_xl import StableDiffusionXLPipeline, StableDiffusionXLPipelineOutput
    from .pipeline_stable_diffusion_xl_img2img import StableDiffusionXLImg2ImgPipeline
    from .pipeline_stable_diffusion_xl_inpaint import StableDiffusionXLInpaintPipeline

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
