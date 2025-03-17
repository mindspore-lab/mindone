from typing import TYPE_CHECKING

from ...utils import _LazyModule

_import_structure = {
    "pipeline_output": ["FluxPipelineOutput"],
    "pipeline_flux": ["FluxPipeline"],
    "pipeline_flux_controlnet": ["FluxControlNetPipeline"],
    "pipeline_flux_controlnet_image_to_image": ["FluxControlNetImg2ImgPipeline"],
    "pipeline_flux_controlnet_inpainting": ["FluxControlNetInpaintPipeline"],
    "pipeline_flux_img2img": ["FluxImg2ImgPipeline"],
    "pipeline_flux_inpaint": ["FluxInpaintPipeline"],
}

if TYPE_CHECKING:
    from .pipeline_flux import FluxPipeline
    from .pipeline_flux_controlnet import FluxControlNetPipeline
    from .pipeline_flux_controlnet_image_to_image import FluxControlNetImg2ImgPipeline
    from .pipeline_flux_controlnet_inpainting import FluxControlNetInpaintPipeline
    from .pipeline_flux_img2img import FluxImg2ImgPipeline
    from .pipeline_flux_inpaint import FluxInpaintPipeline
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
