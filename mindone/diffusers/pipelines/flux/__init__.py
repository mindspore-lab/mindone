"""Adapted from https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/flux/__init__.py."""

from typing import TYPE_CHECKING

from ...utils import _LazyModule

_import_structure = {
    "pipeline_output": ["FluxPipelineOutput", "FluxPriorReduxPipelineOutput"],
    "modeling_flux": ["ReduxImageEncoder"],
    "pipeline_flux": ["FluxPipeline"],
    "pipeline_flux_control": ["FluxControlPipeline"],
    "pipeline_flux_control_img2img": ["FluxControlImg2ImgPipeline"],
    "pipeline_flux_control_inpaint": ["FluxControlInpaintPipeline"],
    "pipeline_flux_controlnet": ["FluxControlNetPipeline"],
    "pipeline_flux_controlnet_image_to_image": ["FluxControlNetImg2ImgPipeline"],
    "pipeline_flux_controlnet_inpainting": ["FluxControlNetInpaintPipeline"],
    "pipeline_flux_fill": ["FluxFillPipeline"],
    "pipeline_flux_img2img": ["FluxImg2ImgPipeline"],
    "pipeline_flux_inpaint": ["FluxInpaintPipeline"],
    "pipeline_flux_kontext": ["FluxKontextPipeline"],
    "pipeline_flux_kontext_inpaint": ["FluxKontextInpaintPipeline"],
    "pipeline_flux_prior_redux": ["FluxPriorReduxPipeline"],
}

if TYPE_CHECKING:
    from .modeling_flux import ReduxImageEncoder
    from .pipeline_flux import FluxPipeline
    from .pipeline_flux_control import FluxControlPipeline
    from .pipeline_flux_control_img2img import FluxControlImg2ImgPipeline
    from .pipeline_flux_control_inpaint import FluxControlInpaintPipeline
    from .pipeline_flux_controlnet import FluxControlNetPipeline
    from .pipeline_flux_controlnet_image_to_image import FluxControlNetImg2ImgPipeline
    from .pipeline_flux_controlnet_inpainting import FluxControlNetInpaintPipeline
    from .pipeline_flux_fill import FluxFillPipeline
    from .pipeline_flux_img2img import FluxImg2ImgPipeline
    from .pipeline_flux_inpaint import FluxInpaintPipeline
    from .pipeline_flux_kontext import FluxKontextPipeline
    from .pipeline_flux_kontext_inpaint import FluxKontextInpaintPipeline
    from .pipeline_flux_prior_redux import FluxPriorReduxPipeline
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
