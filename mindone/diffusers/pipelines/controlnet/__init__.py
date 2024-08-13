from typing import TYPE_CHECKING

from ...utils import _LazyModule

_import_structure = {}

_import_structure["multicontrolnet"] = ["MultiControlNetModel"]
_import_structure["pipeline_controlnet"] = ["StableDiffusionControlNetPipeline"]
_import_structure["pipeline_controlnet_blip_diffusion"] = ["BlipDiffusionControlNetPipeline"]
_import_structure["pipeline_controlnet_img2img"] = ["StableDiffusionControlNetImg2ImgPipeline"]
_import_structure["pipeline_controlnet_inpaint"] = ["StableDiffusionControlNetInpaintPipeline"]
_import_structure["pipeline_controlnet_inpaint_sd_xl"] = ["StableDiffusionXLControlNetInpaintPipeline"]
_import_structure["pipeline_controlnet_sd_xl"] = ["StableDiffusionXLControlNetPipeline"]
_import_structure["pipeline_controlnet_sd_xl_img2img"] = ["StableDiffusionXLControlNetImg2ImgPipeline"]


if TYPE_CHECKING:
    from .multicontrolnet import MultiControlNetModel
    from .pipeline_controlnet import StableDiffusionControlNetPipeline
    from .pipeline_controlnet_blip_diffusion import BlipDiffusionControlNetPipeline
    from .pipeline_controlnet_img2img import StableDiffusionControlNetImg2ImgPipeline
    from .pipeline_controlnet_inpaint import StableDiffusionControlNetInpaintPipeline
    from .pipeline_controlnet_inpaint_sd_xl import StableDiffusionXLControlNetInpaintPipeline
    from .pipeline_controlnet_sd_xl import StableDiffusionXLControlNetPipeline
    from .pipeline_controlnet_sd_xl_img2img import StableDiffusionXLControlNetImg2ImgPipeline
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
