from typing import TYPE_CHECKING

from ...utils import _LazyModule

_import_structure = {}

_import_structure["pipeline_pag_controlnet_sd"] = ["StableDiffusionControlNetPAGPipeline"]
_import_structure["pipeline_pag_controlnet_sd_xl"] = ["StableDiffusionXLControlNetPAGPipeline"]
_import_structure["pipeline_pag_hunyuandit"] = ["HunyuanDiTPAGPipeline"]
_import_structure["pipeline_pag_kolors"] = ["KolorsPAGPipeline"]
_import_structure["pipeline_pag_pixart_sigma"] = ["PixArtSigmaPAGPipeline"]
_import_structure["pipeline_pag_sd"] = ["StableDiffusionPAGPipeline"]
_import_structure["pipeline_pag_sd_3"] = ["StableDiffusion3PAGPipeline"]
_import_structure["pipeline_pag_sd_animatediff"] = ["AnimateDiffPAGPipeline"]
_import_structure["pipeline_pag_sd_xl"] = ["StableDiffusionXLPAGPipeline"]
_import_structure["pipeline_pag_sd_xl_img2img"] = ["StableDiffusionXLPAGImg2ImgPipeline"]
_import_structure["pipeline_pag_sd_xl_inpaint"] = ["StableDiffusionXLPAGInpaintPipeline"]

if TYPE_CHECKING:
    from .pipeline_pag_controlnet_sd import StableDiffusionControlNetPAGPipeline
    from .pipeline_pag_controlnet_sd_xl import StableDiffusionXLControlNetPAGPipeline
    from .pipeline_pag_hunyuandit import HunyuanDiTPAGPipeline
    from .pipeline_pag_kolors import KolorsPAGPipeline
    from .pipeline_pag_pixart_sigma import PixArtSigmaPAGPipeline
    from .pipeline_pag_sd import StableDiffusionPAGPipeline
    from .pipeline_pag_sd_3 import StableDiffusion3PAGPipeline
    from .pipeline_pag_sd_animatediff import AnimateDiffPAGPipeline
    from .pipeline_pag_sd_xl import StableDiffusionXLPAGPipeline
    from .pipeline_pag_sd_xl_img2img import StableDiffusionXLPAGImg2ImgPipeline
    from .pipeline_pag_sd_xl_inpaint import StableDiffusionXLPAGInpaintPipeline

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
