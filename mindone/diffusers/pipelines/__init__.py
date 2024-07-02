from typing import TYPE_CHECKING

from ..utils import _LazyModule

# These modules contain pipelines from multiple libraries/frameworks
_import_structure = {
    "animatediff": [
        "AnimateDiffPipeline",
        "AnimateDiffVideoToVideoPipeline",
    ],
    "controlnet": [
        "StableDiffusionControlNetImg2ImgPipeline",
        "StableDiffusionControlNetInpaintPipeline",
        "StableDiffusionControlNetPipeline",
        "StableDiffusionXLControlNetImg2ImgPipeline",
        "StableDiffusionXLControlNetInpaintPipeline",
        "StableDiffusionXLControlNetPipeline",
    ],
    "ddim": ["DDIMPipeline"],
    "ddpm": ["DDPMPipeline"],
    "deepfloyd_if": [
        "IFImg2ImgPipeline",
        "IFImg2ImgSuperResolutionPipeline",
        "IFInpaintingPipeline",
        "IFInpaintingSuperResolutionPipeline",
        "IFPipeline",
        "IFSuperResolutionPipeline",
    ],
    "pixart_alpha": ["PixArtAlphaPipeline"],
    "shap_e": ["ShapEImg2ImgPipeline", "ShapEPipeline"],
    "stable_diffusion": [
        "CLIPImageProjection",
        "StableDiffusionPipeline",
        "StableDiffusionImg2ImgPipeline",
    ],
    "stable_diffusion_3": [
        "StableDiffusion3Pipeline",
    ],
    "stable_diffusion_gligen": [
        "StableDiffusionGLIGENPipeline",
        "StableDiffusionGLIGENTextImagePipeline",
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
    "t2i_adapter": [
        "StableDiffusionAdapterPipeline",
        "StableDiffusionXLAdapterPipeline",
    ],
    "i2vgen_xl": ["I2VGenXLPipeline"],
    "unclip": ["UnCLIPImageVariationPipeline", "UnCLIPPipeline"],
}

if TYPE_CHECKING:
    from .animatediff import AnimateDiffPipeline, AnimateDiffVideoToVideoPipeline
    from .controlnet import (
        StableDiffusionControlNetImg2ImgPipeline,
        StableDiffusionControlNetInpaintPipeline,
        StableDiffusionControlNetPipeline,
        StableDiffusionXLControlNetImg2ImgPipeline,
        StableDiffusionXLControlNetInpaintPipeline,
        StableDiffusionXLControlNetPipeline,
    )
    from .ddim import DDIMPipeline
    from .ddpm import DDPMPipeline
    from .deepfloyd_if import (
        IFImg2ImgPipeline,
        IFImg2ImgSuperResolutionPipeline,
        IFInpaintingPipeline,
        IFInpaintingSuperResolutionPipeline,
        IFPipeline,
        IFSuperResolutionPipeline,
    )
    from .i2vgen_xl import I2VGenXLPipeline
    from .pipeline_utils import DiffusionPipeline, ImagePipelineOutput
    from .pixart_alpha import PixArtAlphaPipeline
    from .shap_e import ShapEImg2ImgPipeline, ShapEPipeline
    from .stable_diffusion import CLIPImageProjection, StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
    from .stable_diffusion_3 import StableDiffusion3Pipeline
    from .stable_diffusion_gligen import StableDiffusionGLIGENPipeline, StableDiffusionGLIGENTextImagePipeline
    from .stable_diffusion_xl import (
        StableDiffusionXLImg2ImgPipeline,
        StableDiffusionXLInpaintPipeline,
        StableDiffusionXLPipeline,
    )
    from .t2i_adapter import StableDiffusionAdapterPipeline, StableDiffusionXLAdapterPipeline
    from .unclip import UnCLIPImageVariationPipeline, UnCLIPPipeline
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
