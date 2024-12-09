from typing import TYPE_CHECKING

from ..utils import _LazyModule

# These modules contain pipelines from multiple libraries/frameworks
_import_structure = {
    "animatediff": [
        "AnimateDiffPipeline",
        "AnimateDiffControlNetPipeline",
        "AnimateDiffSDXLPipeline",
        "AnimateDiffSparseControlNetPipeline",
        "AnimateDiffVideoToVideoPipeline",
    ],
    "aura_flow": ["AuraFlowPipeline"],
    "auto_pipeline": [
        "AutoPipelineForImage2Image",
        "AutoPipelineForInpainting",
        "AutoPipelineForText2Image",
    ],
    "blip_diffusion": ["BlipDiffusionPipeline"],
    "cogvideo": [
        "CogVideoXPipeline",
        "CogVideoXImageToVideoPipeline",
        "CogVideoXVideoToVideoPipeline",
    ],
    "consistency_models": ["ConsistencyModelPipeline"],
    "controlnet": [
        "BlipDiffusionControlNetPipeline",
        "StableDiffusionControlNetImg2ImgPipeline",
        "StableDiffusionControlNetInpaintPipeline",
        "StableDiffusionControlNetPipeline",
        "StableDiffusionXLControlNetImg2ImgPipeline",
        "StableDiffusionXLControlNetInpaintPipeline",
        "StableDiffusionXLControlNetPipeline",
    ],
    "controlnet_hunyuandit": ["HunyuanDiTControlNetPipeline"],
    "controlnet_xs": [
        "StableDiffusionControlNetXSPipeline",
        "StableDiffusionXLControlNetXSPipeline",
    ],
    "controlnet_sd3": [
        "StableDiffusion3ControlNetPipeline",
    ],
    "dance_diffusion": ["DanceDiffusionPipeline"],
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
    "dit": ["DiTPipeline"],
    "flux": ["FluxPipeline"],
    "hunyuandit": ["HunyuanDiTPipeline"],
    "i2vgen_xl": ["I2VGenXLPipeline"],
    "latent_diffusion": ["LDMSuperResolutionPipeline", "LDMTextToImagePipeline"],
    "kandinsky": [
        "KandinskyCombinedPipeline",
        "KandinskyImg2ImgCombinedPipeline",
        "KandinskyImg2ImgPipeline",
        "KandinskyInpaintCombinedPipeline",
        "KandinskyInpaintPipeline",
        "KandinskyPipeline",
        "KandinskyPriorPipeline",
    ],
    "kandinsky2_2": [
        "KandinskyV22CombinedPipeline",
        "KandinskyV22ControlnetImg2ImgPipeline",
        "KandinskyV22ControlnetPipeline",
        "KandinskyV22Img2ImgCombinedPipeline",
        "KandinskyV22Img2ImgPipeline",
        "KandinskyV22InpaintCombinedPipeline",
        "KandinskyV22InpaintPipeline",
        "KandinskyV22Pipeline",
        "KandinskyV22PriorEmb2EmbPipeline",
        "KandinskyV22PriorPipeline",
    ],
    "kandinsky3": [
        "Kandinsky3Img2ImgPipeline",
        "Kandinsky3Pipeline",
    ],
    "kolors": [
        "KolorsPipeline",
        "KolorsImg2ImgPipeline",
    ],
    "latent_consistency_models": [
        "LatentConsistencyModelImg2ImgPipeline",
        "LatentConsistencyModelPipeline",
    ],
    "latte": ["LattePipeline"],
    "lumina": ["LuminaText2ImgPipeline"],
    "marigold": [
        "MarigoldDepthPipeline",
        "MarigoldNormalsPipeline",
    ],
    "pag": [
        "AnimateDiffPAGPipeline",
        "KolorsPAGPipeline",
        "HunyuanDiTPAGPipeline",
        "StableDiffusion3PAGPipeline",
        "StableDiffusionPAGPipeline",
        "StableDiffusionControlNetPAGPipeline",
        "StableDiffusionXLPAGPipeline",
        "StableDiffusionXLPAGInpaintPipeline",
        "StableDiffusionXLControlNetPAGPipeline",
        "StableDiffusionXLPAGImg2ImgPipeline",
        "PixArtSigmaPAGPipeline",
    ],
    "pixart_alpha": [
        "PixArtAlphaPipeline",
        "PixArtSigmaPipeline",
    ],
    "shap_e": ["ShapEImg2ImgPipeline", "ShapEPipeline"],
    "stable_cascade": [
        "StableCascadeCombinedPipeline",
        "StableCascadeDecoderPipeline",
        "StableCascadePriorPipeline",
    ],
    "stable_diffusion": [
        "CLIPImageProjection",
        "StableDiffusionDepth2ImgPipeline",
        "StableDiffusionImageVariationPipeline",
        "StableDiffusionImg2ImgPipeline",
        "StableDiffusionInpaintPipeline",
        "StableDiffusionInstructPix2PixPipeline",
        "StableDiffusionLatentUpscalePipeline",
        "StableDiffusionPipeline",
        "StableDiffusionUpscalePipeline",
    ],
    "stable_diffusion_3": [
        "StableDiffusion3Pipeline",
        "StableDiffusion3Img2ImgPipeline",
        "StableDiffusion3InpaintPipeline",
    ],
    "stable_diffusion_gligen": [
        "StableDiffusionGLIGENPipeline",
        "StableDiffusionGLIGENTextImagePipeline",
    ],
    "stable_diffusion_xl": [
        "StableDiffusionXLInpaintPipeline",
        "StableDiffusionXLImg2ImgPipeline",
        "StableDiffusionXLInstructPix2PixPipeline",
        "StableDiffusionXLPipeline",
    ],
    "stable_diffusion_diffedit": ["StableDiffusionDiffEditPipeline"],
    "stable_video_diffusion": ["StableVideoDiffusionPipeline"],
    "t2i_adapter": [
        "StableDiffusionAdapterPipeline",
        "StableDiffusionXLAdapterPipeline",
    ],
    "unclip": ["UnCLIPImageVariationPipeline", "UnCLIPPipeline"],
    "wuerstchen": [
        "WuerstchenCombinedPipeline",
        "WuerstchenDecoderPipeline",
        "WuerstchenPriorPipeline",
    ],
    "pipeline_utils": [
        "DiffusionPipeline",
        "ImagePipelineOutput",
        "StableDiffusionMixin",
    ],
}

if TYPE_CHECKING:
    from .animatediff import (
        AnimateDiffControlNetPipeline,
        AnimateDiffPipeline,
        AnimateDiffSDXLPipeline,
        AnimateDiffSparseControlNetPipeline,
        AnimateDiffVideoToVideoPipeline,
    )
    from .aura_flow import AuraFlowPipeline
    from .auto_pipeline import AutoPipelineForImage2Image, AutoPipelineForInpainting, AutoPipelineForText2Image
    from .blip_diffusion import BlipDiffusionPipeline
    from .cogvideo import CogVideoXImageToVideoPipeline, CogVideoXPipeline, CogVideoXVideoToVideoPipeline
    from .consistency_models import ConsistencyModelPipeline
    from .controlnet import (
        BlipDiffusionControlNetPipeline,
        StableDiffusionControlNetImg2ImgPipeline,
        StableDiffusionControlNetInpaintPipeline,
        StableDiffusionControlNetPipeline,
        StableDiffusionXLControlNetImg2ImgPipeline,
        StableDiffusionXLControlNetInpaintPipeline,
        StableDiffusionXLControlNetPipeline,
    )
    from .controlnet_hunyuandit import HunyuanDiTControlNetPipeline
    from .controlnet_sd3 import StableDiffusion3ControlNetPipeline
    from .controlnet_xs import StableDiffusionControlNetXSPipeline, StableDiffusionXLControlNetXSPipeline
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
    from .dit import DiTPipeline
    from .flux import FluxPipeline
    from .hunyuandit import HunyuanDiTPipeline
    from .i2vgen_xl import I2VGenXLPipeline
    from .kandinsky import (
        KandinskyCombinedPipeline,
        KandinskyImg2ImgCombinedPipeline,
        KandinskyImg2ImgPipeline,
        KandinskyInpaintCombinedPipeline,
        KandinskyInpaintPipeline,
        KandinskyPipeline,
        KandinskyPriorPipeline,
    )
    from .kandinsky2_2 import (
        KandinskyV22CombinedPipeline,
        KandinskyV22ControlnetImg2ImgPipeline,
        KandinskyV22ControlnetPipeline,
        KandinskyV22Img2ImgCombinedPipeline,
        KandinskyV22Img2ImgPipeline,
        KandinskyV22InpaintCombinedPipeline,
        KandinskyV22InpaintPipeline,
        KandinskyV22Pipeline,
        KandinskyV22PriorEmb2EmbPipeline,
        KandinskyV22PriorPipeline,
    )
    from .kandinsky3 import Kandinsky3Img2ImgPipeline, Kandinsky3Pipeline
    from .kolors import KolorsImg2ImgPipeline, KolorsPipeline
    from .latent_consistency_models import LatentConsistencyModelImg2ImgPipeline, LatentConsistencyModelPipeline
    from .latent_diffusion import LDMSuperResolutionPipeline, LDMTextToImagePipeline
    from .latte import LattePipeline
    from .lumina import LuminaText2ImgPipeline
    from .marigold import MarigoldDepthPipeline, MarigoldNormalsPipeline
    from .pag import (
        AnimateDiffPAGPipeline,
        HunyuanDiTPAGPipeline,
        KolorsPAGPipeline,
        PixArtSigmaPAGPipeline,
        StableDiffusion3PAGPipeline,
        StableDiffusionControlNetPAGPipeline,
        StableDiffusionPAGPipeline,
        StableDiffusionXLControlNetPAGPipeline,
        StableDiffusionXLPAGImg2ImgPipeline,
        StableDiffusionXLPAGInpaintPipeline,
        StableDiffusionXLPAGPipeline,
    )
    from .pipeline_utils import DiffusionPipeline, ImagePipelineOutput, StableDiffusionMixin
    from .pixart_alpha import PixArtAlphaPipeline, PixArtSigmaPipeline
    from .shap_e import ShapEImg2ImgPipeline, ShapEPipeline
    from .stable_cascade import StableCascadeCombinedPipeline, StableCascadeDecoderPipeline, StableCascadePriorPipeline
    from .stable_diffusion import (
        CLIPImageProjection,
        StableDiffusionDepth2ImgPipeline,
        StableDiffusionImageVariationPipeline,
        StableDiffusionImg2ImgPipeline,
        StableDiffusionInpaintPipeline,
        StableDiffusionInstructPix2PixPipeline,
        StableDiffusionLatentUpscalePipeline,
        StableDiffusionPipeline,
        StableDiffusionUpscalePipeline,
    )
    from .stable_diffusion_3 import (
        StableDiffusion3Img2ImgPipeline,
        StableDiffusion3InpaintPipeline,
        StableDiffusion3Pipeline,
    )
    from .stable_diffusion_diffedit import StableDiffusionDiffEditPipeline
    from .stable_diffusion_gligen import StableDiffusionGLIGENPipeline, StableDiffusionGLIGENTextImagePipeline
    from .stable_diffusion_xl import (
        StableDiffusionXLImg2ImgPipeline,
        StableDiffusionXLInpaintPipeline,
        StableDiffusionXLInstructPix2PixPipeline,
        StableDiffusionXLPipeline,
    )
    from .stable_video_diffusion import StableVideoDiffusionPipeline
    from .t2i_adapter import StableDiffusionAdapterPipeline, StableDiffusionXLAdapterPipeline
    from .unclip import UnCLIPImageVariationPipeline, UnCLIPPipeline
    from .wuerstchen import WuerstchenCombinedPipeline, WuerstchenDecoderPipeline, WuerstchenPriorPipeline
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
