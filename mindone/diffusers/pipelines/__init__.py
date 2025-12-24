"""Adapted from https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/__init__.py."""

from typing import TYPE_CHECKING

from ..utils import _LazyModule

# These modules contain pipelines from multiple libraries/frameworks
_import_structure = {
    "allegro": ["AllegroPipeline"],
    "amused": ["AmusedImg2ImgPipeline", "AmusedInpaintPipeline", "AmusedPipeline"],
    "animatediff": [
        "AnimateDiffPipeline",
        "AnimateDiffControlNetPipeline",
        "AnimateDiffSDXLPipeline",
        "AnimateDiffSparseControlNetPipeline",
        "AnimateDiffVideoToVideoPipeline",
        "AnimateDiffVideoToVideoControlNetPipeline",
    ],
    "audioldm": ["AudioLDMPipeline"],
    "audioldm2": [
        "AudioLDM2Pipeline",
        "AudioLDM2ProjectionModel",
        "AudioLDM2UNet2DConditionModel",
    ],
    "aura_flow": ["AuraFlowPipeline"],
    "auto_pipeline": [
        "AutoPipelineForImage2Image",
        "AutoPipelineForInpainting",
        "AutoPipelineForText2Image",
    ],
    "blip_diffusion": ["BlipDiffusionPipeline"],
    "bria": ["BriaPipeline"],
    "chroma": ["ChromaPipeline", "ChromaImg2ImgPipeline"],
    "cogvideo": [
        "CogVideoXPipeline",
        "CogVideoXImageToVideoPipeline",
        "CogVideoXVideoToVideoPipeline",
        "CogVideoXFunControlPipeline",
    ],
    "consistency_models": ["ConsistencyModelPipeline"],
    "cogview3": ["CogView3PlusPipeline"],
    "cogview4": ["CogView4Pipeline", "CogView4ControlPipeline"],
    "consisid": ["ConsisIDPipeline"],
    "cosmos": [
        "CosmosTextToWorldPipeline",
        "CosmosVideoToWorldPipeline",
        "Cosmos2TextToImagePipeline",
        "Cosmos2VideoToWorldPipeline",
    ],
    "controlnet": [
        "BlipDiffusionControlNetPipeline",
        "StableDiffusionControlNetImg2ImgPipeline",
        "StableDiffusionControlNetInpaintPipeline",
        "StableDiffusionControlNetPipeline",
        "StableDiffusionXLControlNetImg2ImgPipeline",
        "StableDiffusionXLControlNetInpaintPipeline",
        "StableDiffusionXLControlNetPipeline",
        "StableDiffusionXLControlNetUnionPipeline",
        "StableDiffusionXLControlNetUnionInpaintPipeline",
        "StableDiffusionXLControlNetUnionImg2ImgPipeline",
    ],
    "controlnet_hunyuandit": ["HunyuanDiTControlNetPipeline"],
    "controlnet_xs": [
        "StableDiffusionControlNetXSPipeline",
        "StableDiffusionXLControlNetXSPipeline",
    ],
    "controlnet_sd3": [
        "StableDiffusion3ControlNetPipeline",
        "StableDiffusion3ControlNetInpaintingPipeline",
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
    "easyanimate": [
        "EasyAnimatePipeline",
        "EasyAnimateInpaintPipeline",
        "EasyAnimateControlPipeline",
    ],
    "flux": [
        "FluxControlImg2ImgPipeline",
        "FluxControlInpaintPipeline",
        "FluxControlNetImg2ImgPipeline",
        "FluxControlNetInpaintPipeline",
        "FluxControlNetPipeline",
        "FluxControlPipeline",
        "FluxFillPipeline",
        "FluxImg2ImgPipeline",
        "FluxInpaintPipeline",
        "FluxPipeline",
        "FluxPriorReduxPipeline",
        "ReduxImageEncoder",
        "FluxKontextPipeline",
        "FluxKontextInpaintPipeline",
    ],
    "flux2": ["Flux2Pipeline"],
    "hidream_image": ["HiDreamImagePipeline"],
    "hunyuandit": ["HunyuanDiTPipeline"],
    "hunyuan_video": [
        "HunyuanVideoPipeline",
        "HunyuanSkyreelsImageToVideoPipeline",
        "HunyuanVideoImageToVideoPipeline",
        "HunyuanVideoFramepackPipeline",
    ],
    "i2vgen_xl": ["I2VGenXLPipeline"],
    "latent_diffusion": ["LDMSuperResolutionPipeline", "LDMTextToImagePipeline"],
    "ledits_pp": ["LEditsPPPipelineStableDiffusion", "LEditsPPPipelineStableDiffusionXL"],
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
    "kandinsky5": ["Kandinsky5T2VPipeline"],
    "kolors": [
        "KolorsPipeline",
        "KolorsImg2ImgPipeline",
    ],
    "latent_consistency_models": [
        "LatentConsistencyModelImg2ImgPipeline",
        "LatentConsistencyModelPipeline",
    ],
    "latte": ["LattePipeline"],
    "ltx": ["LTXPipeline", "LTXImageToVideoPipeline", "LTXConditionPipeline", "LTXLatentUpsamplePipeline"],
    "lumina": ["LuminaPipeline", "LuminaText2ImgPipeline"],
    "lumina2": ["Lumina2Pipeline", "Lumina2Text2ImgPipeline"],
    "lucy": ["LucyEditPipeline"],
    "marigold": [
        "MarigoldDepthPipeline",
        "MarigoldIntrinsicsPipeline",
        "MarigoldNormalsPipeline",
    ],
    "mochi": ["MochiPipeline"],
    "musicldm": ["MusicLDMPipeline"],
    "omnigen": ["OmniGenPipeline"],
    "visualcloze": ["VisualClozePipeline", "VisualClozeGenerationPipeline"],
    "pag": [
        "StableDiffusionControlNetPAGInpaintPipeline",
        "AnimateDiffPAGPipeline",
        "KolorsPAGPipeline",
        "HunyuanDiTPAGPipeline",
        "SanaPAGPipeline",
        "StableDiffusion3PAGPipeline",
        "StableDiffusion3PAGImg2ImgPipeline",
        "StableDiffusionPAGPipeline",
        "StableDiffusionPAGImg2ImgPipeline",
        "StableDiffusionPAGInpaintPipeline",
        "StableDiffusionControlNetPAGPipeline",
        "StableDiffusionXLPAGPipeline",
        "StableDiffusionXLPAGInpaintPipeline",
        "StableDiffusionXLControlNetPAGImg2ImgPipeline",
        "StableDiffusionXLControlNetPAGPipeline",
        "StableDiffusionXLPAGImg2ImgPipeline",
        "PixArtSigmaPAGPipeline",
    ],
    "paint_by_example": ["PaintByExamplePipeline"],
    "pia": ["PIAPipeline"],
    "pixart_alpha": [
        "PixArtAlphaPipeline",
        "PixArtSigmaPipeline",
    ],
    "qwenimage": [
        "QwenImagePipeline",
        "QwenImageImg2ImgPipeline",
        "QwenImageInpaintPipeline",
        "QwenImageEditPipeline",
        "QwenImageEditPlusPipeline",
        "QwenImageEditInpaintPipeline",
        "QwenImageControlNetInpaintPipeline",
        "QwenImageControlNetPipeline",
    ],
    "sana": ["SanaPipeline", "SanaSprintPipeline", "SanaControlNetPipeline", "SanaSprintImg2ImgPipeline"],
    "semantic_stable_diffusion": ["SemanticStableDiffusionPipeline"],
    "shap_e": ["ShapEImg2ImgPipeline", "ShapEPipeline"],
    "stable_audio": ["StableAudioProjectionModel", "StableAudioPipeline"],
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
        "StableUnCLIPImg2ImgPipeline",
        "StableUnCLIPPipeline",
    ],
    "stable_diffusion_3": [
        "StableDiffusion3Pipeline",
        "StableDiffusion3Img2ImgPipeline",
        "StableDiffusion3InpaintPipeline",
    ],
    "stable_diffusion_attend_and_excite": ["StableDiffusionAttendAndExcitePipeline"],
    "stable_diffusion_safe": ["StableDiffusionPipelineSafe"],
    "stable_diffusion_sag": ["StableDiffusionSAGPipeline"],
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
    "stable_diffusion_k_diffusion": [
        "StableDiffusionKDiffusionPipeline",
        "StableDiffusionXLKDiffusionPipeline",
    ],
    "stable_diffusion_diffedit": ["StableDiffusionDiffEditPipeline"],
    "stable_diffusion_ldm3d": ["StableDiffusionLDM3DPipeline"],
    "stable_diffusion_panorama": ["StableDiffusionPanoramaPipeline"],
    "stable_video_diffusion": ["StableVideoDiffusionPipeline"],
    "t2i_adapter": [
        "StableDiffusionAdapterPipeline",
        "StableDiffusionXLAdapterPipeline",
    ],
    "text_to_video_synthesis": [
        "TextToVideoSDPipeline",
        "TextToVideoZeroSDXLPipeline",
        "VideoToVideoSDPipeline",
        "TextToVideoZeroPipeline",
    ],
    "unclip": ["UnCLIPImageVariationPipeline", "UnCLIPPipeline"],
    "unidiffuser": [
        "ImageTextPipelineOutput",
        "UniDiffuserModel",
        "UniDiffuserPipeline",
        "UniDiffuserTextDecoder",
    ],
    "wuerstchen": [
        "WuerstchenCombinedPipeline",
        "WuerstchenDecoderPipeline",
        "WuerstchenPriorPipeline",
    ],
    "wan": ["WanPipeline", "WanImageToVideoPipeline", "WanVideoToVideoPipeline", "WanVACEPipeline"],
    "skyreels_v2": [
        "SkyReelsV2DiffusionForcingPipeline",
        "SkyReelsV2DiffusionForcingImageToVideoPipeline",
        "SkyReelsV2DiffusionForcingVideoToVideoPipeline",
        "SkyReelsV2ImageToVideoPipeline",
        "SkyReelsV2Pipeline",
    ],
    "pipeline_utils": [
        "AudioPipelineOutput",
        "DiffusionPipeline",
        "ImagePipelineOutput",
        "StableDiffusionMixin",
    ],
}

if TYPE_CHECKING:
    from .allegro import AllegroPipeline
    from .amused import AmusedImg2ImgPipeline, AmusedInpaintPipeline, AmusedPipeline
    from .animatediff import (
        AnimateDiffControlNetPipeline,
        AnimateDiffPipeline,
        AnimateDiffSDXLPipeline,
        AnimateDiffSparseControlNetPipeline,
        AnimateDiffVideoToVideoControlNetPipeline,
        AnimateDiffVideoToVideoPipeline,
    )
    from .audioldm import AudioLDMPipeline
    from .audioldm2 import AudioLDM2Pipeline, AudioLDM2ProjectionModel, AudioLDM2UNet2DConditionModel
    from .aura_flow import AuraFlowPipeline
    from .auto_pipeline import AutoPipelineForImage2Image, AutoPipelineForInpainting, AutoPipelineForText2Image
    from .blip_diffusion import BlipDiffusionPipeline
    from .bria import BriaPipeline
    from .chroma import ChromaImg2ImgPipeline, ChromaPipeline
    from .cogvideo import (
        CogVideoXFunControlPipeline,
        CogVideoXImageToVideoPipeline,
        CogVideoXPipeline,
        CogVideoXVideoToVideoPipeline,
    )
    from .cogview3 import CogView3PlusPipeline
    from .cogview4 import CogView4ControlPipeline, CogView4Pipeline
    from .consisid import ConsisIDPipeline
    from .consistency_models import ConsistencyModelPipeline
    from .controlnet import (
        BlipDiffusionControlNetPipeline,
        StableDiffusionControlNetImg2ImgPipeline,
        StableDiffusionControlNetInpaintPipeline,
        StableDiffusionControlNetPipeline,
        StableDiffusionXLControlNetImg2ImgPipeline,
        StableDiffusionXLControlNetInpaintPipeline,
        StableDiffusionXLControlNetPipeline,
        StableDiffusionXLControlNetUnionImg2ImgPipeline,
        StableDiffusionXLControlNetUnionInpaintPipeline,
        StableDiffusionXLControlNetUnionPipeline,
    )
    from .controlnet_hunyuandit import HunyuanDiTControlNetPipeline
    from .controlnet_sd3 import StableDiffusion3ControlNetInpaintingPipeline, StableDiffusion3ControlNetPipeline
    from .controlnet_xs import StableDiffusionControlNetXSPipeline, StableDiffusionXLControlNetXSPipeline
    from .cosmos import (
        Cosmos2TextToImagePipeline,
        Cosmos2VideoToWorldPipeline,
        CosmosTextToWorldPipeline,
        CosmosVideoToWorldPipeline,
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
    from .dit import DiTPipeline
    from .easyanimate import EasyAnimateControlPipeline, EasyAnimateInpaintPipeline, EasyAnimatePipeline
    from .flux import (
        FluxControlImg2ImgPipeline,
        FluxControlInpaintPipeline,
        FluxControlNetImg2ImgPipeline,
        FluxControlNetInpaintPipeline,
        FluxControlNetPipeline,
        FluxControlPipeline,
        FluxFillPipeline,
        FluxImg2ImgPipeline,
        FluxInpaintPipeline,
        FluxKontextInpaintPipeline,
        FluxKontextPipeline,
        FluxPipeline,
        FluxPriorReduxPipeline,
        ReduxImageEncoder,
    )
    from .flux2 import Flux2Pipeline
    from .hidream_image import HiDreamImagePipeline
    from .hunyuan_video import (
        HunyuanSkyreelsImageToVideoPipeline,
        HunyuanVideoFramepackPipeline,
        HunyuanVideoImageToVideoPipeline,
        HunyuanVideoPipeline,
    )
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
    from .kandinsky5 import Kandinsky5T2VPipeline
    from .kolors import KolorsImg2ImgPipeline, KolorsPipeline
    from .latent_consistency_models import LatentConsistencyModelImg2ImgPipeline, LatentConsistencyModelPipeline
    from .latent_diffusion import LDMSuperResolutionPipeline, LDMTextToImagePipeline
    from .latte import LattePipeline
    from .ledits_pp import (
        LEditsPPDiffusionPipelineOutput,
        LEditsPPInversionPipelineOutput,
        LEditsPPPipelineStableDiffusion,
        LEditsPPPipelineStableDiffusionXL,
    )
    from .ltx import LTXConditionPipeline, LTXImageToVideoPipeline, LTXLatentUpsamplePipeline, LTXPipeline
    from .lucy import LucyEditPipeline
    from .lumina import LuminaPipeline, LuminaText2ImgPipeline
    from .lumina2 import Lumina2Pipeline, Lumina2Text2ImgPipeline
    from .marigold import MarigoldDepthPipeline, MarigoldIntrinsicsPipeline, MarigoldNormalsPipeline
    from .mochi import MochiPipeline
    from .musicldm import MusicLDMPipeline
    from .omnigen import OmniGenPipeline
    from .pag import (
        AnimateDiffPAGPipeline,
        HunyuanDiTPAGPipeline,
        KolorsPAGPipeline,
        PixArtSigmaPAGPipeline,
        SanaPAGPipeline,
        StableDiffusion3PAGImg2ImgPipeline,
        StableDiffusion3PAGPipeline,
        StableDiffusionControlNetPAGInpaintPipeline,
        StableDiffusionControlNetPAGPipeline,
        StableDiffusionPAGImg2ImgPipeline,
        StableDiffusionPAGInpaintPipeline,
        StableDiffusionPAGPipeline,
        StableDiffusionXLControlNetPAGImg2ImgPipeline,
        StableDiffusionXLControlNetPAGPipeline,
        StableDiffusionXLPAGImg2ImgPipeline,
        StableDiffusionXLPAGInpaintPipeline,
        StableDiffusionXLPAGPipeline,
    )
    from .paint_by_example import PaintByExamplePipeline
    from .pia import PIAPipeline
    from .pipeline_utils import AudioPipelineOutput, DiffusionPipeline, ImagePipelineOutput, StableDiffusionMixin
    from .pixart_alpha import PixArtAlphaPipeline, PixArtSigmaPipeline
    from .qwenimage import (
        QwenImageControlNetInpaintPipeline,
        QwenImageControlNetPipeline,
        QwenImageEditInpaintPipeline,
        QwenImageEditPipeline,
        QwenImageEditPlusPipeline,
        QwenImageImg2ImgPipeline,
        QwenImageInpaintPipeline,
        QwenImagePipeline,
    )
    from .sana import SanaControlNetPipeline, SanaPipeline, SanaSprintImg2ImgPipeline, SanaSprintPipeline
    from .semantic_stable_diffusion import SemanticStableDiffusionPipeline
    from .shap_e import ShapEImg2ImgPipeline, ShapEPipeline
    from .skyreels_v2 import (
        SkyReelsV2DiffusionForcingImageToVideoPipeline,
        SkyReelsV2DiffusionForcingPipeline,
        SkyReelsV2DiffusionForcingVideoToVideoPipeline,
        SkyReelsV2ImageToVideoPipeline,
        SkyReelsV2Pipeline,
    )
    from .stable_audio import StableAudioPipeline, StableAudioProjectionModel
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
        StableUnCLIPImg2ImgPipeline,
        StableUnCLIPPipeline,
    )
    from .stable_diffusion_3 import (
        StableDiffusion3Img2ImgPipeline,
        StableDiffusion3InpaintPipeline,
        StableDiffusion3Pipeline,
    )
    from .stable_diffusion_attend_and_excite import StableDiffusionAttendAndExcitePipeline
    from .stable_diffusion_diffedit import StableDiffusionDiffEditPipeline
    from .stable_diffusion_gligen import StableDiffusionGLIGENPipeline, StableDiffusionGLIGENTextImagePipeline
    from .stable_diffusion_k_diffusion import StableDiffusionKDiffusionPipeline, StableDiffusionXLKDiffusionPipeline
    from .stable_diffusion_ldm3d import StableDiffusionLDM3DPipeline
    from .stable_diffusion_panorama import StableDiffusionPanoramaPipeline
    from .stable_diffusion_safe import StableDiffusionPipelineSafe
    from .stable_diffusion_sag import StableDiffusionSAGPipeline
    from .stable_diffusion_xl import (
        StableDiffusionXLImg2ImgPipeline,
        StableDiffusionXLInpaintPipeline,
        StableDiffusionXLInstructPix2PixPipeline,
        StableDiffusionXLPipeline,
    )
    from .stable_video_diffusion import StableVideoDiffusionPipeline
    from .t2i_adapter import StableDiffusionAdapterPipeline, StableDiffusionXLAdapterPipeline
    from .text_to_video_synthesis import (
        TextToVideoSDPipeline,
        TextToVideoZeroPipeline,
        TextToVideoZeroSDXLPipeline,
        VideoToVideoSDPipeline,
    )
    from .unclip import UnCLIPImageVariationPipeline, UnCLIPPipeline
    from .unidiffuser import ImageTextPipelineOutput, UniDiffuserModel, UniDiffuserPipeline, UniDiffuserTextDecoder
    from .visualcloze import VisualClozeGenerationPipeline, VisualClozePipeline
    from .wan import WanImageToVideoPipeline, WanPipeline, WanVACEPipeline, WanVideoToVideoPipeline
    from .wuerstchen import WuerstchenCombinedPipeline, WuerstchenDecoderPipeline, WuerstchenPriorPipeline
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
