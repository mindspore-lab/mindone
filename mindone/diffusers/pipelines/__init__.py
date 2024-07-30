from typing import TYPE_CHECKING

from ..utils import _LazyModule

# These modules contain pipelines from multiple libraries/frameworks
_import_structure = {
    "blip_diffusion": ["BlipDiffusionPipeline"],
    "consistency_models": ["ConsistencyModelPipeline"],
    "dance_diffusion": ["DanceDiffusionPipeline"],
    "ddim": ["DDIMPipeline"],
    "ddpm": ["DDPMPipeline"],
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
    "latent_consistency_models": [
        "LatentConsistencyModelImg2ImgPipeline",
        "LatentConsistencyModelPipeline",
    ],
    "stable_cascade": [
        "StableCascadeCombinedPipeline",
        "StableCascadeDecoderPipeline",
        "StableCascadePriorPipeline",
    ],
    "stable_diffusion": [
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
    ],
    "stable_diffusion_xl": [
        "StableDiffusionXLInpaintPipeline",
        "StableDiffusionXLImg2ImgPipeline",
        "StableDiffusionXLInstructPix2PixPipeline",
        "StableDiffusionXLPipeline",
    ],
    "stable_video_diffusion": ["StableVideoDiffusionPipeline"],
    "wuerstchen": [
        "WuerstchenCombinedPipeline",
        "WuerstchenDecoderPipeline",
        "WuerstchenPriorPipeline",
    ],
    "pipeline_utils": [
        "DiffusionPipeline",
        "ImagePipelineOutput",
    ],
}

if TYPE_CHECKING:
    from .blip_diffusion import BlipDiffusionPipeline
    from .consistency_models import ConsistencyModelPipeline
    from .ddim import DDIMPipeline
    from .ddpm import DDPMPipeline
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
    from .latent_consistency_models import LatentConsistencyModelImg2ImgPipeline, LatentConsistencyModelPipeline
    from .pipeline_utils import DiffusionPipeline, ImagePipelineOutput
    from .stable_cascade import StableCascadeCombinedPipeline, StableCascadeDecoderPipeline, StableCascadePriorPipeline
    from .stable_diffusion import (
        StableDiffusionDepth2ImgPipeline,
        StableDiffusionImageVariationPipeline,
        StableDiffusionImg2ImgPipeline,
        StableDiffusionInpaintPipeline,
        StableDiffusionInstructPix2PixPipeline,
        StableDiffusionLatentUpscalePipeline,
        StableDiffusionPipeline,
        StableDiffusionUpscalePipeline,
    )
    from .stable_diffusion_3 import StableDiffusion3Pipeline
    from .stable_diffusion_xl import (
        StableDiffusionXLImg2ImgPipeline,
        StableDiffusionXLInpaintPipeline,
        StableDiffusionXLInstructPix2PixPipeline,
        StableDiffusionXLPipeline,
    )
    from .stable_video_diffusion import StableVideoDiffusionPipeline
    from .wuerstchen import WuerstchenCombinedPipeline, WuerstchenDecoderPipeline, WuerstchenPriorPipeline
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
