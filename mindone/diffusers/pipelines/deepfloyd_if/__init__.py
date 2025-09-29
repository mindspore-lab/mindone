"""Adapted from https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/deepfloyd_if/__init__.py."""

from typing import TYPE_CHECKING

from ...utils import _LazyModule

_dummy_objects = {}
_import_structure = {
    "timesteps": [
        "fast27_timesteps",
        "smart100_timesteps",
        "smart185_timesteps",
        "smart27_timesteps",
        "smart50_timesteps",
        "super100_timesteps",
        "super27_timesteps",
        "super40_timesteps",
    ]
}

_import_structure["pipeline_if"] = ["IFPipeline"]
_import_structure["pipeline_if_img2img"] = ["IFImg2ImgPipeline"]
_import_structure["pipeline_if_img2img_superresolution"] = ["IFImg2ImgSuperResolutionPipeline"]
_import_structure["pipeline_if_inpainting"] = ["IFInpaintingPipeline"]
_import_structure["pipeline_if_inpainting_superresolution"] = ["IFInpaintingSuperResolutionPipeline"]
_import_structure["pipeline_if_superresolution"] = ["IFSuperResolutionPipeline"]
_import_structure["pipeline_output"] = ["IFPipelineOutput"]
_import_structure["safety_checker"] = ["IFSafetyChecker"]
_import_structure["watermark"] = ["IFWatermarker"]


if TYPE_CHECKING:
    from .pipeline_if import IFPipeline
    from .pipeline_if_img2img import IFImg2ImgPipeline
    from .pipeline_if_img2img_superresolution import IFImg2ImgSuperResolutionPipeline
    from .pipeline_if_inpainting import IFInpaintingPipeline
    from .pipeline_if_inpainting_superresolution import IFInpaintingSuperResolutionPipeline
    from .pipeline_if_superresolution import IFSuperResolutionPipeline
    from .pipeline_output import IFPipelineOutput
    from .safety_checker import IFSafetyChecker
    from .timesteps import (
        fast27_timesteps,
        smart27_timesteps,
        smart50_timesteps,
        smart100_timesteps,
        smart185_timesteps,
        super27_timesteps,
        super40_timesteps,
        super100_timesteps,
    )
    from .watermark import IFWatermarker

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )

    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
