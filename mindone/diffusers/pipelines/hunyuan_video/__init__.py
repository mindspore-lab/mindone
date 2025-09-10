"""Adapted from https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/hunyuan_video/__init__.py."""

from typing import TYPE_CHECKING

from ...utils import _LazyModule

_dummy_objects = {}
_import_structure = {}

_import_structure["pipeline_hunyuan_skyreels_image2video"] = ["HunyuanSkyreelsImageToVideoPipeline"]
_import_structure["pipeline_hunyuan_video"] = ["HunyuanVideoPipeline"]
_import_structure["pipeline_hunyuan_video_framepack"] = ["HunyuanVideoFramepackPipeline"]
_import_structure["pipeline_hunyuan_video_image2video"] = ["HunyuanVideoImageToVideoPipeline"]

if TYPE_CHECKING:
    from .pipeline_hunyuan_skyreels_image2video import HunyuanSkyreelsImageToVideoPipeline
    from .pipeline_hunyuan_video import HunyuanVideoPipeline
    from .pipeline_hunyuan_video_framepack import HunyuanVideoFramepackPipeline
    from .pipeline_hunyuan_video_image2video import HunyuanVideoImageToVideoPipeline
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
