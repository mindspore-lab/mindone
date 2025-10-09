"""Adapted from https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/skyreels_v2/__init__.py."""
from typing import TYPE_CHECKING

from ...utils import _LazyModule

_dummy_objects = {}
_import_structure = {}

_import_structure["pipeline_skyreels_v2"] = ["SkyReelsV2Pipeline"]
_import_structure["pipeline_skyreels_v2_diffusion_forcing"] = ["SkyReelsV2DiffusionForcingPipeline"]
_import_structure["pipeline_skyreels_v2_diffusion_forcing_i2v"] = ["SkyReelsV2DiffusionForcingImageToVideoPipeline"]
_import_structure["pipeline_skyreels_v2_diffusion_forcing_v2v"] = ["SkyReelsV2DiffusionForcingVideoToVideoPipeline"]
_import_structure["pipeline_skyreels_v2_i2v"] = ["SkyReelsV2ImageToVideoPipeline"]

if TYPE_CHECKING:
    from .pipeline_skyreels_v2 import SkyReelsV2Pipeline
    from .pipeline_skyreels_v2_diffusion_forcing import SkyReelsV2DiffusionForcingPipeline
    from .pipeline_skyreels_v2_diffusion_forcing_i2v import SkyReelsV2DiffusionForcingImageToVideoPipeline
    from .pipeline_skyreels_v2_diffusion_forcing_v2v import SkyReelsV2DiffusionForcingVideoToVideoPipeline
    from .pipeline_skyreels_v2_i2v import SkyReelsV2ImageToVideoPipeline

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
