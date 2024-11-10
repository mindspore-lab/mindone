from typing import TYPE_CHECKING

from ...utils import _LazyModule

_import_structure = {}

_import_structure["pipeline_pag_hunyuandit"] = ["HunyuanDiTPAGPipeline"]
_import_structure["pipeline_pag_kolors"] = ["KolorsPAGPipeline"]
_import_structure["pipeline_pag_sd"] = ["StableDiffusionPAGPipeline"]
_import_structure["pipeline_pag_sd_3"] = ["StableDiffusion3PAGPipeline"]
_import_structure["pipeline_pag_sd_animatediff"] = ["AnimateDiffPAGPipeline"]

if TYPE_CHECKING:
    from .pipeline_pag_hunyuandit import HunyuanDiTPAGPipeline
    from .pipeline_pag_kolors import KolorsPAGPipeline
    from .pipeline_pag_sd import StableDiffusionPAGPipeline
    from .pipeline_pag_sd_3 import StableDiffusion3PAGPipeline
    from .pipeline_pag_sd_animatediff import AnimateDiffPAGPipeline

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
