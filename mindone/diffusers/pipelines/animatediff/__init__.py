from typing import TYPE_CHECKING

from ...utils import _LazyModule

_dummy_objects = {}
_import_structure = {"pipeline_output": ["AnimateDiffPipelineOutput"]}


_import_structure["pipeline_animatediff"] = ["AnimateDiffPipeline"]
_import_structure["pipeline_animatediff_controlnet"] = ["AnimateDiffControlNetPipeline"]
_import_structure["pipeline_animatediff_sdxl"] = ["AnimateDiffSDXLPipeline"]
_import_structure["pipeline_animatediff_sparsectrl"] = ["AnimateDiffSparseControlNetPipeline"]
_import_structure["pipeline_animatediff_video2video"] = ["AnimateDiffVideoToVideoPipeline"]


if TYPE_CHECKING:
    from .pipeline_animatediff import AnimateDiffPipeline
    from .pipeline_animatediff_controlnet import AnimateDiffControlNetPipeline
    from .pipeline_animatediff_sdxl import AnimateDiffSDXLPipeline
    from .pipeline_animatediff_sparsectrl import AnimateDiffSparseControlNetPipeline
    from .pipeline_animatediff_video2video import AnimateDiffVideoToVideoPipeline
    from .pipeline_output import AnimateDiffPipelineOutput
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
