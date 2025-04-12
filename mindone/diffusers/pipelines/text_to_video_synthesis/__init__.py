from typing import TYPE_CHECKING

from ...utils import _LazyModule

_import_structure = {}

_import_structure["pipeline_output"] = ["TextToVideoSDPipelineOutput"]
_import_structure["pipeline_text_to_video_synth"] = ["TextToVideoSDPipeline"]
_import_structure["pipeline_text_to_video_synth_img2img"] = ["VideoToVideoSDPipeline"]
_import_structure["pipeline_text_to_video_zero"] = ["TextToVideoZeroPipeline"]
_import_structure["pipeline_text_to_video_zero_sdxl"] = ["TextToVideoZeroSDXLPipeline"]

if TYPE_CHECKING:
    from .pipeline_output import TextToVideoSDPipelineOutput
    from .pipeline_text_to_video_synth import TextToVideoSDPipeline
    from .pipeline_text_to_video_synth_img2img import VideoToVideoSDPipeline
    from .pipeline_text_to_video_zero import TextToVideoZeroPipeline
    from .pipeline_text_to_video_zero_sdxl import TextToVideoZeroSDXLPipeline

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
