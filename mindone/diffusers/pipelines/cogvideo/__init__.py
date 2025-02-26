from typing import TYPE_CHECKING

from ...utils import _LazyModule

_import_structure = {
    "pipeline_cogvideox": ["CogVideoXPipeline"],
    "pipeline_cogvideox_image2video": ["CogVideoXImageToVideoPipeline"],
    "pipeline_cogvideox_video2video": ["CogVideoXVideoToVideoPipeline"],
}
if TYPE_CHECKING:
    from .pipeline_cogvideox import CogVideoXPipeline
    from .pipeline_cogvideox_image2video import CogVideoXImageToVideoPipeline
    from .pipeline_cogvideox_video2video import CogVideoXVideoToVideoPipeline
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
