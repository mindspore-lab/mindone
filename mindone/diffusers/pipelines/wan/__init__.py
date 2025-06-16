from typing import TYPE_CHECKING

from ...utils import _LazyModule, is_transformers_available

_dummy_objects = {}
_import_structure = {}

_import_structure["pipeline_wan"] = ["WanPipeline"]
_import_structure["pipeline_wan_i2v"] = ["WanImageToVideoPipeline"]
_import_structure["pipeline_wan_video2video"] = ["WanVideoToVideoPipeline"]

if TYPE_CHECKING:
    from .pipeline_wan import WanPipeline
    from .pipeline_wan_i2v import WanImageToVideoPipeline
    from .pipeline_wan_video2video import WanVideoToVideoPipeline

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
