from typing import TYPE_CHECKING

from ...utils import _LazyModule

_dummy_objects = {}
_import_structure = {}

_import_structure["pipeline_ltx"] = ["LTXPipeline"]
_import_structure["pipeline_ltx_condition"] = ["LTXConditionPipeline"]
_import_structure["pipeline_ltx_image2video"] = ["LTXImageToVideoPipeline"]

if TYPE_CHECKING:
    from .pipeline_ltx import LTXPipeline
    from .pipeline_ltx_condition import LTXConditionPipeline
    from .pipeline_ltx_image2video import LTXImageToVideoPipeline

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
