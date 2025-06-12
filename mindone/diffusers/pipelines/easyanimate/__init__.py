from typing import TYPE_CHECKING

from ...utils import _LazyModule

_import_structure = {}


_import_structure["pipeline_easyanimate"] = ["EasyAnimatePipeline"]
_import_structure["pipeline_easyanimate_control"] = ["EasyAnimateControlPipeline"]
_import_structure["pipeline_easyanimate_inpaint"] = ["EasyAnimateInpaintPipeline"]

if TYPE_CHECKING:
    from .pipeline_easyanimate import EasyAnimatePipeline
    from .pipeline_easyanimate_control import EasyAnimateControlPipeline
    from .pipeline_easyanimate_inpaint import EasyAnimateInpaintPipeline

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
