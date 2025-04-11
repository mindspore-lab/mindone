from typing import TYPE_CHECKING

from ...utils import _LazyModule

_import_structure = {}


_import_structure["pipeline_amused"] = ["AmusedPipeline"]
_import_structure["pipeline_amused_img2img"] = ["AmusedImg2ImgPipeline"]
_import_structure["pipeline_amused_inpaint"] = ["AmusedInpaintPipeline"]


if TYPE_CHECKING:
    from .pipeline_amused import AmusedPipeline
    from .pipeline_amused_img2img import AmusedImg2ImgPipeline
    from .pipeline_amused_inpaint import AmusedInpaintPipeline

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
