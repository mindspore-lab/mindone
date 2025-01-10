from typing import TYPE_CHECKING

from ...utils import _LazyModule

_import_structure = {}

_import_structure["pipeline_lumina"] = ["LuminaText2ImgPipeline"]

if TYPE_CHECKING:
    from .pipeline_lumina import LuminaText2ImgPipeline

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
