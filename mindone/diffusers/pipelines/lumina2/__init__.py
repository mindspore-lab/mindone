from typing import TYPE_CHECKING

from ...utils import _LazyModule

_import_structure = {}


_import_structure["pipeline_lumina2"] = ["Lumina2Pipeline", "Lumina2Text2ImgPipeline"]

if TYPE_CHECKING:
    from .pipeline_lumina2 import Lumina2Pipeline, Lumina2Text2ImgPipeline

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
