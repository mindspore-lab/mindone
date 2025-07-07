from typing import TYPE_CHECKING

from ...utils import _LazyModule, is_transformers_available

_dummy_objects = {}
_import_structure = {}

_import_structure["image_encoder"] = ["PaintByExampleImageEncoder"]
_import_structure["pipeline_paint_by_example"] = ["PaintByExamplePipeline"]


if TYPE_CHECKING:
    from .image_encoder import PaintByExampleImageEncoder
    from .pipeline_paint_by_example import PaintByExamplePipeline

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
