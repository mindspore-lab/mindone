from typing import TYPE_CHECKING

from ...utils import _LazyModule

_dummy_objects = {}
_import_structure = {}

_import_structure["encoders"] = ["FluxTextEncoderStep"]
_import_structure["modular_blocks"] = [
    "ALL_BLOCKS",
    "AUTO_BLOCKS",
    "TEXT2IMAGE_BLOCKS",
    "FluxAutoBeforeDenoiseStep",
    "FluxAutoBlocks",
    "FluxAutoBlocks",
    "FluxAutoDecodeStep",
    "FluxAutoDenoiseStep",
]
_import_structure["modular_pipeline"] = ["FluxModularPipeline"]

if TYPE_CHECKING:
    from .encoders import FluxTextEncoderStep
    from .modular_blocks import (
        ALL_BLOCKS,
        AUTO_BLOCKS,
        TEXT2IMAGE_BLOCKS,
        FluxAutoBeforeDenoiseStep,
        FluxAutoBlocks,
        FluxAutoDecodeStep,
        FluxAutoDenoiseStep,
    )
    from .modular_pipeline import FluxModularPipeline
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
