from typing import TYPE_CHECKING

from ...utils import _LazyModule

_dummy_objects = {}
_import_structure = {}

_import_structure["encoders"] = ["QwenImageTextEncoderStep"]
_import_structure["modular_blocks"] = [
    "ALL_BLOCKS",
    "AUTO_BLOCKS",
    "CONTROLNET_BLOCKS",
    "EDIT_AUTO_BLOCKS",
    "EDIT_BLOCKS",
    "EDIT_INPAINT_BLOCKS",
    "EDIT_PLUS_AUTO_BLOCKS",
    "EDIT_PLUS_BLOCKS",
    "IMAGE2IMAGE_BLOCKS",
    "INPAINT_BLOCKS",
    "TEXT2IMAGE_BLOCKS",
    "QwenImageAutoBlocks",
    "QwenImageEditAutoBlocks",
    "QwenImageEditPlusAutoBlocks",
]
_import_structure["modular_pipeline"] = [
    "QwenImageEditModularPipeline",
    "QwenImageEditPlusModularPipeline",
    "QwenImageModularPipeline",
]

if TYPE_CHECKING:
    from .encoders import QwenImageTextEncoderStep
    from .modular_blocks import (
        ALL_BLOCKS,
        AUTO_BLOCKS,
        CONTROLNET_BLOCKS,
        EDIT_AUTO_BLOCKS,
        EDIT_BLOCKS,
        EDIT_INPAINT_BLOCKS,
        EDIT_PLUS_AUTO_BLOCKS,
        EDIT_PLUS_BLOCKS,
        IMAGE2IMAGE_BLOCKS,
        INPAINT_BLOCKS,
        TEXT2IMAGE_BLOCKS,
        QwenImageAutoBlocks,
        QwenImageEditAutoBlocks,
        QwenImageEditPlusAutoBlocks,
    )
    from .modular_pipeline import (
        QwenImageEditModularPipeline,
        QwenImageEditPlusModularPipeline,
        QwenImageModularPipeline,
    )
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
