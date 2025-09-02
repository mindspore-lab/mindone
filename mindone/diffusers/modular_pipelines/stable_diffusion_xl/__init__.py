from typing import TYPE_CHECKING

from ...utils import _LazyModule

_dummy_objects = {}
_import_structure = {}

_import_structure["encoders"] = ["StableDiffusionXLTextEncoderStep"]
_import_structure["modular_blocks"] = [
    "ALL_BLOCKS",
    "AUTO_BLOCKS",
    "CONTROLNET_BLOCKS",
    "IMAGE2IMAGE_BLOCKS",
    "INPAINT_BLOCKS",
    "IP_ADAPTER_BLOCKS",
    "TEXT2IMAGE_BLOCKS",
    "StableDiffusionXLAutoBlocks",
    "StableDiffusionXLAutoControlnetStep",
    "StableDiffusionXLAutoDecodeStep",
    "StableDiffusionXLAutoIPAdapterStep",
    "StableDiffusionXLAutoVaeEncoderStep",
]
_import_structure["modular_pipeline"] = ["StableDiffusionXLModularPipeline"]

if TYPE_CHECKING:
    from .encoders import StableDiffusionXLTextEncoderStep
    from .modular_blocks import (
        ALL_BLOCKS,
        AUTO_BLOCKS,
        CONTROLNET_BLOCKS,
        IMAGE2IMAGE_BLOCKS,
        INPAINT_BLOCKS,
        IP_ADAPTER_BLOCKS,
        TEXT2IMAGE_BLOCKS,
        StableDiffusionXLAutoBlocks,
        StableDiffusionXLAutoControlnetStep,
        StableDiffusionXLAutoDecodeStep,
        StableDiffusionXLAutoIPAdapterStep,
        StableDiffusionXLAutoVaeEncoderStep,
    )
    from .modular_pipeline import StableDiffusionXLModularPipeline
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
