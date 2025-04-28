from typing import TYPE_CHECKING

from ...utils import _LazyModule

_dummy_objects = {}
_import_structure = {}

_import_structure["pipeline_audioldm2"] = ["AudioLDM2Pipeline"]
_import_structure["modeling_audioldm2"] = ["AudioLDM2ProjectionModel", "AudioLDM2UNet2DConditionModel"]

if TYPE_CHECKING:
    from .modeling_audioldm2 import AudioLDM2ProjectionModel, AudioLDM2UNet2DConditionModel
    from .pipeline_audioldm2 import AudioLDM2Pipeline

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
