from typing import TYPE_CHECKING

from ...utils import _LazyModule

_dummy_objects = {}
_additional_imports = {}
_import_structure = {"pipeline_output": ["Flux2PipelineOutput"]}
_import_structure["pipeline_flux2"] = ["Flux2Pipeline"]

if TYPE_CHECKING:
    from .pipeline_flux2 import Flux2Pipeline
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
    for name, value in _additional_imports.items():
        setattr(sys.modules[__name__], name, value)
