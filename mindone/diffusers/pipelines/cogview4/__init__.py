from typing import TYPE_CHECKING

from ...utils import _LazyModule


_additional_imports = {}
_import_structure = {"pipeline_output": ["CogView4PlusPipelineOutput"]}

_import_structure["pipeline_cogview4"] = ["CogView4Pipeline"]
if TYPE_CHECKING:
    from .pipeline_cogview4 import CogView4Pipeline
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )

    for name, value in _additional_imports.items():
        setattr(sys.modules[__name__], name, value)
