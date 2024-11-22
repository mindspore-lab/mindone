from typing import TYPE_CHECKING

from ...utils import _LazyModule

_import_structure = {
    "pipeline_output": ["FluxPipelineOutput"],
    "pipeline_flux": ["FluxPipeline"],
}

if TYPE_CHECKING:
    from .pipeline_flux import FluxPipeline
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
