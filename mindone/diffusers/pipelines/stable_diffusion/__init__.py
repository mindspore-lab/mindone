from typing import TYPE_CHECKING

from ...utils import _LazyModule

_import_structure = {
    "pipeline_output": ["StableDiffusionPipelineOutput"],
    "pipeline_stable_diffusion": ["StableDiffusionPipeline"],
    "safety_checker": ["StableDiffusionSafetyChecker"],
}

if TYPE_CHECKING:
    from .pipeline_stable_diffusion import (
        StableDiffusionPipeline,
        StableDiffusionPipelineOutput,
        StableDiffusionSafetyChecker,
    )

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
