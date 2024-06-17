from typing import TYPE_CHECKING

from ...utils import _LazyModule

_import_structure = {"pipeline_output": ["StableDiffusion3PipelineOutput"]}
_import_structure["pipeline_stable_diffusion_3"] = ["StableDiffusion3Pipeline"]

if TYPE_CHECKING:
    from .pipeline_stable_diffusion_3 import StableDiffusion3Pipeline

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
