from typing import TYPE_CHECKING

from ...utils import _LazyModule

_import_structure = {}
_import_structure["pipeline_stable_diffusion_3_controlnet"] = ["StableDiffusion3ControlNetPipeline"]

if TYPE_CHECKING:
    from .pipeline_stable_diffusion_3_controlnet import StableDiffusion3ControlNetPipeline
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
