from typing import TYPE_CHECKING

from ..utils import _LazyModule

# These modules contain pipelines from multiple libraries/frameworks
_import_structure = {
    "ddim": ["DDIMPipeline"],
    "ddpm": ["DDPMPipeline"],
    "stable_diffusion_xl": [
        "StableDiffusionXLPipeline",
    ],
    "pipeline_utils": [
        "DiffusionPipeline",
        "ImagePipelineOutput",
    ],
}

if TYPE_CHECKING:
    from .ddim import DDIMPipeline
    from .ddpm import DDPMPipeline
    from .pipeline_utils import DiffusionPipeline, ImagePipelineOutput
    from .stable_diffusion_xl import StableDiffusionXLPipeline
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
