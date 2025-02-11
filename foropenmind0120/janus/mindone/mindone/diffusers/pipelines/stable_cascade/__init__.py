from typing import TYPE_CHECKING

from ...utils import _LazyModule

_import_structure = {
    "pipeline_stable_cascade": ["StableCascadeDecoderPipeline"],
    "pipeline_stable_cascade_combined": ["StableCascadeCombinedPipeline"],
    "pipeline_stable_cascade_prior": ["StableCascadePriorPipeline"],
}


if TYPE_CHECKING:
    from .pipeline_stable_cascade import StableCascadeDecoderPipeline
    from .pipeline_stable_cascade_combined import StableCascadeCombinedPipeline
    from .pipeline_stable_cascade_prior import StableCascadePriorPipeline
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
