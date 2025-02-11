from typing import TYPE_CHECKING

from ...utils import _LazyModule

_import_structure = {
    "modeling_paella_vq_model": ["PaellaVQModel"],
    "modeling_wuerstchen_diffnext": ["WuerstchenDiffNeXt"],
    "modeling_wuerstchen_prior": ["WuerstchenPrior"],
    "pipeline_wuerstchen": ["WuerstchenDecoderPipeline"],
    "pipeline_wuerstchen_combined": ["WuerstchenCombinedPipeline"],
    "pipeline_wuerstchen_prior": ["DEFAULT_STAGE_C_TIMESTEPS", "WuerstchenPriorPipeline"],
}


if TYPE_CHECKING:
    from .modeling_paella_vq_model import PaellaVQModel
    from .modeling_wuerstchen_diffnext import WuerstchenDiffNeXt
    from .modeling_wuerstchen_prior import WuerstchenPrior
    from .pipeline_wuerstchen import WuerstchenDecoderPipeline
    from .pipeline_wuerstchen_combined import WuerstchenCombinedPipeline
    from .pipeline_wuerstchen_prior import DEFAULT_STAGE_C_TIMESTEPS, WuerstchenPriorPipeline
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
