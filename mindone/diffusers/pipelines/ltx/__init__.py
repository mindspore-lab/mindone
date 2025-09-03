"""Adapted from https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/ltx/__init__.py."""

from typing import TYPE_CHECKING

from ...utils import _LazyModule

_dummy_objects = {}
_import_structure = {}

_import_structure["modeling_latent_upsampler"] = ["LTXLatentUpsamplerModel"]
_import_structure["pipeline_ltx"] = ["LTXPipeline"]
_import_structure["pipeline_ltx_condition"] = ["LTXConditionPipeline"]
_import_structure["pipeline_ltx_image2video"] = ["LTXImageToVideoPipeline"]
_import_structure["pipeline_ltx_latent_upsample"] = ["LTXLatentUpsamplePipeline"]

if TYPE_CHECKING:
    from .modeling_latent_upsampler import LTXLatentUpsamplerModel
    from .pipeline_ltx import LTXPipeline
    from .pipeline_ltx_condition import LTXConditionPipeline
    from .pipeline_ltx_image2video import LTXImageToVideoPipeline
    from .pipeline_ltx_latent_upsample import LTXLatentUpsamplePipeline

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
