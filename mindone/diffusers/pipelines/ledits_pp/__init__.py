"""Adapted from https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/ledits_pp/__init__.py."""

from typing import TYPE_CHECKING

from ...utils import _LazyModule

_import_structure = {}

_import_structure["pipeline_leditspp_stable_diffusion"] = ["LEditsPPPipelineStableDiffusion"]
_import_structure["pipeline_leditspp_stable_diffusion_xl"] = ["LEditsPPPipelineStableDiffusionXL"]

_import_structure["pipeline_output"] = ["LEditsPPDiffusionPipelineOutput", "LEditsPPDiffusionPipelineOutput"]

if TYPE_CHECKING:
    from .pipeline_leditspp_stable_diffusion import (
        LEditsPPDiffusionPipelineOutput,
        LEditsPPInversionPipelineOutput,
        LEditsPPPipelineStableDiffusion,
    )
    from .pipeline_leditspp_stable_diffusion_xl import LEditsPPPipelineStableDiffusionXL

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
