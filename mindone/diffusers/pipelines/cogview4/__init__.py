"""Adapted from https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/cogview4/__init__.py."""

from typing import TYPE_CHECKING

from ...utils import _LazyModule

_import_structure = {
    "pipeline_output": ["CogView4PlusPipelineOutput"],
    "pipeline_cogview4": ["CogView4Pipeline"],
    "pipeline_cogview4_control": ["CogView4ControlPipeline"],
}


if TYPE_CHECKING:
    from .pipeline_cogview4 import CogView4Pipeline
    from .pipeline_cogview4_control import CogView4ControlPipeline
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
