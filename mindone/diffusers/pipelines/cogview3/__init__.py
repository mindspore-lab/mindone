"""Adapted from https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/cogview3/__init__.py."""

from typing import TYPE_CHECKING

from ...utils import _LazyModule

_import_structure = {
    "pipeline_output": ["CogView3PlusPipelineOutput"],
    "pipeline_cogview3plus": ["CogView3PlusPipeline"],
}


if TYPE_CHECKING:
    from .pipeline_cogview3plus import CogView3PlusPipeline
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
