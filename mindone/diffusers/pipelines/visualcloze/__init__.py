#
# This code is adapted from https://github.com/huggingface/diffusers
# with modifications to run diffusers on mindspore.

from typing import TYPE_CHECKING

from ...utils import _LazyModule, is_transformers_available

_dummy_objects = {}
_import_structure = {}

_import_structure["pipeline_visualcloze_combined"] = ["VisualClozePipeline"]
_import_structure["pipeline_visualcloze_generation"] = ["VisualClozeGenerationPipeline"]


if TYPE_CHECKING:
    from .pipeline_visualcloze_combined import VisualClozePipeline
    from .pipeline_visualcloze_generation import VisualClozeGenerationPipeline


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
