"""Adapted from https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/stable_audio/__init__.py."""

from typing import TYPE_CHECKING

from ...utils import _LazyModule

_dummy_objects = {}
_import_structure = {}
_import_structure["modeling_stable_audio"] = ["StableAudioProjectionModel"]
_import_structure["pipeline_stable_audio"] = ["StableAudioPipeline"]


if TYPE_CHECKING:
    from .modeling_stable_audio import StableAudioProjectionModel
    from .pipeline_stable_audio import StableAudioPipeline

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
