"""
Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/
pipelines/chroma/__init__.py.
"""

from typing import TYPE_CHECKING

from ...utils import _LazyModule

_dummy_objects = {}
_additional_imports = {}
_import_structure = {"pipeline_output": ["ChromaPipelineOutput"]}


_import_structure["pipeline_chroma"] = ["ChromaPipeline"]
_import_structure["pipeline_chroma_img2img"] = ["ChromaImg2ImgPipeline"]

if TYPE_CHECKING:
    from .pipeline_chroma import ChromaPipeline
    from .pipeline_chroma_img2img import ChromaImg2ImgPipeline
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
    for name, value in _additional_imports.items():
        setattr(sys.modules[__name__], name, value)
