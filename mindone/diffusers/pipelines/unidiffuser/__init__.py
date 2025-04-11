from typing import TYPE_CHECKING

from ...utils import _LazyModule

_dummy_objects = {}
_import_structure = {}

_import_structure["modeling_text_decoder"] = ["UniDiffuserTextDecoder"]
_import_structure["modeling_uvit"] = ["UniDiffuserModel", "UTransformer2DModel"]
_import_structure["pipeline_unidiffuser"] = ["ImageTextPipelineOutput", "UniDiffuserPipeline"]


if TYPE_CHECKING:
    from .modeling_text_decoder import UniDiffuserTextDecoder
    from .modeling_uvit import UniDiffuserModel, UTransformer2DModel
    from .pipeline_unidiffuser import ImageTextPipelineOutput, UniDiffuserPipeline

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
