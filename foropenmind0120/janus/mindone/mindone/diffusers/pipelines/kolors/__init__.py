from typing import TYPE_CHECKING

from ...utils import _LazyModule

_import_structure = {}

_import_structure["pipeline_kolors"] = ["KolorsPipeline"]
_import_structure["pipeline_kolors_img2img"] = ["KolorsImg2ImgPipeline"]
_import_structure["text_encoder"] = ["ChatGLMModel"]
_import_structure["tokenizer"] = ["ChatGLMTokenizer"]

if TYPE_CHECKING:
    from .pipeline_kolors import KolorsPipeline
    from .pipeline_kolors_img2img import KolorsImg2ImgPipeline
    from .text_encoder import ChatGLMModel
    from .tokenizer import ChatGLMTokenizer

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
