from typing import TYPE_CHECKING

from ...utils import _LazyModule

_import_structure = {
    "pipeline_kandinsky3": ["Kandinsky3Pipeline"],
    "pipeline_kandinsky3_img2img": ["Kandinsky3Img2ImgPipeline"],
}


if TYPE_CHECKING:
    from .pipeline_kandinsky3 import Kandinsky3Pipeline
    from .pipeline_kandinsky3_img2img import Kandinsky3Img2ImgPipeline
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
