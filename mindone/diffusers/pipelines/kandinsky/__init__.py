from typing import TYPE_CHECKING

from ...utils import _LazyModule

_import_structure = {
    "pipeline_kandinsky": ["KandinskyPipeline"],
    "pipeline_kandinsky_combined": [
        "KandinskyCombinedPipeline",
        "KandinskyImg2ImgCombinedPipeline",
        "KandinskyInpaintCombinedPipeline",
    ],
    "pipeline_kandinsky_img2img": ["KandinskyImg2ImgPipeline"],
    "pipeline_kandinsky_inpaint": ["KandinskyInpaintPipeline"],
    "pipeline_kandinsky_prior": ["KandinskyPriorPipeline", "KandinskyPriorPipelineOutput"],
    "text_encoder": ["MultilingualCLIP"],
}


if TYPE_CHECKING:
    from .pipeline_kandinsky import KandinskyPipeline
    from .pipeline_kandinsky_combined import (
        KandinskyCombinedPipeline,
        KandinskyImg2ImgCombinedPipeline,
        KandinskyInpaintCombinedPipeline,
    )
    from .pipeline_kandinsky_img2img import KandinskyImg2ImgPipeline
    from .pipeline_kandinsky_inpaint import KandinskyInpaintPipeline
    from .pipeline_kandinsky_prior import KandinskyPriorPipeline, KandinskyPriorPipelineOutput
    from .text_encoder import MultilingualCLIP

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
