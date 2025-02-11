from typing import TYPE_CHECKING

from ...utils import _LazyModule

_import_structure = {
    "pipeline_kandinsky2_2": ["KandinskyV22Pipeline"],
    "pipeline_kandinsky2_2_combined": [
        "KandinskyV22CombinedPipeline",
        "KandinskyV22Img2ImgCombinedPipeline",
        "KandinskyV22InpaintCombinedPipeline",
    ],
    "pipeline_kandinsky2_2_controlnet": ["KandinskyV22ControlnetPipeline"],
    "pipeline_kandinsky2_2_controlnet_img2img": ["KandinskyV22ControlnetImg2ImgPipeline"],
    "pipeline_kandinsky2_2_img2img": ["KandinskyV22Img2ImgPipeline"],
    "pipeline_kandinsky2_2_inpainting": ["KandinskyV22InpaintPipeline"],
    "pipeline_kandinsky2_2_prior": ["KandinskyV22PriorPipeline"],
    "pipeline_kandinsky2_2_prior_emb2emb": ["KandinskyV22PriorEmb2EmbPipeline"],
}


if TYPE_CHECKING:
    from .pipeline_kandinsky2_2 import KandinskyV22Pipeline
    from .pipeline_kandinsky2_2_combined import (
        KandinskyV22CombinedPipeline,
        KandinskyV22Img2ImgCombinedPipeline,
        KandinskyV22InpaintCombinedPipeline,
    )
    from .pipeline_kandinsky2_2_controlnet import KandinskyV22ControlnetPipeline
    from .pipeline_kandinsky2_2_controlnet_img2img import KandinskyV22ControlnetImg2ImgPipeline
    from .pipeline_kandinsky2_2_img2img import KandinskyV22Img2ImgPipeline
    from .pipeline_kandinsky2_2_inpainting import KandinskyV22InpaintPipeline
    from .pipeline_kandinsky2_2_prior import KandinskyV22PriorPipeline
    from .pipeline_kandinsky2_2_prior_emb2emb import KandinskyV22PriorEmb2EmbPipeline

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
