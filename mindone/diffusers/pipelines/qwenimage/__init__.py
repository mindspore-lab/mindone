"""Adapted from https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/qwenimage/__init__.py."""

from typing import TYPE_CHECKING

from ...utils import _LazyModule

_import_structure = {
    "modeling_qwenimage": ["ReduxImageEncoder"],
    "pipeline_qwenimage": ["QwenImagePipeline"],
    "pipeline_qwenimage_controlnet": ["QwenImageControlNetPipeline"],
    "pipeline_qwenimage_controlnet_inpaint": ["QwenImageControlNetInpaintPipeline"],
    "pipeline_qwenimage_edit": ["QwenImageEditPipeline"],
    "pipeline_qwenimage_edit_inpaint": ["QwenImageEditInpaintPipeline"],
    "pipeline_qwenimage_edit_plus": ["QwenImageEditPlusPipeline"],
    "pipeline_qwenimage_img2img": ["QwenImageImg2ImgPipeline"],
    "pipeline_qwenimage_inpaint": ["QwenImageInpaintPipeline"],
}

if TYPE_CHECKING:
    from .pipeline_qwenimage import QwenImagePipeline
    from .pipeline_qwenimage_controlnet import QwenImageControlNetPipeline
    from .pipeline_qwenimage_controlnet_inpaint import QwenImageControlNetInpaintPipeline
    from .pipeline_qwenimage_edit import QwenImageEditPipeline
    from .pipeline_qwenimage_edit_inpaint import QwenImageEditInpaintPipeline
    from .pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
    from .pipeline_qwenimage_img2img import QwenImageImg2ImgPipeline
    from .pipeline_qwenimage_inpaint import QwenImageInpaintPipeline
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
