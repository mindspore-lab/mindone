"""Adapted from https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/latent_diffusion/__init__.py."""

from typing import TYPE_CHECKING

from ...utils import _LazyModule

_dummy_objects = {}
_import_structure = {}

_import_structure["pipeline_latent_diffusion"] = ["LDMBertModel", "LDMTextToImagePipeline"]
_import_structure["pipeline_latent_diffusion_superresolution"] = ["LDMSuperResolutionPipeline"]


if TYPE_CHECKING:
    from .pipeline_latent_diffusion import LDMBertModel, LDMTextToImagePipeline
    from .pipeline_latent_diffusion_superresolution import LDMSuperResolutionPipeline

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
