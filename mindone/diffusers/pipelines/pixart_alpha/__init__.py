"""Adapted from https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/pixart_alpha/__init__.py."""

from typing import TYPE_CHECKING

from ...utils import _LazyModule

_import_structure = {}
_import_structure["pipeline_pixart_alpha"] = ["PixArtAlphaPipeline"]
_import_structure["pipeline_pixart_sigma"] = ["PixArtSigmaPipeline"]

if TYPE_CHECKING:
    from .pipeline_pixart_alpha import (
        ASPECT_RATIO_256_BIN,
        ASPECT_RATIO_512_BIN,
        ASPECT_RATIO_1024_BIN,
        PixArtAlphaPipeline,
    )
    from .pipeline_pixart_sigma import ASPECT_RATIO_2048_BIN, PixArtSigmaPipeline

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
