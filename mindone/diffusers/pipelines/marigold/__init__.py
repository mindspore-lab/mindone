"""Adapted from https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/marigold/__init__.py."""

from typing import TYPE_CHECKING

from ...utils import _LazyModule

_import_structure = {}
_import_structure["marigold_image_processing"] = ["MarigoldImageProcessor"]
_import_structure["pipeline_marigold_depth"] = ["MarigoldDepthOutput", "MarigoldDepthPipeline"]
_import_structure["pipeline_marigold_intrinsics"] = ["MarigoldIntrinsicsOutput", "MarigoldIntrinsicsPipeline"]
_import_structure["pipeline_marigold_normals"] = ["MarigoldNormalsOutput", "MarigoldNormalsPipeline"]

if TYPE_CHECKING:
    from .marigold_image_processing import MarigoldImageProcessor
    from .pipeline_marigold_depth import MarigoldDepthOutput, MarigoldDepthPipeline
    from .pipeline_marigold_intrinsics import MarigoldIntrinsicsOutput, MarigoldIntrinsicsPipeline
    from .pipeline_marigold_normals import MarigoldNormalsOutput, MarigoldNormalsPipeline

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
