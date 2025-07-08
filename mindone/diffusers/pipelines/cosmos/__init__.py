from typing import TYPE_CHECKING

from ...utils import _LazyModule

_import_structure = {
    "pipeline_cosmos_text2world": ["CosmosTextToWorldPipeline"],
    "pipeline_cosmos_video2world": ["CosmosVideoToWorldPipeline"],
    "pipeline_cosmos2_text2image": ["Cosmos2TextToImagePipeline"],
    "pipeline_cosmos2_text2world": ["Cosmos2VideoToWorldPipeline"],
}

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:

    from .pipeline_cosmos_text2world import CosmosTextToWorldPipeline
    from .pipeline_cosmos_video2world import CosmosVideoToWorldPipeline
    from .pipeline_cosmos2_text2image import Cosmos2TextToImagePipeline
    from .pipeline_cosmos2_video2world import Cosmos2VideoToWorldPipeline

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
