import sys

sys.path.append("../stable_diffusion_xl")

from gm.modules.embedders.modules import FrozenOpenCLIPEmbedder2 as FrozenOpenCLIPEmbedder2_SDXL
from gm.modules.embedders.modules import FrozenOpenCLIPImageEmbedder as FrozenOpenCLIPImageEmbedderV2

# __all__ = ["FrozenOpenCLIPEmbedder2", "FrozenOpenCLIPImageEmbedder"]


class FrozenOpenCLIPEmbedder(FrozenOpenCLIPEmbedder2_SDXL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.model.visual


# class FrozenOpenCLIPImageEmbedderV2(FrozenOpenCLIPImageEmbedder_SDXL):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         del self.model.transformer