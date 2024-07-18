import sys

sys.path.append("../stable_diffusion_v2")

from ldm.modules.encoders.modules import FrozenOpenCLIPEmbedder as FrozenOpenCLIPEmbedder_SDv2

__all__ = ["FrozenOpenCLIPEmbedder"]


class FrozenOpenCLIPEmbedder(FrozenOpenCLIPEmbedder_SDv2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
