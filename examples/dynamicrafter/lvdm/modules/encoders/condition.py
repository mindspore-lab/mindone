import sys

sys.path.append("../stable_diffusion_xl")

from gm.modules.embedders.modules import FrozenOpenCLIPEmbedder2, FrozenOpenCLIPImageEmbedder 

__all__ = ["FrozenOpenCLIPEmbedder2", "FrozenOpenCLIPImageEmbedder"]
