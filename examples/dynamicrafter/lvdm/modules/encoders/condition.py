import sys

sys.path.append("../stable_diffusion_xl")

from gm.modules.embedders.modules import FrozenOpenCLIPImageEmbedder

__all__ = ["FrozenOpenCLIPImageEmbedder"]
