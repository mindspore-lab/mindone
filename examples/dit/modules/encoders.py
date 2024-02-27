import sys

sys.path.append("../stable_diffusion_v2")
from ldm.modules.encoders.modules import FrozenCLIPEmbedder

__all__ = ["FrozenCLIPEmbedder"]
