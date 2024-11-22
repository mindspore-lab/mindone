import sys

sys.path.append("../stable_diffusion_xl")

from gm.modules.embedders.modules import FrozenOpenCLIPEmbedder2 as FrozenOpenCLIPEmbedder2_SDXL

__all__ = ["FrozenOpenCLIPEmbedder"]


class FrozenOpenCLIPEmbedder(FrozenOpenCLIPEmbedder2_SDXL):
    """
    Uses the OpenCLIP transformer encoder for text
    """

    def __init__(self, arch="ViT-H-14", layer="penultimate", *args, **kwargs):
        super().__init__()

        del self.model.visual

    def construct(self, text):
        tokens, _ = self.tokenize(text)
        z = self.encode_with_transformer(tokens)
        return z
