import sys

from mindspore import Tensor

sys.path.append("../stable_diffusion_xl")

from gm.modules.embedders.modules import FrozenOpenCLIPEmbedder2 as FrozenOpenCLIPEmbedder2_SDXL
from gm.modules.embedders.open_clip import create_model as openclip_create_model

__all__ = ["FrozenOpenCLIPEmbedder"]


class FrozenOpenCLIPEmbedder(FrozenOpenCLIPEmbedder2_SDXL):
    """
    Uses the OpenCLIP transformer encoder for text
    """

    def __init__(self, arch="ViT-H-14", pretrained="laion2b_s32b_b79k", layer="penultimate", *args, **kwargs):
        super().__init__()
        self.model = openclip_create_model(arch, pretrained=pretrained)
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()
        del self.model.visual

    def construct(self, text):
        tokens, _ = self.tokenize(text)
        z = self.encode_with_transformer(Tensor(tokens))
        return z
