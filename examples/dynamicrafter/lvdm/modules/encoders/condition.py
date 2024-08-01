import sys

sys.path.append("../stable_diffusion_xl")

from gm.modules.embedders.modules import FrozenOpenCLIPEmbedder2 as FrozenOpenCLIPEmbedder2_SDXL
from gm.modules.embedders.modules import FrozenOpenCLIPImageEmbedder as FrozenOpenCLIPImageEmbedder_SDXL

import mindspore.ops as ops

__all__ = ["FrozenOpenCLIPEmbedder", "FrozenOpenCLIPImageEmbedderV2"]


class FrozenOpenCLIPEmbedder(FrozenOpenCLIPEmbedder2_SDXL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.model.visual


class FrozenOpenCLIPImageEmbedderV2(FrozenOpenCLIPImageEmbedder_SDXL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def construct(self, image, no_dropout=False):
        # image: b c h w
        z = self.encode_with_vision_transformer(image)
        return z

    def encode_with_vision_transformer(self, x):
        x = self.preprocess(x)

        # to patches - whether to use dual patchnorm - https://arxiv.org/abs/2302.01327v1
        if self.model.visual.input_patchnorm:
            # einops - rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)')
            x = x.reshape(
                x.shape[0],
                x.shape[1],
                self.model.visual.grid_size[0],
                self.model.visual.patch_size[0],
                self.model.visual.grid_size[1],
                self.model.visual.patch_size[1],
            )
            x = x.permute(0, 2, 4, 1, 3, 5)
            x = x.reshape(x.shape[0], self.model.visual.grid_size[0] * self.model.visual.grid_size[1], -1)
            x = self.model.visual.patchnorm_pre_ln(x)
            x = self.model.visual.conv1(x)
        else:
            x = self.model.visual.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        # class embeddings and positional embeddings
        x = ops.cat(
            [self.model.visual.class_embedding.to(x.dtype) + ops.zeros((x.shape[0], 1, x.shape[-1]), dtype=x.dtype), x],
            axis=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.model.visual.positional_embedding.to(x.dtype)

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        # x = self.model.visual.patch_dropout(x)
        x = self.model.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        return x
