from typing import Optional

import numpy as np
from gm.modules.embedders.modules import AbstractEmbModel
from ldm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder
from tools._common.clip.clip_modules import LayerNorm

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor


class ImageProjModel(nn.Cell):
    """Projection Model"""

    def __init__(
        self,
        cross_attention_dim: int = 2048,
        clip_embeddings_dim: int = 1280,
        clip_extra_context_tokens: int = 4,
        use_fp16: bool = False,
    ) -> None:
        super().__init__()
        dtype = ms.float16 if use_fp16 else ms.float32

        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = nn.Dense(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim).to_float(dtype)
        self.norm = LayerNorm([cross_attention_dim], epsilon=1e-5)

    def construct(self, image_embeds: Tensor) -> Tensor:
        clip_extra_context_tokens = self.proj(image_embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class IPAdapterImageEmbedder(AbstractEmbModel):
    def __init__(
        self,
        embed_dim: int = 1280,
        image_resolution: int = 224,
        vision_layers: int = 48,
        vision_width: int = 1664,
        vision_patch_size: int = 14,
        vision_head_width: int = 104,
        unet_cross_attention_dim: int = 2048,
        num_tokens: int = 4,
        mlp_ratio: float = 4.9231,
        use_fp16: bool = False,
    ) -> None:
        super().__init__()
        self.dtype = ms.float16 if use_fp16 else ms.float32
        self.embed_dim = embed_dim

        # load image encoder
        self.image_encoder = FrozenOpenCLIPImageEmbedder(
            use_fp16=use_fp16,
            embed_dim=embed_dim,
            image_resolution=image_resolution,
            vision_layers=vision_layers,
            vision_width=vision_width,
            vision_patch_size=vision_patch_size,
            vision_head_width=vision_head_width,
            mlp_ratio=mlp_ratio,
        )
        self.image_proj = ImageProjModel(
            cross_attention_dim=unet_cross_attention_dim,
            clip_embeddings_dim=embed_dim,
            clip_extra_context_tokens=num_tokens,
            use_fp16=use_fp16,
        )

    def freeze(self):
        self.image_encoder.set_train(False)
        self.image_encoder.set_grad(False)
        self.image_proj.set_train(False)
        self.image_proj.set_grad(False)
        for _, p in self.parameters_and_names():
            p.requires_grad = False

    def tokenize(self, img: Optional[np.ndarray] = None):
        return img, None

    def get_image_embeds(self, image_tensor: Optional[Tensor] = None) -> Tensor:
        if image_tensor is None:
            clip_image_embeds = ops.zeros((1, self.embed_dim), dtype=self.dtype)
        else:
            clip_image_embeds = self.image_encoder.encode(image_tensor)
        image_prompt_embeds = self.image_proj(clip_image_embeds)
        return image_prompt_embeds

    @ms.jit
    def construct(self, image_tensor: Tensor) -> Tensor:
        image_prompt_embeds = self.get_image_embeds(image_tensor)
        return image_prompt_embeds
