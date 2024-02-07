import math
from typing import Optional

from gm.modules.embedders.modules import AbstractEmbModel
from tools._common.clip.clip_modules import LayerNorm

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, Tensor


def feedfoward(dim: int, mult: int = 4):
    inner_dim = int(dim * mult)
    return nn.SequentialCell(
        LayerNorm([dim], epsilon=1e-5),
        nn.Dense(dim, inner_dim, has_bias=False),
        nn.GELU(),
        nn.Dense(inner_dim, dim, has_bias=False),
    )


def reshape_tensor(x: Tensor, heads: int):
    bs, length, _ = x.shape
    x = x.reshape(bs, length, heads, -1)
    x = x.transpose(0, 2, 1, 3)
    return x


class PerceiverAttention(nn.Cell):
    def __init__(self, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads
        self.scale = 1 / math.sqrt(math.sqrt(self.dim_head))

        self.norm1 = LayerNorm([dim], epsilon=1e-5)
        self.norm2 = LayerNorm([dim], epsilon=1e-5)

        self.to_q = nn.Dense(dim, inner_dim, has_bias=False)
        self.to_kv = nn.Dense(dim, inner_dim * 2, has_bias=False)
        self.to_out = nn.Dense(inner_dim, dim, has_bias=False)

    def construct(self, x, latents):
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = ops.concat([x, latents], axis=-2)
        k, v = self.to_kv(kv_input).chunk(2, axis=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        weight = (q * self.scale) @ (k * self.scale).transpose(0, 1, 3, 2)
        weight = ops.softmax(weight.to(ms.float32), axis=-1).to(weight.dtype)
        out = weight @ v

        out = out.transpose(0, 2, 1, 3).reshape(b, l, -1)
        return self.to_out(out)


class ImageProjModel(nn.Cell):
    """Projection Model / Resampler"""

    def __init__(
        self,
        dim: int = 1024,
        depth: int = 8,
        dim_head: int = 64,
        heads: int = 16,
        num_queries: int = 8,
        embeddings_dim: int = 768,
        output_dim: int = 1024,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()

        self.latents = Parameter(ops.randn(1, num_queries, dim) / dim**0.5)
        self.proj_in = nn.Dense(embeddings_dim, dim)
        self.proj_out = nn.Dense(dim, output_dim)
        self.norm_out = LayerNorm([output_dim], epsilon=1e-5)

        self.layers = nn.CellList([])
        for _ in range(depth):
            self.layers.append(
                nn.CellList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        feedfoward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def construct(self, image_embeds: Tensor) -> Tensor:
        latents = ops.tile(self.latents, (image_embeds.shape[0], 1, 1))

        x = self.proj_in(image_embeds)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        latents = self.proj_out(latents)
        latents = self.norm_out(latents)
        return latents


class InstantIDImageEmbedder(AbstractEmbModel):
    def __init__(
        self,
        freeze: bool = False,
        dim: int = 1280,
        depth: int = 4,
        dim_head: int = 64,
        heads: int = 20,
        num_tokens: int = 16,
        image_emb_dim: int = 512,
        unet_cross_attention_dim: int = 2048,
        ff_mult: int = 4,
        embedding_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding_dropout = embedding_dropout
        self.image_emb_dim = image_emb_dim

        # TODO; repalce identiy layer with face model
        self.image_encoder = nn.Identity()

        self.image_proj = ImageProjModel(
            dim=dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            num_queries=num_tokens,
            embeddings_dim=image_emb_dim,
            output_dim=unet_cross_attention_dim,
            ff_mult=ff_mult,
        )

        self._freeze = freeze
        if self._freeze:
            self.freeze()

    def freeze(self):
        # only freeze the image encoder
        self.image_encoder.set_train(False)
        self.image_encoder.set_grad(False)
        for _, p in self.image_encoder.parameters_and_names():
            p.requires_grad = False

    def get_image_embeds(self, image_tensor: Optional[Tensor] = None) -> Tensor:
        if image_tensor is None:
            clip_image_embeds = ops.zeros((1, 1, self.image_emb_dim), dtype=ms.float32)
        else:
            clip_image_embeds = self.image_encoder(image_tensor)
            mask = ops.rand((clip_image_embeds.shape[0], 1)) > self.embedding_dropout
            clip_image_embeds = mask * clip_image_embeds

        if self._freeze:
            clip_image_embeds = ops.stop_gradient(clip_image_embeds)

        image_prompt_embeds = self.image_proj(clip_image_embeds)
        return image_prompt_embeds

    @ms.jit
    def construct(self, image_tensor: Optional[Tensor] = None) -> Tensor:
        image_prompt_embeds = self.get_image_embeds(image_tensor)
        return image_prompt_embeds
