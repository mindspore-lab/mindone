# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

# ******************************************************************************
#   Code modified by Zexin He in 2023-2024.
#   Modifications are marked with clearly visible comments
#   licensed under the Apache License, Version 2.0.
# ******************************************************************************

import logging
import math
from functools import partial
from typing import Callable, Sequence, Tuple, Union

import mindspore as ms
from mindspore import mint, nn, ops
from mindspore.common.initializer import Normal, TruncatedNormal, initializer

# ********** Modified by Zexin He in 2023-2024 **********
# Avoid using nested tensor for now, deprecating usage of NestedTensorBlock
from ..layers import Block, BlockWithModulation, MemEffAttention, Mlp, PatchEmbed, SwiGLUFFNFused

# ********************************************************


logger = logging.getLogger("dinov2")


def named_apply(fn: Callable, module: nn.Cell, name="", depth_first=True, include_root=False) -> nn.Cell:
    if not depth_first and include_root:
        fn(module=module, name=name)

    for cell_name, cell in module.name_cells().items():
        cell_name = ".".join((name, cell_name)) if name else cell_name
        named_apply(fn=fn, module=cell, name=cell_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.CellList):
    def construct(self, x):
        for b in self:
            x = b(x)
        return x


class DinoVisionTransformer(nn.Cell):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        # ********** Modified by Zexin He in 2023-2024 **********
        modulation_dim: int = None,
        # ********************************************************
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Cell): patch embedding layer
            act_layer (nn.Cell): MLP activation layer
            block_fn (nn.Cell): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating positional embeddings
            interpolate_offset: (float) work-around offset to apply when interpolating positional embeddings
        """
        super().__init__()

        # ********** Modified by Zexin He in 2023-2024 **********
        block_norm_layer = None
        if modulation_dim is not None:
            from ....modulate import ModLN

            block_norm_layer = partial(ModLN, mod_dim=modulation_dim)
        else:
            block_norm_layer = nn.LayerNorm
        block_norm_layer = partial(block_norm_layer, epsilon=1e-6)
        # ********************************************************
        norm_layer = partial(nn.LayerNorm, epsilon=1e-6)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = ms.Parameter(mint.zeros((1, 1, embed_dim), dtype=ms.float32))
        self.pos_embed = ms.Parameter(mint.zeros((1, num_patches + self.num_tokens, embed_dim), dtype=ms.float32))
        assert num_register_tokens >= 0
        self.register_tokens = (
            ms.Parameter(mint.zeros((1, num_register_tokens, embed_dim), dtype=ms.float32))
            if num_register_tokens
            else None
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in ops.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                # ********** Modified by Zexin He in 2023-2024 **********
                norm_layer=block_norm_layer,
                # ********************************************************
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            self.blocks = nn.CellList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.CellList(blocks_list)

        self.norm = norm_layer((embed_dim,))
        self.head = nn.Identity()

        # ********** Modified by Zexin He in 2023-2024 **********
        # hacking unused mask_token for better DDP
        # self.mask_token = ms.Parameter(mint.zeros((1, embed_dim)))
        # ********************************************************

        self.init_weights()

    def init_weights(self):
        weight = initializer(TruncatedNormal(sigma=0.02, mean=0.0, a=-2.0, b=2.0), self.pos_embed.shape)
        self.pos_embed.set_data(weight)

        weight = initializer(Normal(sigma=1e-6, mean=0.0), self.cls_token.shape)
        self.cls_token.set_data(weight)

        if self.register_tokens is not None:
            weight = initializer(Normal(sigma=1e-6, mean=0.0), self.register_tokens.shape)
            self.register_tokens.set_data(weight)

        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + self.interpolate_offset, h0 + self.interpolate_offset

        sqrt_N = math.sqrt(N)
        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
        patch_pos_embed = ops.interpolate(
            patch_pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(0, 3, 1, 2),
            scale_factor=(sx, sy),  # ms does not support bicubic by directly passing this parameter yet,
            # mode="bicubic", # not support bf16
            # antialias=self.interpolate_antialias, # ms not supported
            recompute_scale_factor=True,  # to compute scale factor, need to set this True
        )

        assert int(w0) == patch_pos_embed.shape[-2]
        assert int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute((0, 2, 3, 1)).view((1, -1, dim))
        return mint.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            # ********** Modified by Zexin He in 2023-2024 **********
            raise NotImplementedError("Masking is not supported in hacked DINOv2")
            # x = ops.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
            # ********************************************************

        x = mint.cat((self.cls_token.broadcast_to((x.shape[0], -1, -1)), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)

        if self.register_tokens is not None:
            x = mint.cat(
                (
                    x[:, :1],
                    self.register_tokens.broadcast_to((x.shape[0], -1, -1)),
                    x[:, 1:],
                ),
                dim=1,
            )

        return x

    def forward_features_list(self, x_list, masks_list):
        x = [self.prepare_tokens_with_masks(x, masks) for x, masks in zip(x_list, masks_list)]
        for blk in self.blocks:
            x = blk(x)

        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
                    "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    # ********** Modified by Zexin He in 2023-2024 **********
    def forward_features(self, x, masks=None, mod=None):
        if isinstance(x, list):
            raise DeprecationWarning("forward_features_list is deprecated, use forward_features")
            return self.forward_features_list(x, masks)

        x = self.prepare_tokens_with_masks(x, masks)

        if mod is None:
            for blk in self.blocks:
                x = blk(x)
        else:
            for blk in self.blocks:
                x = blk(x, mod)

        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }

    # ********************************************************

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: ms.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> Tuple[Union[ms.Tensor, Tuple[ms.Tensor]]]:
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens :] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def construct(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])


def init_weights_vit_timm(module: nn.Cell, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Dense):
        weight = initializer(TruncatedNormal(sigma=0.02, mean=0.0, a=-2.0, b=2.0), module.weight.shape)
        module.weight.set_data(weight)
        if module.bias is not None:
            bias_weight = initializer("zeros", module.bias.shape)
            module.bias.set_data(bias_weight)


# ********** Modified by Zexin He in 2023-2024 **********
# block class selected from Block and BlockWithModulation


def _block_cls(**kwargs):
    modulation_dim = kwargs.get("modulation_dim", None)
    if modulation_dim is None:
        block_cls = Block
    else:
        block_cls = BlockWithModulation
    return block_cls


def vit_small(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(_block_cls(**kwargs), attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_base(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(_block_cls(**kwargs), attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(_block_cls(**kwargs), attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, num_register_tokens=0, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        block_fn=partial(_block_cls(**kwargs), attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


# ********************************************************
