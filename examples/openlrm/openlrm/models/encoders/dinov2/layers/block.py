# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

# ******************************************************************************
#   Code modified by Zexin He in 2023-2024.
#   Modifications are marked with clearly visible comments
#   licensed under the Apache License, Version 2.0.
# ******************************************************************************

import logging
from typing import Callable

from mindspore import Tensor, nn, ops

from .attention import Attention  # , MemEffAttention
from .drop_path import DropPath
from .layer_scale import LayerScale
from .mlp import Mlp

logger = logging.getLogger("dinov2")

XFORMERS_AVAILABLE = False


class Block(nn.Cell):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Cell] = nn.GELU,
        norm_layer: Callable[..., nn.Cell] = nn.LayerNorm,
        attn_class: Callable[..., nn.Cell] = Attention,
        ffn_layer: Callable[..., nn.Cell] = Mlp,
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer((dim,))
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer((dim,))
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def construct(self, x: Tensor) -> Tensor:
        def attn_residual_func(x: Tensor) -> Tensor:
            return self.ls1(self.attn(self.norm1(x)))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x))
            x = x + self.drop_path1(ffn_residual_func(x))
        else:
            x = x + attn_residual_func(x)
            x = x + ffn_residual_func(x)
        return x


# ********** Modified by Zexin He in 2023-2024 **********
# Override forward with modulation input
class BlockWithModulation(Block):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def construct(self, x: Tensor, mod: Tensor) -> Tensor:
        def attn_residual_func(x: Tensor, mod: Tensor) -> Tensor:
            return self.ls1(self.attn(self.norm1(x, mod)))

        def ffn_residual_func(x: Tensor, mod: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x, mod)))

        if self.training and self.sample_drop_ratio > 0.1:
            raise NotImplementedError("Modulation with drop path ratio larger than 0.1 is not supported yet")
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x, mod))
            x = x + self.drop_path1(ffn_residual_func(x, mod))  # FIXME: drop_path2
        else:
            x = x + attn_residual_func(x, mod)
            x = x + ffn_residual_func(x, mod)
        return x


# ********************************************************


def drop_add_residual_stochastic_depth(
    x: Tensor,
    residual_func: Callable[[Tensor], Tensor],
    sample_drop_ratio: float = 0.0,
) -> Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (ops.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    residual = residual_func(x_subset)

    x_flat = x.flatten(start_dim=1)
    residual = residual.flatten(start_dim=1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    alpha = residual_scale_factor
    x_plus_residual = ops.index_add(x_flat, axis=0, indices=brange, y=residual.to(dtype=x.dtype) * alpha)
    return x_plus_residual.view_as(x)
