# imagebind mindspore
import copy
import fnmatch
import logging
from functools import partial
from typing import Callable, List

import numpy as np

import mindspore as ms
from mindspore import nn, ops, Parameter, Tensor
from mindspore.common.initializer import (
    Constant,
    TruncatedNormal,
    initializer,
)

from .helpers import DropPath, constant_, trunc_normal_

class Attention(nn.Cell):
    """
    Attention layer implementation, Rearrange Input -> B x N x hidden size.

    Args:
        dim (int): The dimension of input features.
        num_heads (int): The number of attention heads. Default: 8.
        qkv_bias (bool): Specifies whether the linear layer uses a bias vector. Default: True.
        qk_norm (bool): Specifies whether to do normalization to q and k.
        attn_drop (float): The drop rate of attention, greater than 0 and less equal than 1. Default: 0.0.
        proj_drop (float): The drop rate of output, greater than 0 and less equal than 1. Default: 0.0.

    Returns:
        Tensor, output tensor.

    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        attn_drop: float = 0.5,
        proj_drop: float = 0.5,
        norm_layer: nn.Cell = nn.LayerNorm,
    ):
        super(Attention, self).__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = Tensor(self.head_dim ** -0.5)

        self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)

        # TODO: align dropout params.
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.mul = ops.Mul()
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.unstack = ops.Unstack(axis=0)
        self.attn_matmul_v = ops.BatchMatMul()
        self.q_matmul_k = ops.BatchMatMul(transpose_b=True)

    def construct(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x)
        qkv = self.reshape(qkv, (b, n, 3, self.num_heads, self.head_dim))
        qkv = self.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = self.unstack(qkv)
        

        attn = self.q_matmul_k(q, k) * self.scale

        attn = ops.softmax(attn.astype(ms.float32), axis=-1).astype(attn.dtype)
        attn = self.attn_drop(attn)

        out = self.attn_matmul_v(attn, v)
        out = self.transpose(out, (0, 2, 1, 3))
        out = self.reshape(out, (b, n, c))
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class Mlp(nn.Cell):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Dense(hidden_features, out_features)
        self.drop = nn.Dropout(p=drop)

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ViTAttention(Attention):
    def construct(self, x: ms.Tensor, attn_mask: ms.Tensor):
        assert attn_mask is None
        return super().construct(x)


class BlockWithMasking(nn.Cell):
    def __init__(
        self,
        dim: int,
        attn_target: Callable,
        mlp_ratio: int = 4,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        ffn_dropout_rate: float = 0.0,
        drop_path: float = 0.0,
        layer_scale_type: str = None,
        layer_scale_init_value: float = 1e-4,
    ):
        super().__init__()

        self.attn = attn_target()

        if drop_path > 0.0:
            self.drop_path = DropPath(drop_path)
        else:
            self.drop_path = nn.Identity()
        self.drop_path = nn.Identity()

        self.norm_1 = norm_layer((dim,))
        mlp_hidden_dim = int(mlp_ratio * dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=ffn_dropout_rate,
        )
        self.norm_2 = norm_layer((dim,))
        self.layer_scale_type = layer_scale_type
        if self.layer_scale_type is not None:
            assert self.layer_scale_type in [
                "per_channel",
                "scalar",
            ], f"Found Layer scale type {self.layer_scale_type}"
            if self.layer_scale_type == "per_channel":
                # one gamma value per channel
                gamma_shape = [1, 1, dim]
            elif self.layer_scale_type == "scalar":
                # single gamma value for all channels
                gamma_shape = [1, 1, 1]
            # two gammas: for each part of the fwd in the encoder
            self.layer_scale_gamma1 = nn.Parameter(
                ops.ones(shpae=gamma_shape) * layer_scale_init_value,
                requires_grad=True,
            )
            self.layer_scale_gamma2 = nn.Parameter(
                ops.ones(shape=gamma_shape) * layer_scale_init_value,
                requires_grad=True,
            )

    def construct(self, x: ms.Tensor, attn_mask: ms.Tensor):
        norm_1_x = self.norm_1(x)
        norm_2_x = self.norm_2(x)
        if self.layer_scale_type is None:
            x = x + self.drop_path(self.attn(norm_1_x, norm_1_x, norm_1_x, attn_mask)[0])
            x = x + self.drop_path(self.mlp(norm_2_x))
        else:
            x = (
                x
                + self.drop_path(self.attn(norm_1_x, norm_1_x, norm_1_x, attn_mask)[0])
                * self.layer_scale_gamma1
            )
            x = x + self.drop_path(self.mlp(norm_2_x)) * self.layer_scale_gamma2
        return x


_LAYER_NORM = partial(nn.LayerNorm, epsilon=1e-6)


class SimpleTransformer(nn.Cell):
    def __init__(
        self,
        attn_target: Callable,
        embed_dim: int,
        num_blocks: int,
        block: Callable = BlockWithMasking,
        pre_transformer_layer: Callable = None,
        post_transformer_layer: Callable = None,
        drop_path_rate: float = 0.5,
        drop_path_type: str = "progressive",
        norm_layer: Callable = _LAYER_NORM,
        mlp_ratio: int = 4,
        ffn_dropout_rate: float = 0.5,
        layer_scale_type: str = None,  # from cait; possible values are None, "per_channel", "scalar"
        layer_scale_init_value: float = 1e-4,  # from cait; float
    ):

        super().__init__()
        self.pre_transformer_layer = pre_transformer_layer

        if drop_path_type == "progressive":
            dpr = [x.item() for x in np.linspace(0, drop_path_rate, num_blocks)]
        elif drop_path_type == "uniform":
            dpr = [drop_path_rate for i in range(num_blocks)]
        else:
            raise ValueError(f"Unknown drop_path_type: {drop_path_type}")

        self.blocks = nn.SequentialCell(
            *[
                block(
                    dim=embed_dim,
                    attn_target=attn_target,
                    mlp_ratio=mlp_ratio,
                    ffn_dropout_rate=ffn_dropout_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    layer_scale_type=layer_scale_type,
                    layer_scale_init_value=layer_scale_init_value,
                )
                for i in range(num_blocks)
            ]
        )
        self.post_transformer_layer = post_transformer_layer
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Dense):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            constant_(m.beta, 0)
            constant_(m.gamma, 1.0)

    def construct(
        self,
        tokens: ms.Tensor,
        attn_mask: ms.Tensor = None,
        use_checkpoint: bool = False,
        checkpoint_every_n: int = 1,
        checkpoint_blk_ids: List[int] = None,
    ):
        """
        Inputs
        - tokens: data of shape N x L x D (or L x N x D depending on the attention implementation)
        - attn: mask of shape L x L

        Output
        - x: data of shape N x L x D (or L x N x D depending on the attention implementation)
        """
        if self.pre_transformer_layer:
            tokens = self.pre_transformer_layer(tokens)
        if use_checkpoint and checkpoint_blk_ids is None:
            checkpoint_blk_ids = [
                blk_id
                for blk_id in range(len(self.blocks))
                if blk_id % checkpoint_every_n == 0
            ]
        if checkpoint_blk_ids:
            checkpoint_blk_ids = set(checkpoint_blk_ids)
        for blk_id, blk in enumerate(self.blocks):
            if use_checkpoint and blk_id in checkpoint_blk_ids:
                tokens = checkpoint.checkpoint(
                    blk, tokens, attn_mask, use_reentrant=False
                )
            else:
                tokens = blk(tokens, attn_mask=attn_mask)
        if self.post_transformer_layer:
            tokens = self.post_transformer_layer(tokens)
        return tokens


