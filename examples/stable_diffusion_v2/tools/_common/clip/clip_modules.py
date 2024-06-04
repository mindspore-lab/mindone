"""
Modules of CLIPModel, including MultiheadAttention，VisionTransformer,
QuickGELU, ResidualAttentionBlock, Transformer
"""
from collections import OrderedDict
from typing import Optional

import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, Tensor
from mindspore import dtype as mstype
from mindspore.common.initializer import Normal
from mindspore.ops import operations as P


class LayerNorm(nn.LayerNorm):
    r"""Implementation That Supports Fp16 Inputs But Fp32 Gains Biases.

    Args:
        x (ms.Tensor): Input tensor.
            The detailed function could refer to mindspore.nn.LayerNorm.

    Return:
        y (ms.Tensor): Normalized tensor.
    """

    def construct(self, x: ms.Tensor):
        """construct"""
        y = super().construct(P.Cast()(x, ms.float32))
        y = P.Cast()(y, x.dtype)
        return y


class QuickGELU(nn.Cell):
    """
    A quick approximation of the GELU activation function.
    """

    def construct(self, x: Tensor) -> Tensor:
        return x * ops.sigmoid(1.702 * x)


class MultiheadAttention(nn.Cell):
    r"""MultiheadAttention, With Layers As Input For Initialization

    Args:
        d_model (int): The feature dimension
        n_head (int): The number of attention heads
        layers (int): The number of transformers, used for weight initialization
        dtype (mstype): The type of calculation, [mstype.float32, mstype.float16].
    """

    def __init__(self, d_model: int, n_head: int, layers: int, dtype: mstype):
        super(MultiheadAttention, self).__init__()

        self.num_heads = n_head
        self.head_dim = d_model // n_head

        self.softmax = nn.Softmax(-1)
        self.transpose = ops.Transpose()
        self.scaling = self.head_dim**-0.5

        proj_std = (d_model**-0.5) * ((2 * layers) ** -0.5)
        attn_std = d_model**-0.5
        self.out_proj = nn.Dense(d_model, d_model, weight_init=Normal(mean=0.0, sigma=proj_std)).to_float(dtype)
        self.in_proj = nn.Dense(d_model, 3 * d_model, weight_init=Normal(mean=0.0, sigma=attn_std)).to_float(dtype)

    def construct(self, query: ms.Tensor, attn_mask: Optional[ms.Tensor] = None):
        r"""Construct

        Args:
            query (ms.Tensor): query of attention.
            attn_mask (Optional[ms.Tensor]): attention mask.

        Returns:
            attn_output (ms.Tensor): attention output.
        """
        len_tgt, batch_size, width = query.shape
        qkv = self.in_proj(query).view(len_tgt, batch_size, 3, width).transpose((2, 0, 1, 3))

        att_q = qkv[0:1]
        att_q = ops.Squeeze(0)(att_q)
        att_q = att_q * self.scaling
        att_q = att_q.view(len_tgt, batch_size * self.num_heads, self.head_dim).transpose((1, 0, 2))

        att_k = qkv[1:2]
        att_k = ops.Squeeze(0)(att_k)
        att_k = att_k.view(-1, batch_size * self.num_heads, self.head_dim).transpose((1, 0, 2))

        att_v = qkv[2:3]
        att_v = ops.Squeeze(0)(att_v)
        att_v = att_v.view(-1, batch_size * self.num_heads, self.head_dim).transpose((1, 0, 2))

        if attn_mask is not None:
            attn_output_weights = attn_mask + ops.matmul(att_q, att_k.transpose((0, 2, 1)))
        else:
            attn_output_weights = ops.matmul(att_q, att_k.transpose((0, 2, 1)))
        attn_output_weights = self.softmax(attn_output_weights)
        attn_output = ops.matmul(attn_output_weights, att_v)
        attn_output = self.transpose(attn_output, (1, 0, 2))
        attn_output = attn_output.view(len_tgt, batch_size, width)
        attn_output = self.out_proj(attn_output)
        return attn_output


class VisionTransformer(nn.Cell):
    r"""VisionTransformer Of CLIPModel

    Args:
        input_resolution (int): The image size of input.
        patch_size (int): The patch size of vision transformer.
        width (int): The dimension of vision transformer.
        layers (int): The number of layers of vision transformer.
        heads (int): The number of attention heads.
        output_dim (int): The output dimension of vision transformer.
        dtype (mstype): The type of calculation, [mstype.float32, mstype.float16].
    """

    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        dtype: mstype,
        hidden_act: str,
    ):
        super(VisionTransformer, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, has_bias=False
        ).to_float(dtype)

        scale = width**-0.5
        self.class_embedding = Parameter(Tensor(scale * np.random.normal(0, 1, size=(width)), dtype))
        self.positional_embedding = Parameter(
            Tensor(scale * np.random.normal(0, 1, size=((input_resolution // patch_size) ** 2 + 1, width)), dtype)
        )
        self.ln_pre = LayerNorm([width], epsilon=1e-5)
        self.transformer = Transformer(width, layers, heads, dtype, hidden_act)
        self.ln_post = LayerNorm([width], epsilon=1e-5)
        self.proj = Parameter(Tensor(scale * np.random.normal(0, 1, size=(width, output_dim)), dtype))
        self.cat = ops.Concat(1)
        self.tile = ops.Tile()
        self.slice = P.StridedSlice()
        self.dtype = dtype

    def construct(self, input_x: ms.Tensor):
        r"""Construct

        Args:
            input_x (ms.Tensor): Input tensor.

        Returns:
            input_x (ms.Tensor): Output tensor.
        """
        input_x = self.conv1(input_x)
        input_x = input_x.reshape(input_x.shape[0], input_x.shape[1], -1)
        input_x = input_x.transpose(0, 2, 1)
        class_embedding = self.tile(self.class_embedding, (input_x.shape[0] + 1, 1, 1)).astype(self.dtype)
        input_x = self.cat(
            [
                self.slice(
                    class_embedding, (0, 0, 0), (-1, class_embedding.shape[1], class_embedding.shape[2]), (1, 1, 1)
                ),
                input_x,
            ]
        )
        input_x = ops.Add()(input_x, self.positional_embedding)
        input_x = self.ln_pre(input_x)
        input_x = input_x.transpose(1, 0, 2)
        input_x = self.transformer(input_x)
        input_x = input_x.transpose(1, 0, 2)
        input_x = self.ln_post(input_x[:, 0, :])
        input_x = ops.matmul(input_x, self.proj)
        return input_x


class ResidualAttentionBlock(nn.Cell):
    r"""
    ResidualAttentionBlock of CLIP

    Args:
        d_model (int): The dimension of features.
        n_head (int): The number of attention heads.
        layers (int): The number of transformer layers for weight initialization.
        dtype (mstype): The type of calculation, [mstype.float32, mstype.float16].
        attn_mask (Optional[ms.Tensor]): attention mask.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        layers: int,
        dtype: mstype,
        hidden_act: str,
        attn_mask: Optional[ms.Tensor] = None,
    ):
        super().__init__()

        proj_std = (d_model**-0.5) * ((2 * layers) ** -0.5)
        fc_std = (2 * d_model) ** -0.5
        self.dtype = dtype
        self.attn = MultiheadAttention(d_model, n_head, layers, dtype)
        self.ln_1 = LayerNorm([d_model], epsilon=1e-5)

        assert hidden_act in ["gelu", "quick_gelu"], "`hidden_act` should be `gelu` or `quick_gelu`."
        gelu = nn.GELU if hidden_act == "gelu" else QuickGELU

        self.mlp = nn.SequentialCell(
            OrderedDict(
                [
                    (
                        "c_fc",
                        nn.Dense(d_model, d_model * 4, weight_init=Normal(mean=0.0, sigma=fc_std)).to_float(dtype),
                    ),
                    ("gelu", gelu()),
                    (
                        "c_proj",
                        nn.Dense(d_model * 4, d_model, weight_init=Normal(mean=0.0, sigma=proj_std)).to_float(dtype),
                    ),
                ]
            )
        )
        self.ln_2 = LayerNorm([d_model], epsilon=1e-5)
        self.attn_mask = attn_mask

    def construct(self, input_x: ms.Tensor):
        r"""Construct"""
        input_x = ops.Add()(input_x, self.attention(self.ln_1(input_x)))
        input_x = ops.Add()(input_x, self.mlp(self.ln_2(input_x)))
        return input_x

    def attention(self, input_x: ms.Tensor):
        r"""Attention"""
        return self.attn(input_x, self.attn_mask)


class Transformer(nn.Cell):
    r"""
    Text Transformer of CLIP

    Args:
        width (int): The dimension of input features.
        layers (int): The number of transformer layers.
        heads (int): The number of attention heads.
        attn_mask (ms.Tensor):  Attention mask.
        dtype (mstype): The type of calculation, [mstype.float32, mstype.float16].
    """

    def __init__(self, width, layers, heads, dtype, hidden_act, attn_mask=None):
        super(Transformer, self).__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.SequentialCell(
            *[ResidualAttentionBlock(width, heads, layers, dtype, hidden_act, attn_mask) for _ in range(layers)]
        )

    def construct(self, input_x):
        r"""Construct"""
        return self.resblocks(input_x)
