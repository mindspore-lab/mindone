from collections import OrderedDict

import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import Parameter, Tensor, ops
from mindspore.common.initializer import TruncatedNormal, initializer

from ._common import LayerNorm, QuickGELU


class SequentialCellWithMaskInput(nn.SequentialCell):
    def construct(self, input_data, mask):
        for cell in self.cell_list:
            input_data = cell(input_data, mask)
        return input_data


class MultiheadAttention(nn.Cell):
    def __init__(self, d_model, n_head, dtype=ms.float32):
        """

        :param d_model: width of tensor/embedding dim
        :param n_head: output of mutlithead attention/num_heads
        """
        super(MultiheadAttention, self).__init__()
        self.embed_dim = d_model
        self.num_heads = n_head
        self.head_dim = self.embed_dim // self.num_heads
        self.in_proj = nn.Dense(self.embed_dim, 3 * self.embed_dim).to_float(dtype)
        self.out_proj = nn.Dense(self.embed_dim, self.embed_dim).to_float(dtype)
        self.split = ops.Split(-1, 3)
        self.expand_dims = ops.ExpandDims()
        self.softmax = nn.Softmax(-1)
        self.transpose = ops.Transpose()
        self.scaling = self.head_dim**-0.5
        self.dtype = dtype

    def construct(self, query, key, value, attn_mask):
        tgt_len, bsz, embed_dim = query.shape
        qkv = self.in_proj(query).view(tgt_len, bsz, 3, embed_dim).transpose((2, 0, 1, 3))
        q = qkv[0:1]
        k = qkv[1:2]
        v = qkv[2:3]
        q = ops.Squeeze(0)(q)
        k = ops.Squeeze(0)(k)
        v = ops.Squeeze(0)(v)
        q = q * self.scaling
        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose((1, 0, 2))  # (bs) x (HW + 1) x h
        k = k.view(-1, bsz * self.num_heads, self.head_dim).transpose((1, 0, 2))  # (bs) x (HW + 1) x h
        v = v.view(-1, bsz * self.num_heads, self.head_dim).transpose((1, 0, 2))  # (bs) x (HW + 1) x h
        attn_output_weights = ops.matmul(q, k.transpose((0, 2, 1)))  # bs x (HW + 1) x (HW + 1)
        attn_output_weights += self.expand_dims(attn_mask, 0)
        attn_output_weights = self.softmax(attn_output_weights)  # bs x (HW + 1) x (HW + 1)
        attn_output = ops.matmul(attn_output_weights.astype(self.dtype), v)  # bs x (HW + 1) x h
        attn_output = self.transpose(attn_output, (1, 0, 2))
        attn_output = attn_output.view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output


class AttentionWithMask(nn.Cell):
    def __init__(self, d_model, n_head, attn_mask, dtype=ms.float32):
        super(AttentionWithMask, self).__init__()
        self.attn = MultiheadAttention(d_model, n_head, dtype=dtype)
        self.attn_mask = Tensor(attn_mask, dtype)

    def construct(self, x):
        return self.attn(x, x, x, self.attn_mask)


class ResidualAttentionBlock(nn.Cell):
    def __init__(self, d_model, n_head, attn_mask, epsilon, use_quick_gelu, dtype=ms.float32, upcast_attn=False):
        super(ResidualAttentionBlock, self).__init__()
        self.attn = AttentionWithMask(d_model, n_head, attn_mask, dtype=dtype)
        self.ln_1 = (
            LayerNorm([d_model], epsilon=epsilon).to_float(ms.float32)
            if upcast_attn
            else LayerNorm([d_model], epsilon=epsilon)
        )
        self.c_fc = nn.Dense(d_model, d_model * 4).to_float(dtype)

        # In original implementation, CLIP uses fast_gelu. but OpenCLIP uses gelu, referring to:
        # https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/blob/main/config.json
        # https://huggingface.co/openai/clip-vit-large-patch14/blob/main/config.json
        if use_quick_gelu:
            self.gelu = QuickGELU()
        else:
            self.gelu = nn.GELU()

        self.c_proj = nn.Dense(d_model * 4, d_model).to_float(dtype)
        self.mlp = nn.SequentialCell([self.c_fc, self.gelu, self.c_proj])
        self.ln_2 = (
            LayerNorm([d_model], epsilon=epsilon).to_float(ms.float32)
            if upcast_attn
            else LayerNorm([d_model], epsilon=epsilon)
        )

    def construct(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Cell):
    def __init__(
        self,
        width,
        layers,
        heads,
        attn_mask,
        epsilon,
        use_quick_gelu,
        dtype=ms.float32,
        upcast_attn=False,
    ):
        super(Transformer, self).__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.SequentialCell(
            *[
                ResidualAttentionBlock(
                    width, heads, attn_mask, epsilon, use_quick_gelu, dtype=dtype, upcast_attn=upcast_attn
                )
                for _ in range(layers)
            ]
        )

    def construct(self, x):
        return self.resblocks(x)


class TextEncoder(nn.Cell):
    def __init__(
        self,
        context_length,
        vocab_size,
        output_dim,
        width,
        layers,
        heads,
        epsilon,
        use_quick_gelu,
        dtype=ms.float32,
        upcast_attn=False,
    ):
        super(TextEncoder, self).__init__()
        self.dtype = dtype
        self.width = width
        self.layers = layers
        self.vocab_size = vocab_size
        self.embedding_table = Parameter(initializer(TruncatedNormal(0.02), [vocab_size, width], dtype=self.dtype))
        self.gather = ops.Gather()
        self.reshape = ops.Reshape()
        self.cast = ops.Cast()

        self.positional_embedding = Parameter(
            initializer(TruncatedNormal(0.01), [context_length, width], dtype=self.dtype)
        )
        self.ln_final = (
            LayerNorm([self.width], epsilon=epsilon).to_float(ms.float32)
            if upcast_attn
            else LayerNorm([self.width], epsilon=epsilon)
        )
        self.transformer_layer = Transformer(
            width,
            layers,
            heads,
            self.build_attntion_mask(context_length),
            epsilon=epsilon,
            use_quick_gelu=use_quick_gelu,
            dtype=self.dtype,
            upcast_attn=upcast_attn,
        )

    @staticmethod
    def build_attntion_mask(context_length):
        mask = np.triu(np.full((context_length, context_length), -np.inf).astype(np.float32), 1)
        return mask

    def construct(self, text):
        bsz, ctx_len = text.shape
        flatten_id = text.flatten()
        gather_result = self.gather(self.embedding_table, flatten_id, 0)

        x = self.reshape(gather_result, (bsz, ctx_len, -1))
        x = x + self.positional_embedding
        x = x.transpose(1, 0, 2)
        x = self.transformer_layer(x)
        x = x.transpose(1, 0, 2)
        x = self.ln_final(x)
        return x


class OpenCLIPResidualAttentionBlock(nn.Cell):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        is_cross_attention: bool = False,
        epsilon: float = 1e-5,
        use_quick_gelu: bool = False,
        dtype=ms.float32,
        upcast_attn=False,
    ):
        super().__init__()
        self.dtype = dtype
        self.ln_1 = (
            LayerNorm((d_model,), epsilon=epsilon).to_float(ms.float32)
            if upcast_attn
            else LayerNorm((d_model,), epsilon=epsilon)
        )
        self.attn = MultiheadAttention(d_model, n_head, dtype=dtype)
        if is_cross_attention:
            self.ln_1_kv = (
                LayerNorm((d_model,), epsilon=epsilon).to_float(ms.float32)
                if upcast_attn
                else LayerNorm((d_model,), epsilon=epsilon)
            )

        self.ln_2 = (
            LayerNorm([d_model], epsilon=epsilon).to_float(ms.float32)
            if upcast_attn
            else LayerNorm([d_model], epsilon=epsilon)
        )
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.SequentialCell(
            OrderedDict(
                [
                    ("c_fc", nn.Dense(d_model, mlp_width).to_float(self.dtype)),
                    ("gelu", QuickGELU().to_float(self.dtype) if use_quick_gelu else nn.GELU().to_float(self.dtype)),
                    ("c_proj", nn.Dense(mlp_width, d_model).to_float(self.dtype)),
                ]
            )
        )

    def construct(self, x, attn_mask):
        x = x + self.attn(self.ln_1(x), None, None, attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class OpenCLIPTransformer(nn.Cell):
    def __init__(self, width, layers, heads, epsilon, use_quick_gelu, dtype=ms.float32, upcast_attn=False):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = SequentialCellWithMaskInput(
            *[
                OpenCLIPResidualAttentionBlock(
                    width, heads, epsilon=epsilon, use_quick_gelu=use_quick_gelu, dtype=dtype, upcast_attn=upcast_attn
                )
                for _ in range(layers)
            ]
        )

    def construct(self, x, attn_mask):
        return self.resblocks(x, attn_mask)


class OpenClipTextEncoder(nn.Cell):
    def __init__(
        self,
        context_length,
        vocab_size,
        output_dim,
        width,
        layers,
        heads,
        epsilon,
        use_quick_gelu,
        dtype=ms.float32,
        upcast_attn=False,
    ):
        super().__init__()
        self.dtype = dtype
        self.width = width
        self.layers = layers
        self.vocab_size = vocab_size
        self.token_embedding = (
            nn.Embedding(vocab_size, width).to_float(self.dtype) if upcast_attn else nn.Embedding(vocab_size, width)
        )

        self.transformer = OpenCLIPTransformer(
            width, layers, heads, epsilon, use_quick_gelu, dtype=self.dtype, upcast_attn=upcast_attn
        )
        self.positional_embedding = Parameter(
            initializer(TruncatedNormal(0.01), [context_length, width], dtype=self.dtype)
        )
        self.ln_final = nn.LayerNorm((self.width,), epsilon=epsilon).to_float(ms.float32)
        self.attn_mask = ms.Tensor(self.build_attntion_mask(context_length), dtype=self.dtype)

    @staticmethod
    def build_attntion_mask(context_length):
        mask = np.triu(np.full((context_length, context_length), -np.inf).astype(np.float32), 1)
        return mask

    def construct(self, text):
        x = self.token_embedding(text)
        x = x + self.positional_embedding
        x = x.transpose(1, 0, 2)
        x = self.transformer(x, self.attn_mask)
        x = x.transpose(1, 0, 2)
        x = self.ln_final(x)
        return x
