from typing import Optional

import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import Parameter, Tensor, ops
from mindspore.common.initializer import Normal, initializer

from ..utils._common import LayerNorm, QuickGELU


class ResidualAttentionBlock(nn.Cell):
    def __init__(self, d_model, n_head, epsilon, use_quick_gelu, dtype=ms.float32, is_cross_attention=False):
        super(ResidualAttentionBlock, self).__init__()
        self.dtype = dtype
        self.attn = nn.MultiheadAttention(d_model, n_head, dtype=dtype)
        self.ln_1 = LayerNorm([d_model], epsilon=epsilon).to_float(dtype)
        self.c_fc = nn.Dense(d_model, d_model * 4).to_float(dtype)

        # In original implementation, CLIP uses fast_gelu. but OpenCLIP uses gelu, referring to:
        # https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/blob/main/config.json
        # https://huggingface.co/openai/clip-vit-large-patch14/blob/main/config.json
        if use_quick_gelu:
            self.gelu = QuickGELU()
        else:
            self.gelu = nn.GELU(approximate=False)

        if is_cross_attention:
            self.ln_1_kv = LayerNorm([d_model], epsilon=epsilon).to_float(dtype)

        self.c_proj = nn.Dense(d_model * 4, d_model).to_float(dtype)
        self.mlp = nn.SequentialCell([self.c_fc, self.gelu, self.c_proj])
        self.ln_2 = LayerNorm([d_model], epsilon=epsilon).to_float(dtype)

    def attention(self, q_x, k_x=None, v_x=None, attn_mask=None):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x
        attn_mask = attn_mask.to(self.dtype) if attn_mask is not None else None
        return self.attn(q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask)[0]

    def construct(self, q_x, k_x=None, v_x=None, attn_mask=None):
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None

        x = q_x + self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


def text_global_pool(x, text: Optional[Tensor] = None, pool_type: str = "argmax"):
    if pool_type == "first":
        pooled, tokens = x[:, 0], x[:, 1:]
    elif pool_type == "last":
        pooled, tokens = x[:, -1], x[:, :-1]
    elif pool_type == "argmax":
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        assert text is not None
        pooled, tokens = x[ops.arange(x.shape[0]), text.argmax(axis=-1)], x
    else:
        pooled = tokens = x

    return pooled, tokens


class Transformer(nn.Cell):
    def __init__(
        self,
        width,
        layers,
        heads,
        epsilon,
        use_quick_gelu,
        dtype=ms.float32,
    ):
        super(Transformer, self).__init__()
        self.dtype = dtype
        self.width = width
        self.layers = layers
        self.resblocks = nn.SequentialCell(
            *[ResidualAttentionBlock(width, heads, epsilon, use_quick_gelu, dtype=dtype) for _ in range(layers)]
        )

    def construct(self, x, attn_mask=None):
        for res in self.resblocks:
            x = res(x, attn_mask=attn_mask)
        return x


class TextEncoder(nn.Cell):
    def __init__(
        self,
        context_length: Optional[int] = 77,
        vocab_size: Optional[int] = 49408,
        width: Optional[int] = 512,
        heads: Optional[int] = 8,
        layers: Optional[int] = 12,
        epsilon: Optional[float] = 1e-5,
        output_dim: Optional[int] = 512,
        embed_cls: Optional[bool] = False,
        no_causal_mask: Optional[bool] = False,
        pad_id: Optional[int] = 0,
        pool_type: Optional[str] = "argmax",
        proj_bias: Optional[bool] = False,
        output_tokens: Optional[bool] = False,
        use_quick_gelu: Optional[bool] = False,
        dtype=ms.float32,
    ):
        super(TextEncoder, self).__init__()
        assert pool_type in ("first", "last", "argmax", "none")
        self.output_tokens = output_tokens
        self.num_pos = self.context_length = context_length
        self.vocab_size = vocab_size
        self.dtype = dtype
        self.width = width
        self.output_dim = output_dim
        self.heads = heads
        self.pad_id = pad_id
        self.pool_type = pool_type
        self.embedding_table = Parameter(initializer(Normal(sigma=0.02), [vocab_size, width], dtype=self.dtype))
        if embed_cls:
            self.cls_emb = Parameter(initializer(Normal(sigma=0.01), [width], dtype=self.dtype))
            self.num_pos += 1
        else:
            self.cls_emb = None
        self.positional_embedding = Parameter(initializer(Normal(sigma=0.01), [self.num_pos, width], dtype=self.dtype))
        self.transformer_layer = Transformer(
            width,
            layers,
            heads,
            epsilon=epsilon,
            use_quick_gelu=use_quick_gelu,
            dtype=self.dtype,
        )
        self.ln_final = LayerNorm([self.width], epsilon=epsilon).to_float(self.dtype)
        if proj_bias:
            self.text_projection = nn.Dense(width, output_dim)
        else:
            self.text_projection = Parameter(
                initializer(Normal(sigma=width**-0.5), [width, output_dim], dtype=self.dtype)
            )
        if no_causal_mask:
            self.attn_mask = None
        else:
            self.attn_mask = self.build_attntion_mask(self.num_pos)

        self.gather = ops.Gather()
        self.reshape = ops.Reshape()
        self.cast = ops.Cast()

    # @staticmethod
    def build_attntion_mask(self, context_length):
        mask = Tensor(np.triu(np.full((context_length, context_length), -np.inf).astype(np.float32), 1))
        return mask

    def build_cls_mask(self, text):
        cls_mask = (text != self.pad_id).unsqueeze(1)
        cls_mask = ops.pad(cls_mask, (1, 0, cls_mask.shape[2], 0))
        additive_mask = ms.numpy.full(cls_mask.shape, 0).astype(ms.float32)
        additive_mask.masked_fill(~cls_mask, float("-inf"))
        additive_mask = ops.repeat_interleave(additive_mask, self.heads, axis=0)
        return additive_mask

    def construct(self, text):
        bsz, ctx_len = text.shape
        flatten_id = text.flatten()
        gather_result = self.gather(self.embedding_table, flatten_id, 0)
        x = self.reshape(gather_result, (bsz, ctx_len, -1))
        attn_mask = self.attn_mask
        if self.cls_emb is not None:
            ctx_len += 1
            x = ops.cat([x, self.cls_emb + ops.zeros((x.shape[0], 1, x.shape[-1]), dtype=self.dtype)], axis=1)
            cls_mask = self.build_cls_mask(text)
            if attn_mask is not None:
                attn_mask = attn_mask[None, :ctx_len, :ctx_len] + cls_mask[:, :ctx_len, :ctx_len]

        x = x + self.positional_embedding[:ctx_len]
        x = x.transpose(1, 0, 2)
        x = self.transformer_layer(x, attn_mask=attn_mask)
        x = x.transpose(1, 0, 2)

        if self.cls_emb is not None:
            pooled, tokens = text_global_pool(x, pool_type="last")
            pooled = self.ln_final(pooled)
        else:
            x = self.ln_final(x)
            pooled, tokens = text_global_pool(x, pool_type=self.pool_type)

        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Dense):
                pooled = self.text_projection(pooled)
            else:
                pooled = pooled @ self.text_projection

        if self.output_tokens:
            return pooled, tokens

        return pooled


class MultimodalTransformer(Transformer):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        epsilon: float = 1e-5,
        use_quick_gelu: bool = False,
        context_length: int = 77,
        output_dim: int = 512,
        dtype=ms.float32,
    ):
        super().__init__(
            width=width, layers=layers, heads=heads, epsilon=epsilon, use_quick_gelu=use_quick_gelu, dtype=dtype
        )
        self.context_length = context_length
        self.cross_attn = nn.SequentialCell(
            *[
                ResidualAttentionBlock(width, heads, epsilon, use_quick_gelu, dtype=dtype, is_cross_attention=True)
                for _ in range(layers)
            ]
        )
        self.attn_mask = self.build_attntion_mask(self.context_length)
        self.ln_final = LayerNorm([width], epsilon=epsilon).to_float(self.dtype)
        self.text_projection = Parameter(
            initializer(Normal(sigma=width**-0.5), [width, output_dim], dtype=self.dtype)
        )

    @staticmethod
    def build_attntion_mask(context_length):
        mask = Tensor(np.triu(np.full((context_length, context_length), -np.inf).astype(np.float32), 1))
        return mask

    def construct(self, image_embs, text_embs):
        text_embs = text_embs.permute(1, 0, 2)  # NLD -> LND
        image_embs = image_embs.permute(1, 0, 2)  # NLD -> LND
        seq_len = text_embs.shape[0]

        for resblock, cross_attn in zip(self.resblocks, self.cross_attn):
            text_embs = resblock(text_embs, attn_mask=self.attn_mask[:seq_len, :seq_len])
            text_embs = cross_attn(text_embs, k_x=image_embs, v_x=image_embs)

        x = text_embs.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        if self.text_projection is not None:
            x = x @ self.text_projection

        return x
