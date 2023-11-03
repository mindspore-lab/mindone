# reference to https://github.com/mlfoundations/open_clip

import math
from collections import OrderedDict
from typing import Callable, Optional, Tuple

import numpy as np
from gm.modules.util import linear, normalize

import mindspore as ms
from mindspore import Parameter, Tensor, nn, ops
from mindspore.common import initializer as init


class LayerNormFp32(nn.LayerNorm):
    """Subclass mindspore's LayerNorm to handle fp16 (by casting to float32 and back)."""

    def __init__(self, *args, **kwargs):
        super(LayerNormFp32, self).__init__(*args, **kwargs)
        ms.amp.auto_mixed_precision(self, amp_level="O0")  # fp32

    def construct(self, x: Tensor):
        orig_type = x.dtype
        y, _, _ = self.layer_norm(x.astype(ms.float32), self.gamma.astype(ms.float32), self.beta.astype(ms.float32))
        return y.astype(orig_type)


class LayerScale(nn.Cell):
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = Parameter(Tensor(init_values * np.ones(dim), ms.float32))

    def construct(self, x):
        return x * self.gamma


class MultiheadAttention(nn.Cell):
    def __init__(self, d_model, n_head):
        """

        :param d_model: width of tensor/embedding dim
        :param n_head: output of mutlithead attention/num_heads
        """
        super(MultiheadAttention, self).__init__()
        self.embed_dim = d_model
        self.num_heads = n_head
        self.head_dim = self.embed_dim // self.num_heads

        # self.in_proj = nn.Dense(self.embed_dim, 3 * self.embed_dim)
        self.in_proj_weight = Parameter(
            init.initializer(init.XavierUniform(), (3 * self.embed_dim, self.embed_dim), ms.float32), "in_proj_weight"
        )
        self.in_proj_bias = Parameter(init.initializer("zeros", (3 * self.embed_dim), ms.float32), "in_proj_bias")

        self.out_proj = nn.Dense(self.embed_dim, self.embed_dim)
        self.split = ops.Split(-1, 3)
        self.expand_dims = ops.ExpandDims()
        self.softmax = nn.Softmax(-1)
        self.transpose = ops.Transpose()
        self.scaling = self.head_dim**-0.5

    def construct(self, query, key, value, attn_mask):
        tgt_len, bsz, embed_dim = query.shape
        # qkv = self.in_proj(query).view(tgt_len, bsz, 3, embed_dim).transpose((2, 0, 1, 3))
        # qkv = ops.matmul(query.view(-1, embed_dim), ops.swapaxes(self.in_proj_weight, 0, 1))
        qkv = ops.MatMul(transpose_b=True)(query.view(-1, embed_dim), self.in_proj_weight)
        qkv = ops.BiasAdd()(qkv, self.in_proj_bias)
        qkv = qkv.view(tgt_len, bsz, 3, embed_dim).transpose((2, 0, 1, 3))

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
        attn_output = ops.matmul(attn_output_weights, v)  # bs x (HW + 1) x h
        attn_output = self.transpose(attn_output, (1, 0, 2))
        attn_output = attn_output.view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output


class Attention(nn.Cell):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        scaled_cosine=False,
        scale_heads=False,
        logit_scale_max=math.log(1.0 / 0.01),
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.logit_scale_max = logit_scale_max

        # keeping in_proj in this form (instead of nn.Dense) to match weight scheme of original
        self.in_proj_weight = Parameter(Tensor(np.random.randn((dim * 3, dim)) * self.scale, ms.float32))
        if qkv_bias:
            self.in_proj_bias = Parameter(Tensor(np.random.zeros(dim * 3), ms.float32))
        else:
            self.in_proj_bias = None

        if self.scaled_cosine:
            self.logit_scale = Parameter(Tensor(np.log(10 * np.ones((num_heads, 1, 1))), dtype=ms.float32))
        else:
            self.logit_scale = None
        self.attn_drop = nn.Dropout(p=attn_drop)
        if self.scale_heads:
            self.head_scale = Parameter(Tensor(np.ones((num_heads, 1, 1)), ms.float32))
        else:
            self.head_scale = None
        self.out_proj = nn.Dense(dim, dim)
        self.out_drop = nn.Dropout(p=proj_drop)

    def construct(self, x, attn_mask: Optional[Tensor] = None):
        L, N, C = x.shape
        q, k, v = linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, axis=-1)
        q = q.view(L, N * self.num_heads, -1).swapaxes(0, 1)
        k = k.view(L, N * self.num_heads, -1).swapaxes(0, 1)
        v = v.view(L, N * self.num_heads, -1).swapaxes(0, 1)

        if self.logit_scale is not None:
            attn = ops.BatchMatMul()(normalize(q, dim=-1), normalize(k, dim=-1).swapaxes(-1, -2))
            logit_scale = ops.clip(self.logit_scale, max=self.logit_scale_max).exp()
            attn = attn.view(N, self.num_heads, L, L) * logit_scale
            attn = attn.view(-1, L, L)
        else:
            q = q * self.scale
            attn = ops.BatchMatMul()(q, k.swapaxes(-1, -2))

        if attn_mask is not None:
            if attn_mask.dtype == ms.bool_:
                new_attn_mask = ops.zeros_like(attn_mask, dtype=q.dtype)
                new_attn_mask = ops.masked_fill(new_attn_mask, attn_mask, -1e5)
                attn_mask = new_attn_mask
            attn += attn_mask

        attn = ops.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = ops.BatchMatMul()(attn, v)
        if self.head_scale is not None:
            x = x.view(N, self.num_heads, L, C) * self.head_scale
            x = x.view(-1, L, C)
        x = x.swapaxes(0, 1).reshape(L, N, C)
        x = self.out_proj(x)
        x = self.out_drop(x)
        return x


class ResidualAttentionBlock(nn.Cell):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        is_cross_attention: bool = False,
    ):
        super().__init__()

        self.ln_1 = norm_layer([d_model])
        self.attn = MultiheadAttention(d_model, n_head)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        self.is_cross_attention = is_cross_attention
        if is_cross_attention:
            self.ln_1_kv = norm_layer([d_model])

        self.ln_2 = norm_layer([d_model])
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.SequentialCell(
            OrderedDict(
                [
                    ("c_fc", nn.Dense(d_model, mlp_width)),
                    ("gelu", act_layer()),
                    ("c_proj", nn.Dense(mlp_width, d_model)),
                ]
            )
        )
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def attention(
        self,
        q_x: Tensor,
        k_x: Optional[Tensor] = None,
        v_x: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ):
        if k_x is None:
            k_x = q_x
        if v_x is None:
            v_x = q_x

        if attn_mask is not None:
            attn_mask = attn_mask.astype(q_x.dtype)

        return self.attn(q_x, k_x, v_x, attn_mask=attn_mask)

    def construct(
        self,
        q_x: Tensor,
        k_x: Optional[Tensor] = None,
        v_x: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ):
        if k_x is not None and self.is_cross_attention:
            k_x = self.ln_1_kv(k_x)
        if v_x is not None and self.is_cross_attention:
            v_x = self.ln_1_kv(v_x)

        _dtype = q_x.dtype
        x = q_x + self.ls_1(self.attention(q_x=self.ln_1(q_x).astype(_dtype), k_x=k_x, v_x=v_x, attn_mask=attn_mask))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Cell):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
    ):
        super().__init__()
        self.width = width
        self.layers = layers

        self.resblocks = nn.CellList(
            [
                ResidualAttentionBlock(
                    width, heads, mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer
                )
                for _ in range(layers)
            ]
        )

    def construct(self, x: Tensor, attn_mask: Optional[Tensor] = None):
        for r in self.resblocks:
            x = r(x, attn_mask=attn_mask)

        return x


class VisionTransformer(nn.Cell):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float,
        ls_init_value: float = None,
        global_average_pool: bool = False,
        output_dim: int = 512,
        input_patchnorm: bool = False,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
    ):
        super().__init__()
        image_height, image_width = self.image_size = (image_size, image_size)
        patch_height, patch_width = self.patch_size = (patch_size, patch_size)
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.output_dim = output_dim

        # whether to layernorm each patch, as done in dual patchnorm paper - https://arxiv.org/abs/2302.01327v1
        self.input_patchnorm = input_patchnorm

        if input_patchnorm:
            patch_input_dim = patch_height * patch_width * 3
            self.patchnorm_pre_ln = nn.LayerNorm([patch_input_dim], epsilon=1e-5)
            self.conv1 = nn.Dense(patch_input_dim, width)
        else:
            self.patchnorm_pre_ln = nn.Identity()
            self.conv1 = nn.Conv2d(
                in_channels=3,
                out_channels=width,
                kernel_size=patch_size,
                stride=patch_size,
                has_bias=False,
                pad_mode="valid",
            )

        # class embeddings and positional embeddings
        scale = width**-0.5
        self.class_embedding = Parameter(scale * np.random.randn(width))
        self.positional_embedding = Parameter(scale * np.random.randn(self.grid_size[0] * self.grid_size[1] + 1, width))

        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        # self.patch_dropout = nn.Identity()

        self.ln_pre = norm_layer([width])
        self.transformer = Transformer(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

        self.global_average_pool = global_average_pool

        self.attn_pool = None
        self.ln_post = norm_layer([width])
        self.proj = Parameter(Tensor(scale * np.random.randn(width, output_dim), ms.float32))

    def _global_pool(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if self.global_average_pool:
            return x.mean(axis=1), x
        else:
            return x[:, 0], x[:, 1:]

    def construct(self, x: Tensor):
        # to patches - whether to use dual patchnorm - https://arxiv.org/abs/2302.01327v1
        if self.input_patchnorm:
            # einops - rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)')
            x = x.reshape(
                x.shape[0], x.shape[1], self.grid_size[0], self.patch_size[0], self.grid_size[1], self.patch_size[1]
            )
            x = x.transpose(0, 2, 4, 1, 3, 5)
            x = x.reshape(x.shape[0], self.grid_size[0] * self.grid_size[1], -1)
            x = self.patchnorm_pre_ln(x)
            x = self.conv1(x)
        else:
            x = self.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.transpose(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = ops.concat(
            (self.class_embedding.astype(x.dtype) + ops.zeros((x.shape[0], 1, x.shape[-1]), dtype=x.dtype), x), axis=1
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.astype(x.dtype)

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        # x = self.patch_dropout(x)
        x = self.ln_pre(x)
        x = x.transpose(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.transpose(1, 0, 2)  # LND -> NLD

        if self.attn_pool is not None:
            x = self.attn_pool(x)
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)
        else:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)

        if self.proj is not None:
            # pooled = pooled @ self.proj
            pooled = ops.matmul(pooled, self.proj)

        return pooled


class TextTransformer(nn.Cell):
    def __init__(
        self,
        context_length: int = 77,
        vocab_size: int = 49408,
        width: int = 512,
        heads: int = 8,
        layers: int = 12,
        ls_init_value: float = None,
        output_dim: int = 512,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        embed_cls: bool = False,
        pad_id: int = 0,
    ):
        super().__init__()
        self.num_pos = self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim
        self.heads = heads
        self.pad_id = pad_id

        self.text_projection = Parameter(Tensor(np.zeros((width, output_dim)), ms.float32))

        if embed_cls:
            self.cls_emb = Parameter(Tensor(np.zeros(width), ms.float32))
            self.num_pos += 1
        else:
            self.cls_emb = None

        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = Parameter(Tensor(np.zeros((self.num_pos, width)), ms.float32))
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.ln_final = norm_layer([width])

        _attn_mask_tensor = self.build_attention_mask()
        self.attn_mask = Parameter(_attn_mask_tensor, requires_grad=False)

        self.init_parameters()

    def init_parameters(self):
        for weight, sigma in zip((self.token_embedding.embedding_table, self.positional_embedding), (0.02, 0.01)):
            weight.set_data(init.initializer(init.Normal(sigma=sigma), weight.shape, weight.dtype))

        if self.cls_emb is not None:
            self.cls_emb.set_data(init.initializer(init.Normal(sigma=0.01), self.cls_emb.shape, self.cls_emb.dtype))

        proj_std = (self.transformer.width**-0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            for weight, sigma in zip(
                (block.attn.in_proj_weight, block.attn.out_proj.weight, block.mlp.c_fc.weight, block.mlp.c_proj.weight),
                (attn_std, proj_std, fc_std, proj_std),
            ):
                weight.set_data(init.initializer(init.Normal(sigma=sigma), weight.shape, weight.dtype))

        if self.text_projection is not None:
            self.text_projection.set_data(
                init.initializer(
                    init.Normal(sigma=self.transformer.width**-0.5),
                    self.text_projection.shape,
                    self.text_projection.dtype,
                )
            )

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the tokens
        # mindspore uses additive attention mask; fill with -inf
        mask = Tensor(np.ones((self.num_pos, self.num_pos)) * -1e5, ms.float32)
        mask = mask.triu(diagonal=1)  # zero out the lower diagonal
        return mask

    def build_cls_mask(self, text, dtype):
        cls_mask = (text != self.pad_id).unsqueeze(1)
        cls_mask = ops.pad(cls_mask, (1, 0, cls_mask.shape[2], 0), value=1.0)
        additive_mask = ops.zeros(cls_mask.shape, dtype)
        additive_mask = ops.masked_fill(additive_mask, ops.logical_not(cls_mask), -1e5)
        additive_mask = ops.repeat_interleave(additive_mask, self.heads, 0)
        return additive_mask

    def _repeat(self, t, N: int):
        return t.reshape(1, 1, -1).tile((N, 1, 1))

    def construct(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        seq_len = text.shape[1]

        attn_mask = self.attn_mask

        if self.cls_emb is not None:
            seq_len += 1
            x = ops.concat((x, self._repeat(self.cls_emb, x.shape[0]).astype(x.dtype)), axis=1)
            cls_mask = self.build_cls_mask(text, x.dtype)
            attn_mask = attn_mask[None, :seq_len, :seq_len] + cls_mask[:, :seq_len, :seq_len]

        x = x + self.positional_embedding[:seq_len].astype(x.dtype)
        x = x.transpose(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=attn_mask)
        x = x.transpose(1, 0, 2)  # LND -> NLD

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        if self.cls_emb is not None:
            pooled = x[:, -1]
            # tokens = x[:, :-1]
            pooled = self.ln_final(pooled)
        else:
            x = self.ln_final(x)
            pooled = x[ops.arange(x.shape[0]), text.argmax(axis=-1)]
            # tokens = x

        if self.text_projection is not None:
            # pooled = pooled @ self.text_projection
            pooled = ops.matmul(pooled, self.text_projection)

        return pooled
