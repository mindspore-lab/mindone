from typing import Optional, List

import mindspore as ms
from mindspore import nn, ops

from videogvt.models.transformer.attention import FeedForward, CrossAttention
from videogvt.models.transformer.t5 import (
    get_encoded_dim,
    DEFAULT_T5_NAME,
    TextEncoder,
)

# helpers


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def l2norm(t):
    return ops.normalize(t, dim=-1)


# classifier free guidance functions


def uniform(shape, min=0, max=1):
    return ops.zeros(shape).float().uniform_(0, 1)


@ms.jit
def prob_mask_like(shape, prob):
    if prob == 1:
        return ops.ones(shape, dtype=ms.bool_)
    elif prob == 0:
        return ops.zeros(shape, dtype=ms.bool_)
    else:
        return uniform(shape) < prob


# classes


# class LayerNorm(nn.Cell):
#     def __init__(self, dim):
#         super().__init__()
#         self.gamma = ms.Parameter(ops.ones(dim))
#         self.beta = ms.Parameter(ops.zeros(dim), requires_grad=False)
#         # self.register_buffer("beta", ops.zeros(dim))

#     def construct(self, x):
#         return ops.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


# class GEGLU(nn.Cell):
#     """https://arxiv.org/abs/2002.05202"""

#     def construct(self, x):
#         x, gate = x.chunk(2, dim=-1)
#         return gate * ops.gelu(x)


# def FeedForward(dim, mult=4):
#     """https://arxiv.org/abs/2110.09456"""

#     inner_dim = int(dim * mult * 2 / 3)
#     return nn.Sequential(
#         LayerNorm(dim),
#         nn.Dense(dim, inner_dim * 2, has_bias=False),
#         GEGLU(),
#         LayerNorm(inner_dim),
#         nn.Dense(inner_dim, dim, has_bias=False),
#     )


# class Attention(nn.Cell):
#     def __init__(
#         self,
#         dim,
#         dim_head=64,
#         heads=8,
#         cross_attend=False,
#         scale=8,
#         flash=True,
#         dropout=0.0,
#     ):
#         super().__init__()
#         self.scale = scale
#         self.heads = heads
#         inner_dim = dim_head * heads

#         self.cross_attend = cross_attend
#         self.norm = LayerNorm(dim)

#         self.attend = Attend(flash=flash, dropout=dropout, scale=scale)

#         self.null_kv = ms.Parameter(ops.randn(2, heads, 1, dim_head))

#         self.to_q = nn.Dense(dim, inner_dim, has_bias=False)
#         self.to_kv = nn.Dense(dim, inner_dim * 2, has_bias=False)

#         self.q_scale = ms.Parameter(ops.ones(dim_head))
#         self.k_scale = ms.Parameter(ops.ones(dim_head))

#         self.to_out = nn.Dense(inner_dim, dim, has_bias=False)

#     def construct(self, x, context=None, context_mask=None):
#         assert not (exists(context) ^ self.cross_attend)

#         n = x.shape[-2]
#         h, is_cross_attn = self.heads, exists(context)

#         x = self.norm(x)

#         kv_input = context if self.cross_attend else x

#         q, k, v = (self.to_q(x), *self.to_kv(kv_input).chunk(2, axis=-1))

#         q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

#         nk, nv = self.null_kv
#         nk, nv = map(lambda t: repeat(t, "h 1 d -> b h 1 d", b=x.shape[0]), (nk, nv))

#         k = ops.cat((nk, k), axis=-2)
#         v = ops.cat((nv, v), axis=-2)

#         q, k = map(l2norm, (q, k))
#         q = q * self.q_scale
#         k = k * self.k_scale

#         if exists(context_mask):
#             context_mask = repeat(context_mask, "b j -> b h i j", h=h, i=n)
#             context_mask = ops.pad(context_mask, (1, 0), value=True)

#         out = self.attend(q, k, v, mask=context_mask)

#         out = rearrange(out, "b h n d -> b n (h d)")
#         return self.to_out(out)


class TransformerBlocks(nn.Cell):
    def __init__(self, *, dim, depth, dim_head=64, heads=8, ff_mult=4, flash=True):
        super().__init__()
        self.layers = nn.CellList([])

        for _ in range(depth):
            self.layers.append(
                nn.CellList(
                    [
                        CrossAttention(
                            query_dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            enable_flash_attention=flash,
                        ),
                        CrossAttention(
                            query_dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            cross_attend=True,
                            enable_flash_attention=flash,
                        ),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        self.norm = nn.LayerNorm([dim], epsilon=1e-05)

    def construct(self, x, context=None, context_mask=None):
        for attn, cross_attn, ff in self.layers:
            x = attn(x) + x

            x = cross_attn(x, context=context, mask=context_mask) + x

            x = ff(x) + x

        return self.norm(x)


# transformer - it's all we need


class Transformer(nn.Cell):
    def __init__(
        self,
        num_tokens,
        dim,
        seq_len,
        dim_out=None,
        t5_name=DEFAULT_T5_NAME,
        self_cond=False,
        add_mask_id=False,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.mask_id = num_tokens if add_mask_id else None

        self.num_tokens = num_tokens
        self.token_emb = nn.Embedding(num_tokens + int(add_mask_id), dim)
        self.pos_emb = nn.Embedding(seq_len, dim)
        self.seq_len = seq_len

        self.transformer_blocks = TransformerBlocks(dim=dim, **kwargs)
        self.norm = nn.LayerNorm([dim], epsilon=1e-05)

        self.dim_out = default(dim_out, num_tokens)
        self.to_logits = nn.Dense(dim, self.dim_out, has_bias=False)

        # text conditioning
        self.text_encoder = TextEncoder(t5_name)

        text_embed_dim = get_encoded_dim(t5_name)

        self.text_embed_proj = (
            nn.Dense(text_embed_dim, dim, has_bias=False)
            if text_embed_dim != dim
            else nn.Identity()
        )

        # optional self conditioning

        self.self_cond = self_cond
        self.self_cond_to_init_embed = FeedForward(dim)

    def encode_text(self, texts):
        return self.text_encoder.encode(texts)

    def forward_with_cond_scale(
        self, *args, cond_scale=3.0, return_embed=False, **kwargs
    ):
        if cond_scale == 1:
            return self.construct(
                *args, return_embed=return_embed, cond_drop_prob=0.0, **kwargs
            )

        logits, embed = self.construct(
            *args, return_embed=True, cond_drop_prob=0.0, **kwargs
        )

        null_logits = self.construct(*args, cond_drop_prob=1.0, **kwargs)

        scaled_logits = null_logits + (logits - null_logits) * cond_scale

        if return_embed:
            return scaled_logits, embed

        return scaled_logits

    def forward_with_neg_prompt(
        self,
        *args,
        text_embed: ms.Tensor,
        neg_text_embed: ms.Tensor,
        cond_scale=3.0,
        return_embed=False,
        **kwargs
    ):
        neg_logits = self.construct(
            *args, neg_text_embed=neg_text_embed, cond_drop_prob=0.0, **kwargs
        )
        pos_logits, embed = self.construct(
            *args,
            return_embed=True,
            text_embed=text_embed,
            cond_drop_prob=0.0,
            **kwargs
        )

        logits = neg_logits + (pos_logits - neg_logits) * cond_scale

        if return_embed:
            return logits, embed

        return logits

    def construct(
        self,
        x,
        return_embed=False,
        return_logits=False,
        labels=None,
        ignore_index=0,
        self_cond_embed=None,
        cond_drop_prob=0.0,
        conditioning_token_ids: Optional[ms.Tensor] = None,
        texts: Optional[List[str]] = None,
        text_embeds: Optional[ms.Tensor] = None,
    ):
        b, n = x.shape
        assert not self.seq_len < n

        # prepare texts

        if text_embeds is None:
            text_embeds = self.text_encoder.encode(texts)

        context = self.text_embed_proj(text_embeds)

        context_mask = (text_embeds != 0).any(axis=-1)

        # classifier free guidance

        if cond_drop_prob > 0.0:
            mask = prob_mask_like(context_mask.shape, 1.0 - cond_drop_prob)
            context_mask = ops.logical_and(context_mask, mask)

        # concat conditioning video token ids if needed

        if exists(conditioning_token_ids):
            # conditioning_token_ids = rearrange(
            #     conditioning_token_ids, "b ... -> b (...)"
            # )
            conditioning_token_ids = conditioning_token_ids.reshape(
                conditioning_token_ids.shape[0], -1
            )
            cond_token_emb = self.token_emb(conditioning_token_ids)
            context = ops.cat((context, cond_token_emb), axis=-2)
            context_mask = ops.pad(
                context_mask, (0, conditioning_token_ids.shape[-1]), value=True
            )

        # embed tokens

        x = self.token_emb(x)
        x = x + self.pos_emb(ops.arange(n))

        if self.self_cond:
            if not exists(self_cond_embed):
                self_cond_embed = ops.zeros_like(x)
            x = x + self.self_cond_to_init_embed(self_cond_embed)

        embed = self.transformer_blocks(x, context=context, context_mask=context_mask)

        logits = self.to_logits(embed)

        if return_embed:
            return logits, embed

        if not exists(labels):
            return logits

        if self.dim_out == 1:
            # loss = ops.binary_cross_entropy_with_logits(
            #     rearrange(logits, "... 1 -> ..."), labels
            # )
            loss = ops.binary_cross_entropy_with_logits(logits.squeeze(-1), labels)
        else:
            # loss = ops.cross_entropy(
            #     rearrange(logits, "b n c -> b c n"), labels, ignore_index=ignore_index
            # )
            loss = ops.cross_entropy(
                logits.swapaxes(1, 2), labels, ignore_index=ignore_index
            )

        if not return_logits:
            return loss

        return loss, logits


# self critic wrapper


class SelfCritic(nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.to_pred = nn.Dense(net.dim, 1)

    def forward_with_cond_scale(self, x, *args, **kwargs):
        _, embeds = self.net.forward_with_cond_scale(
            x, *args, return_embed=True, **kwargs
        )
        return self.to_pred(embeds)

    def forward_with_neg_prompt(self, x, *args, **kwargs):
        _, embeds = self.net.forward_with_neg_prompt(
            x, *args, return_embed=True, **kwargs
        )
        return self.to_pred(embeds)

    def construct(self, x, *args, labels=None, **kwargs):
        _, embeds = self.net(x, *args, return_embed=True, **kwargs)
        logits = self.to_pred(embeds)

        if not exists(labels):
            return logits

        # logits = rearrange(logits, "... 1 -> ...")
        logits = logits.squeeze(-1)
        return ops.binary_cross_entropy_with_logits(logits, labels)


# specialized transformers


class MAGVITransformer(Transformer):
    def __init__(self, *args, **kwargs):
        assert "add_mask_id" not in kwargs
        super().__init__(*args, add_mask_id=True, **kwargs)


class TokenCritic(Transformer):
    def __init__(self, *args, **kwargs):
        assert "dim_out" not in kwargs
        super().__init__(*args, dim_out=1, **kwargs)
