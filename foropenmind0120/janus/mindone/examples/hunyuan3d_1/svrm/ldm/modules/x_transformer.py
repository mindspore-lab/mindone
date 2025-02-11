"""shout-out to https://github.com/lucidrains/x-transformers/tree/main/x_transformers"""
from collections import namedtuple
from functools import partial
from inspect import isfunction

import mindspore as ms
from mindspore import mint, nn, ops
from mindspore.common.initializer import Normal, initializer

from ..util import dtype_to_max

# constants

DEFAULT_DIM_HEAD = 64

Intermediates = namedtuple("Intermediates", ["pre_softmax_attn", "post_softmax_attn"])

LayerIntermediates = namedtuple("Intermediates", ["hiddens", "attn_intermediates"])


class AbsolutePositionalEmbedding(nn.Cell):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)
        self.init_()

    def init_(self):
        weight = initializer(Normal(sigma=0.02), shape=self.emb.weight.shape)
        self.emb.weight.set_data(weight)

    def construct(self, x):
        n = mint.arange(x.shape[1])
        return self.emb(n)[None, :, :]


class FixedPositionalEmbedding(nn.Cell):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (mint.arange(0, dim, 2).float() / dim))
        self.inv_freq = inv_freq

    def construct(self, x, seq_dim=1, offset=0):
        t = mint.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq) + offset
        sinusoid_inp = ops.einsum("i , j -> i j", t, self.inv_freq)
        emb = mint.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb[None, :, :]


# helpers


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def always(val):
    def inner(*args, **kwargs):
        return val

    return inner


def not_equals(val):
    def inner(x):
        return x != val

    return inner


def equals(val):
    def inner(x):
        return x == val

    return inner


def max_neg_value(tensor):
    return -dtype_to_max(tensor.dtype)


# keyword argument helpers


def pick_and_pop(keys, d):
    values = list(map(lambda key: d.pop(key), keys))
    return dict(zip(keys, values))


def group_dict_by_key(cond, d):
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)


def string_begins_with(prefix, str):
    return str.startswith(prefix)


def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)


def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix) :], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs


# classes
class Scale(nn.Cell):
    def __init__(self, value, fn):
        super().__init__()
        self.value = value
        self.fn = fn

    def construct(self, x, **kwargs):
        x, *rest = self.fn(x, **kwargs)
        return (x * self.value, *rest)


class Rezero(nn.Cell):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = ms.Parameter(mint.zeros(1))

    def construct(self, x, **kwargs):
        x, *rest = self.fn(x, **kwargs)
        return (x * self.g, *rest)


class ScaleNorm(nn.Cell):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = dim**-0.5
        self.eps = eps
        self.g = nn.Parameter(mint.ones(1))

    def construct(self, x):
        norm = ops.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class RMSNorm(nn.Cell):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim**-0.5
        self.eps = eps
        self.g = ms.Parameter(mint.ones(dim))

    def construct(self, x):
        norm = ops.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class Residual(nn.Cell):
    def construct(self, x, residual):
        return x + residual


class GRUGating(nn.Cell):
    def __init__(self, dim):
        super().__init__()
        self.gru = nn.GRUCell(dim, dim)

    def construct(self, x, residual):
        gated_output = self.gru(
            x.reshape(-1, x.shape[-1]), residual.reshape(-1, x.shape[-1])  # 'b n d -> (b n) d'  # 'b n d -> (b n) d'
        )

        return gated_output.reshape_as(x)


# feedforward


class GEGLU(nn.Cell):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Dense(dim_in, dim_out * 2)

    def construct(self, x):
        x, gate = mint.chunk(self.proj(x), 2, dim=-1)
        return x * mint.nn.functional.gelu(gate)


class FeedForward(nn.Cell):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.SequentialCell(nn.Dense(dim, inner_dim), nn.GELU(approximate=False))
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.SequentialCell(project_in, nn.Dropout(p=dropout), nn.Dense(inner_dim, dim_out))

    def construct(self, x):
        return self.net(x)


# attention.
class Attention(nn.Cell):
    def __init__(
        self,
        dim,
        dim_head=DEFAULT_DIM_HEAD,
        heads=8,
        causal=False,
        mask=None,
        talking_heads=False,
        sparse_topk=None,
        use_entmax15=False,
        num_mem_kv=0,
        dropout=0.0,
        on_attn=False,
    ):
        super().__init__()
        if use_entmax15:
            raise NotImplementedError("Check out entmax activation instead of softmax activation!")
        self.scale = dim_head**-0.5
        self.heads = heads
        self.causal = causal
        self.mask = mask

        inner_dim = dim_head * heads

        self.to_q = nn.Dense(dim, inner_dim, has_bias=False)
        self.to_k = nn.Dense(dim, inner_dim, has_bias=False)
        self.to_v = nn.Dense(dim, inner_dim, has_bias=False)
        self.dropout = nn.Dropout(p=dropout)

        # talking heads
        self.talking_heads = talking_heads
        if talking_heads:
            self.pre_softmax_proj = ms.Parameter(ops.randn((heads, heads)))
            self.post_softmax_proj = ms.Parameter(ops.randn((heads, heads)))

        # explicit topk sparse attention
        self.sparse_topk = sparse_topk

        # entmax
        # self.attn_fn = entmax15 if use_entmax15 else F.softmax
        self.attn_fn = mint.nn.functional.softmax

        # add memory key / values
        self.num_mem_kv = num_mem_kv
        if num_mem_kv > 0:
            self.mem_k = ms.Parameter(ops.randn((heads, num_mem_kv, dim_head)))
            self.mem_v = ms.Parameter(ops.randn((heads, num_mem_kv, dim_head)))

        # attention on attention
        self.attn_on_attn = on_attn
        self.to_out = nn.SequentialCell(nn.Dense(inner_dim, dim * 2), nn.GLU()) if on_attn else nn.Dense(inner_dim, dim)

    def construct(
        self, x, context=None, mask=None, context_mask=None, rel_pos=None, sinusoidal_emb=None, prev_attn=None, mem=None
    ):
        b, n, _, h, talking_heads = *x.shape, self.heads, self.talking_heads
        kv_input = default(context, x)

        q_input = x
        k_input = kv_input
        v_input = kv_input

        if exists(mem):
            k_input = mint.cat((mem, k_input), dim=-2)
            v_input = mint.cat((mem, v_input), dim=-2)

        if exists(sinusoidal_emb):
            # in shortformer, the query would start at a position offset depending on the past cached memory
            offset = k_input.shape[-2] - q_input.shape[-2]
            q_input = q_input + sinusoidal_emb(q_input, offset=offset)
            k_input = k_input + sinusoidal_emb(k_input)

        q = self.to_q(q_input)
        k = self.to_k(k_input)
        v = self.to_v(v_input)

        # 'b n (h d) -> b h n d', h=h
        q = q.reshape(q.shape[0], q.shape[1], h, -1)
        k = k.reshape(k.shape[0], k.shape[1], h, -1)
        v = v.reshape(v.shape[0], v.shape[1], h, -1)

        input_mask = None
        if any(map(exists, (mask, context_mask))):
            q_mask = default(mask, lambda: mint.ones((b, n)).bool())
            k_mask = q_mask if not exists(context) else context_mask
            k_mask = default(k_mask, lambda: mint.ones((b, k.shape[-2])).bool())
            q_mask = q_mask[:, None, :, None]  # 'b i -> b () i ()'
            k_mask = k_mask[:, None, None, :]  # 'b j -> b () () j'
            input_mask = q_mask * k_mask

        if self.num_mem_kv > 0:
            # 'h n d -> b h n d', b=b
            mem_k = self.mem_k[None, ...].tile((b, 1, 1, 1))
            mem_v = self.mem_v[None, ...].tile((b, 1, 1, 1))
            k = mint.cat((mem_k, k), dim=-2)
            v = mint.cat((mem_v, v), dim=-2)
            if exists(input_mask):
                input_mask = mint.nn.functional.pad(input_mask, (self.num_mem_kv, 0), value=1.0)

        dots = ops.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        mask_value = max_neg_value(dots)

        if exists(prev_attn):
            dots = dots + prev_attn

        pre_softmax_attn = dots

        if talking_heads:
            dots = ops.einsum("b h i j, h k -> b k i j", dots, self.pre_softmax_proj).contiguous()

        if exists(rel_pos):
            dots = rel_pos(dots)

        if exists(input_mask):
            dots = dots.masked_fill(~input_mask, mask_value)
            del input_mask

        if self.causal:
            i, j = dots.shape[-2:]
            r = mint.arange(i)
            # 'i -> () () i ()' < 'j -> () () () j' ????
            mask = r[None, None, :, None] < r[None, None, None, :]
            mask = mint.nn.functional.pad(mask, (j - i, 0), value=0)  # False
            dots = dots.masked_fill(mask, mask_value)
            del mask

        if exists(self.sparse_topk) and self.sparse_topk < dots.shape[-1]:
            top, _ = dots.topk(self.sparse_topk, dim=-1)
            vk = top[..., -1].unsqueeze(-1).expand_as(dots)
            mask = dots < vk
            dots = dots.masked_fill(mask, mask_value)
            del mask

        attn = self.attn_fn(dots, dim=-1)
        post_softmax_attn = attn

        attn = self.dropout(attn)

        if talking_heads:
            attn = ops.einsum("b h i j, h k -> b k i j", attn, self.post_softmax_proj).contiguous()

        out = ops.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = out.reshape(out.shape[0], out.shape[1], -1)  # 'b h n d -> b n (h d)'

        intermediates = Intermediates(pre_softmax_attn=pre_softmax_attn, post_softmax_attn=post_softmax_attn)

        return self.to_out(out), intermediates


class AttentionLayers(nn.Cell):
    def __init__(
        self,
        dim,
        depth,
        heads=8,
        causal=False,
        cross_attend=False,
        only_cross=False,
        use_scalenorm=False,
        use_rmsnorm=False,
        use_rezero=False,
        rel_pos_num_buckets=32,
        rel_pos_max_distance=128,
        position_infused_attn=False,
        custom_layers=None,
        sandwich_coef=None,
        par_ratio=None,
        residual_attn=False,
        cross_residual_attn=False,
        macaron=False,
        pre_norm=True,
        gate_residual=False,
        **kwargs,
    ):
        super().__init__()
        ff_kwargs, kwargs = groupby_prefix_and_trim("ff_", kwargs)
        attn_kwargs, _ = groupby_prefix_and_trim("attn_", kwargs)

        # dim_head = attn_kwargs.get("dim_head", DEFAULT_DIM_HEAD)

        self.dim = dim
        self.depth = depth
        self.layers = nn.CellList([])

        self.has_pos_emb = position_infused_attn
        self.pia_pos_emb = FixedPositionalEmbedding(dim) if position_infused_attn else None
        self.rotary_pos_emb = always(None)

        assert (
            rel_pos_num_buckets <= rel_pos_max_distance
        ), "number of relative position buckets must be less than the relative position max distance"
        self.rel_pos = None

        self.pre_norm = pre_norm

        self.residual_attn = residual_attn
        self.cross_residual_attn = cross_residual_attn

        norm_class = ScaleNorm if use_scalenorm else nn.LayerNorm
        norm_class = RMSNorm if use_rmsnorm else norm_class
        norm_fn = partial(norm_class, dim)

        norm_fn = nn.Identity if use_rezero else norm_fn
        branch_fn = Rezero if use_rezero else None

        if cross_attend and not only_cross:
            default_block = ("a", "c", "f")
        elif cross_attend and only_cross:
            default_block = ("c", "f")
        else:
            default_block = ("a", "f")

        if macaron:
            default_block = ("f",) + default_block

        if exists(custom_layers):
            layer_types = custom_layers
        elif exists(par_ratio):
            par_depth = depth * len(default_block)
            assert 1 < par_ratio <= par_depth, "par ratio out of range"
            default_block = tuple(filter(not_equals("f"), default_block))
            par_attn = par_depth // par_ratio
            depth_cut = par_depth * 2 // 3  # 2 / 3 attention layer cutoff suggested by PAR paper
            par_width = (depth_cut + depth_cut // par_attn) // par_attn
            assert len(default_block) <= par_width, "default block is too large for par_ratio"
            par_block = default_block + ("f",) * (par_width - len(default_block))
            par_head = par_block * par_attn
            layer_types = par_head + ("f",) * (par_depth - len(par_head))
        elif exists(sandwich_coef):
            assert sandwich_coef > 0 and sandwich_coef <= depth, "sandwich coefficient should be less than the depth"
            layer_types = ("a",) * sandwich_coef + default_block * (depth - sandwich_coef) + ("f",) * sandwich_coef
        else:
            layer_types = default_block * depth

        self.layer_types = layer_types
        self.num_attn_layers = len(list(filter(equals("a"), layer_types)))

        for layer_type in self.layer_types:
            if layer_type == "a":
                layer = Attention(dim, heads=heads, causal=causal, **attn_kwargs)  # TODO: susan comment: no causal!!!
            elif layer_type == "c":
                layer = Attention(dim, heads=heads, **attn_kwargs)
            elif layer_type == "f":
                layer = FeedForward(dim, **ff_kwargs)
                layer = layer if not macaron else Scale(0.5, layer)
            else:
                raise Exception(f"invalid layer type {layer_type}")

            if isinstance(layer, Attention) and exists(branch_fn):
                layer = branch_fn(layer)

            if gate_residual:
                residual_fn = GRUGating(dim)
            else:
                residual_fn = Residual()

            self.layers.append(nn.CellList([norm_fn(), layer, residual_fn]))

    def construct(self, x, context=None, mask=None, context_mask=None, mems=None, return_hiddens=False):
        hiddens = []
        intermediates = []
        prev_attn = None
        prev_cross_attn = None

        mems = mems.copy() if exists(mems) else [None] * self.num_attn_layers

        for ind, (layer_type, (norm, block, residual_fn)) in enumerate(zip(self.layer_types, self.layers)):
            is_last = ind == (len(self.layers) - 1)

            if layer_type == "a":
                hiddens.append(x)
                layer_mem = mems.pop(0)

            residual = x

            if self.pre_norm:
                x = norm(x)

            if layer_type == "a":
                out, inter = block(
                    x,
                    mask=mask,
                    sinusoidal_emb=self.pia_pos_emb,
                    rel_pos=self.rel_pos,
                    prev_attn=prev_attn,
                    mem=layer_mem,
                )
            elif layer_type == "c":
                out, inter = block(x, context=context, mask=mask, context_mask=context_mask, prev_attn=prev_cross_attn)
            elif layer_type == "f":
                out = block(x)

            x = residual_fn(out, residual)

            if layer_type in ("a", "c"):
                intermediates.append(inter)

            if layer_type == "a" and self.residual_attn:
                prev_attn = inter.pre_softmax_attn
            elif layer_type == "c" and self.cross_residual_attn:
                prev_cross_attn = inter.pre_softmax_attn

            if not self.pre_norm and not is_last:
                x = norm(x)

        if return_hiddens:
            intermediates = LayerIntermediates(hiddens=hiddens, attn_intermediates=intermediates)

            return x, intermediates

        return x


class Encoder(AttentionLayers):
    def __init__(self, **kwargs):
        assert "causal" not in kwargs, "cannot set causality on encoder"
        super().__init__(causal=False, **kwargs)


class TransformerWrapper(nn.Cell):
    def __init__(
        self,
        *,
        num_tokens,
        max_seq_len,
        attn_layers,
        emb_dim=None,
        max_mem_len=0.0,
        emb_dropout=0.0,
        num_memory_tokens=None,
        tie_embedding=False,
        use_pos_emb=True,
    ):
        super().__init__()
        assert isinstance(attn_layers, AttentionLayers), "attention layers must be one of Encoder or Decoder"

        dim = attn_layers.dim
        emb_dim = default(emb_dim, dim)

        self.max_seq_len = max_seq_len
        self.max_mem_len = max_mem_len
        self.num_tokens = num_tokens

        self.token_emb = nn.Embedding(num_tokens, emb_dim)
        self.pos_emb = (
            AbsolutePositionalEmbedding(emb_dim, max_seq_len)
            if (use_pos_emb and not attn_layers.has_pos_emb)
            else always(0)
        )
        self.emb_dropout = nn.Dropout(p=emb_dropout)

        self.project_emb = nn.Dense(emb_dim, dim) if emb_dim != dim else nn.Identity()
        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(dim)

        self.init_()

        self.to_logits = nn.Dense(dim, num_tokens) if not tie_embedding else lambda t: t @ self.token_emb.weight.t()

        # memory tokens (like [cls]) from Memory Transformers paper
        num_memory_tokens = default(num_memory_tokens, 0)
        self.num_memory_tokens = num_memory_tokens
        if num_memory_tokens > 0:
            self.memory_tokens = ms.Parameter(ops.randn((num_memory_tokens, dim)))

            # let funnel encoder know number of memory tokens, if specified
            if hasattr(attn_layers, "num_memory_tokens"):
                attn_layers.num_memory_tokens = num_memory_tokens

    def init_(self):
        weight = initializer(Normal(sigma=0.02), shape=self.emb.weight.shape)
        self.token_emb.weight.set_data(weight)

    def construct(
        self, x, return_embeddings=False, mask=None, return_mems=False, return_attn=False, mems=None, **kwargs
    ):
        b, _, num_mem = *x.shape, self.num_memory_tokens
        x = self.token_emb(x)
        x += self.pos_emb(x)
        x = self.emb_dropout(x)

        x = self.project_emb(x)

        if num_mem > 0:
            mem = self.memory_tokens.unsqeeze(0).tile((b, 1, 1))
            x = mint.cat((mem, x), dim=1)

            # auto-handle masking after appending memory tokens
            if exists(mask):
                mask = mint.nn.functional.pad(mask, (num_mem, 0), value=1.0)

        x, intermediates = self.attn_layers(x, mask=mask, mems=mems, return_hiddens=True, **kwargs)
        x = self.norm(x)

        mem, x = x[:, :num_mem], x[:, num_mem:]

        out = self.to_logits(x) if not return_embeddings else x

        if return_mems:
            hiddens = intermediates.hiddens
            new_mems = list(map(lambda pair: mint.cat(pair, dim=-2), zip(mems, hiddens))) if exists(mems) else hiddens
            new_mems = list(map(lambda t: t[..., -self.max_mem_len :, :], new_mems))
            return out, new_mems

        if return_attn:
            attn_maps = list(map(lambda t: t.post_softmax_attn, intermediates.attn_intermediates))
            return out, attn_maps

        return out
