import math
import os
from functools import partial
from typing import Any, Optional

import numpy as np

import mindspore as ms
from mindspore import nn, ops
from mindspore.common.initializer import initializer

from ..utils.pt2ms import load_pt_weights_in_model

# from .mha_flash import FlashAttentionBlock
from .rotary_embedding import RotaryEmbedding

__all__ = ["UNetSD_temporal"]

USE_TEMPORAL_TRANSFORMER = True
_USE_MEMORY_EFFICIENT_ATTENTION = int(os.environ.get("USE_MEMORY_EFFICIENT_ATTENTION", 0)) == 1
_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def sinusoidal_embedding(timesteps, dim):
    # check input
    half = dim // 2
    timesteps = timesteps.float()

    # compute sinusoidal embedding
    sinusoid = ops.outer(timesteps, ops.pow(10000, -ops.arange(half).to(timesteps.dtype).div(half)))
    x = ops.cat([ops.cos(sinusoid), ops.sin(sinusoid)], axis=1)
    if dim % 2 != 0:
        x = ops.cat([x, ops.zeros_like(x[:, :1])], axis=1)
    return x


def prob_mask_like(shape, prob):
    if prob == 1:
        return ops.ones(shape, dtype=ms.bool_)
    elif prob == 0:
        return ops.zeros(shape, dtype=ms.bool_)
    else:
        mask = ops.uniform(shape, ms.Tensor(0), ms.Tensor(1), dtype=ms.float32) < prob
        # avoid mask all, which will cause find_unused_parameters error
        if mask.all():
            mask[0] = False
        return mask


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Dense(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.get_parameters():
        p.set_data(initializer("zeros", p.shape, p.dtype))
    return module


class RelativePositionBias(nn.Cell):
    def __init__(self, heads=8, num_buckets=32, max_distance=128):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = ops.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = (
            max_exact
            + (ops.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)).long()
        )  # todo: cast to int64 may cause precision loss, we may have to use numpy to replace ops.log
        val_if_large = ops.minimum(val_if_large, ops.full_like(val_if_large, num_buckets - 1))

        ret += ops.where(is_small, n, val_if_large)
        return ret

    def construct(self, n):
        q_pos = ops.arange(n, dtype=ms.int64)
        k_pos = ops.arange(n, dtype=ms.int64)
        rel_pos = ops.expand_dims(k_pos, 0) - ops.expand_dims(q_pos, -1)
        rp_bucket = self._relative_position_bucket(
            rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance
        )
        values = self.relative_attention_bias(rp_bucket)
        return ops.transpose(values, (2, 0, 1))


class GroupNorm(nn.GroupNorm):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__(num_groups=num_groups, num_channels=num_channels, eps=eps, affine=affine)

    def construct(self, x):
        x_shape = x.shape
        if x.ndim >= 3:
            x = x.view(x_shape[0], x_shape[1], x_shape[2], -1)
        y = super().construct(x)
        return y.view(x_shape)


class GEGLU(nn.Cell):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Dense(dim_in, dim_out * 2)

    def construct(self, x):
        x, gate = self.proj(x).chunk(2, axis=-1)
        return x * ops.gelu(gate)


class DropPath(nn.Cell):
    r"""DropPath but without rescaling and supports optional all-zero and/or all-keep."""

    def __init__(self, p):
        super(DropPath, self).__init__()
        self.p = p

    def construct(self, *args, zero=None, keep=None):
        if not self.training:
            return args[0] if len(args) == 1 else args

        # params
        x = args[0]
        b = x.shape[0]
        n = (ops.rand(b) < self.p).sum()

        # non-zero and non-keep mask
        mask = x.new_ones(b, dtype=ms.bool_)
        if keep is not None:
            mask[keep] = False
        if zero is not None:
            mask[zero] = False

        # drop-path index
        index = ops.nonzero(mask).t()[0]  # special case for ops.nonzero, that the input is 1-d tensor
        index = index[ops.randperm(len(index))[:n]]
        if zero is not None:
            index = ops.cat([index, ops.nonzero(zero).t()[0]], axis=0)

        # drop-path multiplier
        multiplier = x.new_ones(b)
        multiplier[index] = 0.0
        output = tuple(u * self.broadcast(multiplier, u) for u in args)
        return output[0] if len(args) == 1 else output

    def broadcast(self, src, dst):
        assert src.shape[0] == dst.shape[0]
        shape = (dst.shape[0],) + (1,) * (dst.ndim - 1)
        return src.view(shape)


class FeedForward(nn.Cell):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.SequentialCell([nn.Dense(dim, inner_dim), nn.GELU()]) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.SequentialCell([project_in, nn.Dropout(p=dropout), nn.Dense(inner_dim, dim_out)])

    def construct(self, x):
        return self.net(x)


class MemoryEfficientCrossAttention(nn.Cell):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Dense(query_dim, inner_dim, has_bias=False)
        self.to_k = nn.Dense(context_dim, inner_dim, has_bias=False)
        self.to_v = nn.Dense(context_dim, inner_dim, has_bias=False)

        self.to_out = nn.SequentialCell([nn.Dense(inner_dim, query_dim, has_bias=True), nn.Dropout(p=dropout)])
        self.attention_op: Optional[Any] = None

    def construct(self, x, context=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        # use xformers.ops.memory_efficient_attention
        out = ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)


class CrossAttention(nn.Cell):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Dense(query_dim, inner_dim, has_bias=False)
        self.to_k = nn.Dense(context_dim, inner_dim, has_bias=False)
        self.to_v = nn.Dense(context_dim, inner_dim, has_bias=False)

        self.to_out = nn.SequentialCell([nn.Dense(inner_dim, query_dim, has_bias=True), nn.Dropout(p=dropout)])

    def construct(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        def rearrange_qkv(tensor):
            # b n (h d) -> b n h d -> b h n d -> (b h) n d
            tensor = ops.reshape(tensor, (*tensor.shape[:2], h, tensor.shape[2] // h))
            tensor = ops.transpose(tensor, (0, 2, 1, 3))
            tensor = ops.reshape(tensor, (-1, *tensor.shape[2:]))
            return tensor

        q, k, v = map(lambda t: rearrange_qkv(t), (q, k, v))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION == "fp32":
            q, k = q.float(), k.float()
        sim = ops.bmm(q, ops.transpose(k, (0, 2, 1))) * self.scale

        if exists(mask):
            b = mask.shape[0]
            mask = ops.reshape(mask, (b, -1))
            max_neg_value = -np.finfo(ms.dtype_to_nptype(sim.dtype)).max
            mask = ops.expand_dims(ops.tile(mask, (h, 1)), axis=1)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = ops.softmax(sim, axis=-1)

        out = ops.bmm(sim, v)
        # (b h) n d -> b h n d -> b n h d -> b n (h d)
        out = ops.reshape(out, (out.shape[0] // h, h, *out.shape[1:]))
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (*out.shape[:2], -1))
        return self.to_out(out)


class PreNormAttention(nn.Cell):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm((dim,))
        self.fn = fn

    def construct(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs) + x


class Attention(nn.Cell):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(axis=-1)
        self.to_qkv = nn.Dense(dim, inner_dim * 3, has_bias=False)

        self.to_out = (
            nn.SequentialCell(nn.Dense(inner_dim, dim), nn.Dropout(p=dropout)) if project_out else nn.Identity()
        )

    def construct(self, x):
        b, n, _, h = *x.shape, self.heads  # noqa
        qkv = self.to_qkv(x).chunk(3, axis=-1)

        def rearrange_qkv(tensor: ms.Tensor):
            # b n (h d) -> b n h d -> b h n d
            tensor = ops.reshape(tensor, (*tensor.shape[:2], h, tensor.shape[2] // h))
            tensor = ops.transpose(tensor, (0, 2, 1, 3))
            return tensor

        q, k, v = map(rearrange_qkv, qkv)

        dots = ops.bmm(q, k.transpose(0, 1, 3, 2)) * self.scale

        attn = self.attend(dots)

        out = ops.bmm(attn, v)
        # b h n d -> b n h d -> b n (h d)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (*out.shape[:2], -1))
        return self.to_out(out)


class TransformerV2(nn.Cell):
    def __init__(
        self,
        heads=8,
        dim=2048,
        dim_head_k=256,
        dim_head_v=256,
        dropout_atte=0.05,
        mlp_dim=2048,
        dropout_ffn=0.05,
        depth=1,
    ):
        super().__init__()
        layers = []
        self.depth = depth
        for _ in range(depth):
            layers.append(
                nn.CellList(
                    [
                        PreNormAttention(dim, Attention(dim, heads=heads, dim_head=dim_head_k, dropout=dropout_atte)),
                        FeedForward(dim, mlp_dim, dropout=dropout_ffn),
                    ]
                )
            )
        self.layers = nn.CellList(layers)

    def construct(self, x):
        for attn, ff in self.layers[:1]:
            x = attn(x)
            x = ff(x) + x
        if self.depth > 1:
            for attn, ff in self.layers[1:]:
                x = attn(x)
                x = ff(x) + x
        return x


class BasicTransformerBlock(nn.Cell):
    # ATTENTION_MODES = {
    #     "softmax": CrossAttention,  # vanilla attention
    #     "softmax-xformers": MemoryEfficientCrossAttention
    # }
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        use_checkpoint=True,
        disable_self_attn=False,
    ):
        super().__init__()
        # attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILABLE else "softmax"
        attn_cls = CrossAttention
        # attn_cls = MemoryEfficientCrossAttention
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None,
        )  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(
            query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )  # is self-attn if context is none
        self.norm1 = nn.LayerNorm((dim,))
        self.norm2 = nn.LayerNorm((dim,))
        self.norm3 = nn.LayerNorm((dim,))
        self.use_checkpoint = use_checkpoint

    def forward_(self, x, context=None):
        return ops.checkpoint(self._forward, (x, context), self.parameters(), self.use_checkpoint)

    def construct(self, x, context=None):
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Cell):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        disable_self_attn=False,
        use_linear=False,
        use_checkpoint=True,
    ):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0, has_bias=True)
        else:
            self.proj_in = nn.Dense(in_channels, inner_dim)

        self.transformer_blocks = nn.CellList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    disable_self_attn=disable_self_attn,
                    use_checkpoint=use_checkpoint,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0, has_bias=True)
            )
        else:
            self.proj_out = zero_module(nn.Dense(in_channels, inner_dim))
        self.use_linear = use_linear

    def construct(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        # b c h w -> b h w c -> b (h w) c
        x = ops.transpose(x, (0, 2, 3, 1))
        x = ops.reshape(x, (x.shape[0], -1, x.shape[3]))
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i])
        if self.use_linear:
            x = self.proj_out(x)
        # b (h w) c -> b c (h w) -> b c h w
        x = ops.transpose(x, (0, 2, 1))
        x = ops.reshape(x, (*x.shape[:2], h, w))
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


class TemporalTransformer(nn.Cell):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        disable_self_attn=False,
        use_linear=False,
        use_checkpoint=True,
        only_self_att=True,
        multiply_zero=False,
    ):
        super().__init__()
        self.multiply_zero = multiply_zero
        self.only_self_att = only_self_att
        self.use_adaptor = False
        self.frames = None  # todo: wtf frames
        if self.only_self_att:
            context_dim = None
        if not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        if not use_linear:
            self.proj_in = nn.Conv1d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0, has_bias=True)
        else:
            self.proj_in = nn.Dense(in_channels, inner_dim)
            if self.use_adaptor:
                self.adaptor_in = nn.Dense(self.frames, self.frames)  # todo: wtf frames

        self.transformer_blocks = nn.CellList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    use_checkpoint=use_checkpoint,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv1d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0, has_bias=True)
            )
        else:
            self.proj_out = zero_module(nn.Dense(in_channels, inner_dim))
            if self.use_adaptor:
                self.adaptor_out = nn.Dense(self.frames, self.frames)  # todo: wtf frames
        self.use_linear = use_linear

    def construct(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if self.only_self_att:
            context = None
        if not isinstance(context, list):
            context = [context]
        b, c, f, h, w = x.shape
        x_in = x
        x = self.norm(x)

        if not self.use_linear:
            # b c f h w -> b h w c f -> (b h w) c f
            x = ops.transpose(x, (0, 3, 4, 1, 2))
            x = ops.reshape(x, (-1, *x.shape[3:]))
            x = self.proj_in(x)
        # [16384, 16, 320]
        if self.use_linear:
            # (b f) c h w -> b f c h w -> b h w f c -> b (h w) f c
            x = ops.reshape(x, (x.shape[0] // self.frames, self.frames, *x.shape[1:]))  # todo: wtf frames
            x = ops.transpose(x, (0, 3, 4, 1, 2))
            x = ops.reshape(x, (x.shape[0], -1, *x.shape[3:]))
            x = self.proj_in(x)

        if self.only_self_att:
            x = ops.transpose(x, (0, 2, 1))
            for i, block in enumerate(self.transformer_blocks):
                x = block(x)
            # (b hw) f c -> b hw f c
            x = ops.reshape(x, (b, x.shape[0] // b, *x.shape[1:]))
        else:
            # (b hw) c f -> b hw f c
            x = ops.reshape(x, (b, x.shape[0] // b, *x.shape[1:]))
            x = ops.transpose(x, (0, 1, 3, 2))
            for i, block in enumerate(self.transformer_blocks):
                # context[i] = repeat(context[i], '(b f) l con -> b (f r) l con', r=(h*w)//self.frames, f=self.frames)
                # (b f) l con -> b f l con
                context[i] = ops.reshape(
                    context[i], (context[i].shape[0] // self.frames, self.frames, *context[i].shape[1:])
                )  # todo: wtf frames
                # calculate each batch one by one (since number in shape could not greater than 65,535 for some package)
                for j in range(b):
                    context_i_j = ops.tile(context[i][j], ((h * w) // self.frames, 1, 1))  # todo: wtf frames
                    x[j] = block(x[j], context=context_i_j)

        if self.use_linear:
            x = self.proj_out(x)
            # b (h w) f c -> b h w f c -> b f c h w
            x = ops.reshape(x, (x.shape[0], h, w, *x.shape[2:]))
            x = ops.transpose(x, (0, 3, 4, 1, 2))
        if not self.use_linear:
            # b hw f c -> (b hw) f c -> (b hw) c f
            x = ops.reshape(x, (-1, *x.shape[2:]))
            x = ops.transpose(x, (0, 2, 1))
            x = self.proj_out(x)
            # (b h w) c f -> b h w c f -> b c f h w
            x = ops.reshape(x, (b, h, w, *x.shape[1:]))
            x = ops.transpose(x, (0, 3, 4, 1, 2))

        if self.multiply_zero:
            x = 0.0 * x + x_in
        else:
            x = x + x_in
        return x


class TemporalAttentionBlock(nn.Cell):
    def __init__(self, dim, heads=4, dim_head=32, rotary_emb=None, use_image_dataset=False, use_sim_mask=False):
        super().__init__()
        # consider num_heads first, as pos_bias needs fixed num_heads
        # heads = dim // dim_head if dim_head else heads
        dim_head = dim // heads
        assert heads * dim_head == dim
        self.use_image_dataset = use_image_dataset
        self.use_sim_mask = use_sim_mask

        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = GroupNorm(32, dim)
        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Dense(dim, hidden_dim * 3)  # , bias = False)
        self.to_out = nn.Dense(hidden_dim, dim)  # , bias = False)
        # nn.init.zeros_(self.to_out.weight)
        # nn.init.zeros_(self.to_out.bias)

    def construct(self, x, pos_bias=None, focus_present_mask=None, video_mask=None):
        identity = x
        n, height = x.shape[2], x.shape[-2]

        x = self.norm(x)
        # b c f h w -> b c f (h w) -> b (h w) f c
        x = ops.reshape(x, (*x.shape[:3], -1))
        x = ops.transpose(x, (0, 3, 2, 1))

        qkv = self.to_qkv(x).chunk(3, axis=-1)

        if exists(focus_present_mask) and focus_present_mask.all():
            # if all batch samples are focusing on present
            # it would be equivalent to passing that token's values （v=qkv[-1]） through to the output
            values = qkv[-1]
            out = self.to_out(values)
            # b (h w) f c -> b h w f c -> b c f h w
            out = ops.reshape(out, (out.shape[0], height, out.shape[1] // height, *out.shape[2:]))
            out = ops.transpose(out, (0, 4, 3, 1, 2))

            return out + identity

        # split out heads
        # ... n (h d) -> ... n h d -> ... h n d
        q, k, v = qkv[0], qkv[1], qkv[2]
        permute_idx = tuple(range(q.ndim - 3)) + (q.ndim - 2, q.ndim - 3, q.ndim - 1)
        q = ops.reshape(q, (*q.shape[:-1], self.heads, q.shape[-1] // self.heads))
        q = ops.transpose(q, permute_idx)
        k = ops.reshape(k, (*k.shape[:-1], self.heads, k.shape[-1] // self.heads))
        k = ops.transpose(k, permute_idx)
        v = ops.reshape(v, (*v.shape[:-1], self.heads, v.shape[-1] // self.heads))
        v = ops.transpose(v, permute_idx)

        # scale
        q = q * self.scale

        # rotate positions into queries and keys for time attention
        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # similarity
        # shape [b (hw) h n n], n=f
        sim = ops.bmm(q, ops.transpose(k, (0, 1, 2, 4, 3)))

        # relative positional bias
        if exists(pos_bias):
            sim = sim + pos_bias

        if focus_present_mask is None and video_mask is not None:
            # video_mask: [B, n]
            mask = video_mask[:, None, :] * video_mask[:, :, None]  # [b,n,n]
            mask = mask.unsqueeze(1).unsqueeze(1)  # [b,1,1,n,n]
            sim = sim.masked_fill(~mask, -np.finfo(ms.dtype_to_nptype(sim.dtype)).max)
        elif exists(focus_present_mask) and not (~focus_present_mask).all():
            attend_all_mask = ops.ones((n, n), dtype=ms.bool_)
            attend_self_mask = ops.eye(n, dtype=ms.bool_)

            mask = ops.where(
                ops.reshape(focus_present_mask, (-1, 1, 1, 1, 1)),
                ops.reshape(attend_self_mask, (1, 1, 1, *attend_self_mask.shape)),
                ops.reshape(attend_all_mask, (1, 1, 1, *attend_all_mask.shape)),
            )

            sim = sim.masked_fill(~mask, -np.finfo(ms.dtype_to_nptype(sim.dtype)).max)

        if self.use_sim_mask:
            sim_mask = ops.tril(ops.ones((n, n), dtype=ms.bool_), diagonal=0)
            sim = sim.masked_fill(~sim_mask, -np.finfo(ms.dtype_to_nptype(sim.dtype)).max)

        # numerical stability
        sim = sim - sim.amax(axis=-1, keepdims=True)
        attn = sim.softmax(axis=-1)

        # aggregate values
        out = ops.bmm(attn, v)
        # ... h n d -> ... n h d -> ... n (h d)
        permute_idx = tuple(range(out.ndim - 3)) + (out.ndim - 2, out.ndim - 3, out.ndim - 1)
        out = ops.transpose(out, permute_idx)
        out = ops.reshape(out, (*out.shape[:-2], -1))
        out = self.to_out(out)

        # b (h w) f c -> b h w f c -> b c f h w
        out = ops.reshape(out, (out.shape[0], height, out.shape[1] // height, *out.shape[2:]))
        out = ops.transpose(out, (0, 4, 3, 1, 2))

        if self.use_image_dataset:
            out = identity + 0 * out
        else:
            out = identity + out
        return out


class TemporalAttentionMultiBlock(nn.Cell):
    def __init__(
        self,
        dim,
        heads=4,
        dim_head=32,
        rotary_emb=None,
        use_image_dataset=False,
        use_sim_mask=False,
        temporal_attn_times=1,
    ):
        super().__init__()
        self.att_layers = nn.CellList(
            [
                TemporalAttentionBlock(dim, heads, dim_head, rotary_emb, use_image_dataset, use_sim_mask)
                for _ in range(temporal_attn_times)
            ]
        )

    def construct(self, x, pos_bias=None, focus_present_mask=None, video_mask=None):
        for layer in self.att_layers:
            x = layer(x, pos_bias, focus_present_mask, video_mask)
        return x


class TemporalConvBlockV0(nn.Cell):
    def __init__(self, in_dim, out_dim=None, dropout=0.0, use_image_dataset=False):
        super(TemporalConvBlockV0, self).__init__()
        if out_dim is None:
            out_dim = in_dim  # int(1.5*in_dim)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_image_dataset = use_image_dataset

        # conv layers
        self.conv = nn.SequentialCell(
            GroupNorm(32, out_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), pad_mode="pad", padding=(1, 1, 0, 0, 0, 0), has_bias=True),
        )

        # zero out the last layer params,so the conv block is identity
        self.conv[-1].weight.set_data(initializer("zeros", self.conv[-1].weight.shape, self.conv[-1].weight.dtype))
        self.conv[-1].bias.set_data(initializer("zeros", self.conv[-1].bias.shape, self.conv[-1].bias.dtype))

    def construct(self, x):
        identity = x
        x = self.conv(x)
        if self.use_image_dataset:
            x = identity + 0 * x
        else:
            x = identity + x
        return x


class TemporalConvBlockV1(nn.Cell):
    def __init__(self, in_dim, out_dim=None, dropout=0.0, use_image_dataset=False):
        super(TemporalConvBlockV1, self).__init__()
        if out_dim is None:
            out_dim = in_dim  # int(1.5*in_dim)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_image_dataset = use_image_dataset

        # conv layers
        self.conv1 = nn.SequentialCell(
            GroupNorm(32, in_dim),
            nn.SiLU(),
            nn.Conv3d(in_dim, out_dim, (3, 1, 1), pad_mode="pad", padding=(1, 1, 0, 0, 0, 0), has_bias=True),
        )
        self.conv2 = nn.SequentialCell(
            GroupNorm(32, out_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), pad_mode="pad", padding=(1, 1, 0, 0, 0, 0), has_bias=True),
        )

        # zero out the last layer params,so the conv block is identity
        self.conv2[-1].weight.set_data(initializer("zeros", self.conv2[-1].weight.shape, self.conv2[-1].weight.dtype))
        self.conv2[-1].bias.set_data(initializer("zeros", self.conv2[-1].bias.shape, self.conv2[-1].bias.dtype))

    def construct(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.use_image_dataset:
            x = identity + 0 * x
        else:
            x = identity + x
        return x


class TemporalConvBlockV2(nn.Cell):
    def __init__(self, in_dim, out_dim=None, dropout=0.0, use_image_dataset=False):
        super(TemporalConvBlockV2, self).__init__()
        if out_dim is None:
            out_dim = in_dim  # int(1.5*in_dim)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_image_dataset = use_image_dataset

        # conv layers
        self.conv1 = nn.SequentialCell(
            GroupNorm(32, in_dim),
            nn.SiLU(),
            nn.Conv3d(in_dim, out_dim, (3, 1, 1), pad_mode="pad", padding=(1, 1, 0, 0, 0, 0), has_bias=True),
        )
        self.conv2 = nn.SequentialCell(
            GroupNorm(32, out_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), pad_mode="pad", padding=(1, 1, 0, 0, 0, 0), has_bias=True),
        )
        self.conv3 = nn.SequentialCell(
            GroupNorm(32, out_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), pad_mode="pad", padding=(1, 1, 0, 0, 0, 0), has_bias=True),
        )
        self.conv4 = nn.SequentialCell(
            GroupNorm(32, out_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), pad_mode="pad", padding=(1, 1, 0, 0, 0, 0), has_bias=True),
        )

        # zero out the last layer params,so the conv block is identity
        self.conv4[-1].weight.set_data(initializer("zeros", self.conv4[-1].weight.shape, self.conv4[-1].weight.dtype))
        self.conv4[-1].bias.set_data(initializer("zeros", self.conv4[-1].bias.shape, self.conv4[-1].bias.dtype))

    def construct(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        if self.use_image_dataset:
            x = identity + 0.0 * x
        else:
            x = identity + x
        return x


class Upsample(nn.Cell):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = nn.Conv2d(self.channels, self.out_channels, 3, pad_mode="pad", padding=padding, has_bias=True)

    def construct(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = ops.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        else:
            x = ops.interpolate(x, (x.shape[2] * 2, x.shape[3] * 2), mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Cell):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = nn.Conv2d(
                self.channels, self.out_channels, 3, stride=stride, pad_mode="pad", padding=padding, has_bias=True
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def construct(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class Resample(nn.Cell):
    def __init__(self, in_dim, out_dim, mode):
        assert mode in ["none", "upsample", "downsample"]
        super(Resample, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mode = mode

    def construct(self, x, reference=None):
        if self.mode == "upsample":
            assert reference is not None
            x = ops.interpolate(x, size=reference.shape[-2:], mode="nearest")
        elif self.mode == "downsample":
            x = ops.adaptive_avg_pool2d(x, output_size=tuple(u // 2 for u in x.shape[-2:]))
        return x


class ResidualBlock(nn.Cell):
    def __init__(self, in_dim, embed_dim, out_dim, use_scale_shift_norm=True, mode="none", dropout=0.0):
        super(ResidualBlock, self).__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.use_scale_shift_norm = use_scale_shift_norm
        self.mode = mode

        # layers
        self.layer1 = nn.SequentialCell(
            GroupNorm(32, in_dim), nn.SiLU(), nn.Conv2d(in_dim, out_dim, 3, pad_mode="pad", padding=1, has_bias=True)
        )
        self.resample = Resample(in_dim, in_dim, mode)
        self.embedding = nn.SequentialCell(
            nn.SiLU(), nn.Dense(embed_dim, out_dim * 2 if use_scale_shift_norm else out_dim)
        )
        self.layer2 = nn.SequentialCell(
            GroupNorm(32, out_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_dim, out_dim, 3, pad_mode="pad", padding=1, has_bias=True),
        )
        self.shortcut = nn.Identity() if in_dim == out_dim else nn.Conv2d(in_dim, out_dim, 1, has_bias=True)

        # zero out the last layer params
        self.layer2[-1].weight.set_data(
            initializer("zeros", self.layer2[-1].weight.shape, self.layer2[-1].weight.dtype)
        )

    def construct(self, x, e, reference=None):
        identity = self.resample(x, reference)
        x = self.layer1[-1](self.resample(self.layer1[:-1](x), reference))
        e = self.embedding(e).unsqueeze(-1).unsqueeze(-1).astype(x.dtype)
        if self.use_scale_shift_norm:
            scale, shift = e.chunk(2, axis=1)
            x = self.layer2[0](x) * (1 + scale) + shift
            x = self.layer2[1:](x)
        else:
            x = x + e
            x = self.layer2(x)
        x = x + self.shortcut(identity)
        return x


class ResBlock(nn.Cell):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        up=False,
        down=False,
        use_temporal_conv=True,
        use_image_dataset=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_temporal_conv = use_temporal_conv

        self.in_layers = nn.SequentialCell(
            GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, pad_mode="pad", padding=1, has_bias=True),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.SequentialCell(
            nn.SiLU(),
            nn.Dense(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.SequentialCell(
            GroupNorm(32, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, pad_mode="pad", padding=1, has_bias=True)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, pad_mode="pad", padding=1, has_bias=True
            )
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1, has_bias=True)

        if self.use_temporal_conv:
            self.temporal_conv = TemporalConvBlockV2(
                self.out_channels, self.out_channels, dropout=0.1, use_image_dataset=use_image_dataset
            )
            # self.temporal_conv_2 = TemporalConvBlockV1(
            #     self.out_channels, self.out_channels, dropout=0.1, use_image_dataset=use_image_dataset
            # )

    def construct(self, x, emb, batch_size):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).astype(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = ops.chunk(emb_out, 2, axis=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        h = self.skip_connection(x) + h

        if self.use_temporal_conv:
            # (b f) c h w -> b f c h w -> b c f h w
            h = ops.reshape(h, (batch_size, h.shape[0] // batch_size, *h.shape[1:]))
            h = ops.transpose(h, (0, 2, 1, 3, 4))
            h = self.temporal_conv(h)
            # h = self.temporal_conv_2(h)
            # 'b c f h w -> b f c h w -> (b f) c h w
            h = ops.transpose(h, (0, 2, 1, 3, 4))
            h = ops.reshape(h, (-1, *h.shape[2:]))
        return h


class UNetSD_temporal(nn.Cell):
    def __init__(
        self,
        cfg,
        in_dim=7,
        dim=512,
        y_dim=512,
        context_dim=512,
        hist_dim=156,
        concat_dim=8,
        out_dim=6,
        dim_mult=[1, 2, 3, 4],
        num_heads=None,
        head_dim=64,
        num_res_blocks=3,
        attn_scales=[1 / 2, 1 / 4, 1 / 8],
        use_scale_shift_norm=True,
        dropout=0.1,
        temporal_attn_times=1,
        temporal_attention=True,
        use_checkpoint=False,
        use_image_dataset=False,
        use_fps_condition=False,
        use_sim_mask=False,
        misc_dropout=0.5,
        training=True,
        inpainting=True,
        video_compositions=["text", "mask"],
        p_all_zero=0.1,
        p_all_keep=0.1,
        zero_y=None,
        black_image_feature=None,
    ):
        embed_dim = dim * 4
        num_heads = num_heads if num_heads else dim // 32
        super(UNetSD_temporal, self).__init__()
        self.zero_y = zero_y
        self.black_image_feature = black_image_feature
        self.cfg = cfg
        self.in_dim = in_dim
        self.dim = dim
        self.y_dim = y_dim
        self.context_dim = context_dim
        self.hist_dim = hist_dim
        self.concat_dim = concat_dim
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.dim_mult = dim_mult
        # for temporal attention
        self.num_heads = num_heads
        # for spatial attention
        self.head_dim = head_dim
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.use_scale_shift_norm = use_scale_shift_norm
        self.temporal_attn_times = temporal_attn_times
        self.temporal_attention = temporal_attention
        self.use_checkpoint = use_checkpoint
        self.use_image_dataset = use_image_dataset
        self.use_fps_condition = use_fps_condition
        self.use_sim_mask = use_sim_mask
        self.training = training
        self.inpainting = inpainting
        self.video_compositions = video_compositions
        self.misc_dropout = misc_dropout
        self.p_all_zero = p_all_zero
        self.p_all_keep = p_all_keep

        use_linear_in_temporal = False
        transformer_depth = 1
        disabled_sa = False
        # params
        enc_dims = [dim * u for u in [1] + dim_mult]
        dec_dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        shortcut_dims = []
        scale = 1.0
        if hasattr(cfg, "adapter_transformer_layers") and cfg.adapter_transformer_layers:
            adapter_transformer_layers = cfg.adapter_transformer_layers
        else:
            adapter_transformer_layers = 1

        # embeddings
        self.time_embed = nn.SequentialCell(nn.Dense(dim, embed_dim), nn.SiLU(), nn.Dense(embed_dim, embed_dim))
        self.pre_image_condition = nn.SequentialCell(nn.Dense(1024, 1024), nn.SiLU(), nn.Dense(1024, 1024))

        # depth embedding
        if "depthmap" in self.video_compositions:
            self.depth_embedding = nn.SequentialCell(
                nn.Conv2d(1, concat_dim * 4, 3, pad_mode="pad", padding=1, has_bias=True),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((128, 128)),
                nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=2, pad_mode="pad", padding=1, has_bias=True),
                nn.SiLU(),
                nn.Conv2d(concat_dim * 4, concat_dim, 3, stride=2, pad_mode="pad", padding=1, has_bias=True),
            )
            self.depth_embedding_after = TransformerV2(
                heads=2,
                dim=concat_dim,
                dim_head_k=concat_dim,
                dim_head_v=concat_dim,
                dropout_atte=0.05,
                mlp_dim=concat_dim,
                dropout_ffn=0.05,
                depth=adapter_transformer_layers,
            )

        if "motion" in self.video_compositions:
            self.motion_embedding = nn.SequentialCell(
                nn.Conv2d(2, concat_dim * 4, 3, pad_mode="pad", padding=1, has_bias=True),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((128, 128)),
                nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=2, pad_mode="pad", padding=1, has_bias=True),
                nn.SiLU(),
                nn.Conv2d(concat_dim * 4, concat_dim, 3, stride=2, pad_mode="pad", padding=1, has_bias=True),
            )
            self.motion_embedding_after = TransformerV2(
                heads=2,
                dim=concat_dim,
                dim_head_k=concat_dim,
                dim_head_v=concat_dim,
                dropout_atte=0.05,
                mlp_dim=concat_dim,
                dropout_ffn=0.05,
                depth=adapter_transformer_layers,
            )

        # canny embedding
        if "canny" in self.video_compositions:
            self.canny_embedding = nn.SequentialCell(
                nn.Conv2d(1, concat_dim * 4, 3, pad_mode="pad", padding=1, has_bias=True),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((128, 128)),
                nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=2, pad_mode="pad", padding=1, has_bias=True),
                nn.SiLU(),
                nn.Conv2d(concat_dim * 4, concat_dim, 3, stride=2, pad_mode="pad", padding=1, has_bias=True),
            )
            self.canny_embedding_after = TransformerV2(
                heads=2,
                dim=concat_dim,
                dim_head_k=concat_dim,
                dim_head_v=concat_dim,
                dropout_atte=0.05,
                mlp_dim=concat_dim,
                dropout_ffn=0.05,
                depth=adapter_transformer_layers,
            )

        # masked-image embedding
        if "mask" in self.video_compositions:
            self.masked_embedding = (
                nn.SequentialCell(
                    nn.Conv2d(4, concat_dim * 4, 3, pad_mode="pad", padding=1, has_bias=True),
                    nn.SiLU(),
                    nn.AdaptiveAvgPool2d((128, 128)),
                    nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=2, pad_mode="pad", padding=1, has_bias=True),
                    nn.SiLU(),
                    nn.Conv2d(concat_dim * 4, concat_dim, 3, stride=2, pad_mode="pad", padding=1, has_bias=True),
                )
                if inpainting
                else None
            )
            self.mask_embedding_after = TransformerV2(
                heads=2,
                dim=concat_dim,
                dim_head_k=concat_dim,
                dim_head_v=concat_dim,
                dropout_atte=0.05,
                mlp_dim=concat_dim,
                dropout_ffn=0.05,
                depth=adapter_transformer_layers,
            )

        # sketch embedding
        if "sketch" in self.video_compositions:
            self.sketch_embedding = nn.SequentialCell(
                nn.Conv2d(1, concat_dim * 4, 3, pad_mode="pad", padding=1, has_bias=True),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((128, 128)),
                nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=2, pad_mode="pad", padding=1, has_bias=True),
                nn.SiLU(),
                nn.Conv2d(concat_dim * 4, concat_dim, 3, stride=2, pad_mode="pad", padding=1, has_bias=True),
            )
            self.sketch_embedding_after = TransformerV2(
                heads=2,
                dim=concat_dim,
                dim_head_k=concat_dim,
                dim_head_v=concat_dim,
                dropout_atte=0.05,
                mlp_dim=concat_dim,
                dropout_ffn=0.05,
                depth=adapter_transformer_layers,
            )

        if "single_sketch" in self.video_compositions:
            self.single_sketch_embedding = nn.SequentialCell(
                nn.Conv2d(1, concat_dim * 4, 3, pad_mode="pad", padding=1, has_bias=True),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((128, 128)),
                nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=2, pad_mode="pad", padding=1, has_bias=True),
                nn.SiLU(),
                nn.Conv2d(concat_dim * 4, concat_dim, 3, stride=2, pad_mode="pad", padding=1, has_bias=True),
            )
            self.single_sketch_embedding_after = TransformerV2(
                heads=2,
                dim=concat_dim,
                dim_head_k=concat_dim,
                dim_head_v=concat_dim,
                dropout_atte=0.05,
                mlp_dim=concat_dim,
                dropout_ffn=0.05,
                depth=adapter_transformer_layers,
            )

        if "local_image" in self.video_compositions:
            self.local_image_embedding = nn.SequentialCell(
                nn.Conv2d(3, concat_dim * 4, 3, pad_mode="pad", padding=1, has_bias=True),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((128, 128)),
                nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=2, pad_mode="pad", padding=1, has_bias=True),
                nn.SiLU(),
                nn.Conv2d(concat_dim * 4, concat_dim, 3, stride=2, pad_mode="pad", padding=1, has_bias=True),
            )
            self.local_image_embedding_after = TransformerV2(
                heads=2,
                dim=concat_dim,
                dim_head_k=concat_dim,
                dim_head_v=concat_dim,
                dropout_atte=0.05,
                mlp_dim=concat_dim,
                dropout_ffn=0.05,
                depth=adapter_transformer_layers,
            )

        # Condition Dropout
        self.misc_dropout = DropPath(misc_dropout)

        if temporal_attention and not USE_TEMPORAL_TRANSFORMER:
            self.rotary_emb = RotaryEmbedding(min(32, head_dim))
            self.time_rel_pos_bias = RelativePositionBias(
                heads=num_heads, max_distance=32
            )  # realistically will not be able to generate that many frames of video... yet

        if self.use_fps_condition:
            self.fps_embedding = nn.SequentialCell(nn.Dense(dim, embed_dim), nn.SiLU(), nn.Dense(embed_dim, embed_dim))
            self.fps_embedding[-1].weight.set_data(
                initializer("zeros", self.fps_embedding[-1].weight.shape, self.fps_embedding[-1].weight.dtype)
            )
            self.fps_embedding[-1].bias.set_data(
                initializer("zeros", self.fps_embedding[-1].bias.shape, self.fps_embedding[-1].bias.dtype)
            )

        # encoder
        input_blocks = []
        # init_block = [nn.Conv2d(self.in_dim + concat_dim, dim, 3, padding=1)]
        if cfg.resume:
            self.pre_image = nn.SequentialCell()
            init_block = [nn.Conv2d(self.in_dim + concat_dim, dim, 3, pad_mode="pad", padding=1, has_bias=True)]
        else:
            self.pre_image = nn.SequentialCell(
                nn.Conv2d(self.in_dim + concat_dim, self.in_dim, 1, padding=0, has_bias=True)
            )
            init_block = [nn.Conv2d(self.in_dim, dim, 3, pad_mode="pad", padding=1, has_bias=True)]

        # need an initial temporal attention?
        if temporal_attention:
            if USE_TEMPORAL_TRANSFORMER:
                init_block.append(
                    TemporalTransformer(
                        dim,
                        num_heads,
                        head_dim,
                        depth=transformer_depth,
                        context_dim=context_dim,
                        disable_self_attn=disabled_sa,
                        use_linear=use_linear_in_temporal,
                        multiply_zero=use_image_dataset,
                    )
                )
            else:
                init_block.append(
                    TemporalAttentionMultiBlock(
                        dim,
                        num_heads,
                        head_dim,
                        rotary_emb=self.rotary_emb,
                        temporal_attn_times=temporal_attn_times,
                        use_image_dataset=use_image_dataset,
                    )
                )
        # elif temporal_conv:
        #
        input_blocks.append(nn.CellList(init_block))
        shortcut_dims.append(dim)
        for i, (in_dim, out_dim) in enumerate(zip(enc_dims[:-1], enc_dims[1:])):
            for j in range(num_res_blocks):
                # residual (+attention) blocks
                block = [
                    ResBlock(
                        in_dim,
                        embed_dim,
                        dropout,
                        out_channels=out_dim,
                        use_scale_shift_norm=False,
                        use_image_dataset=use_image_dataset,
                    )
                ]
                if scale in attn_scales:
                    block.append(
                        SpatialTransformer(
                            out_dim,
                            out_dim // head_dim,
                            head_dim,
                            depth=1,
                            context_dim=self.context_dim,
                            disable_self_attn=False,
                            use_linear=True,
                        )
                    )
                    if self.temporal_attention:
                        if USE_TEMPORAL_TRANSFORMER:
                            block.append(
                                TemporalTransformer(
                                    out_dim,
                                    out_dim // head_dim,
                                    head_dim,
                                    depth=transformer_depth,
                                    context_dim=context_dim,
                                    disable_self_attn=disabled_sa,
                                    use_linear=use_linear_in_temporal,
                                    multiply_zero=use_image_dataset,
                                )
                            )
                        else:
                            block.append(
                                TemporalAttentionMultiBlock(
                                    out_dim,
                                    num_heads,
                                    head_dim,
                                    rotary_emb=self.rotary_emb,
                                    use_image_dataset=use_image_dataset,
                                    use_sim_mask=use_sim_mask,
                                    temporal_attn_times=temporal_attn_times,
                                )
                            )
                in_dim = out_dim
                input_blocks.append(nn.CellList(block))
                shortcut_dims.append(out_dim)

                # downsample
                if i != len(dim_mult) - 1 and j == num_res_blocks - 1:
                    downsample = Downsample(out_dim, True, dims=2, out_channels=out_dim)
                    shortcut_dims.append(out_dim)
                    scale /= 2.0
                    # block.append(TemporalConvBlockV1(out_dim,dropout=dropout,use_image_dataset=use_image_dataset))
                    input_blocks.append(downsample)
        self.input_blocks = nn.SequentialCell(input_blocks)

        # middle
        middle_block = [
            ResBlock(
                out_dim,
                embed_dim,
                dropout,
                use_scale_shift_norm=False,
                use_image_dataset=use_image_dataset,
            ),
            SpatialTransformer(
                out_dim,
                out_dim // head_dim,
                head_dim,
                depth=1,
                context_dim=self.context_dim,
                disable_self_attn=False,
                use_linear=True,
            ),
        ]

        if self.temporal_attention:
            if USE_TEMPORAL_TRANSFORMER:
                middle_block.append(
                    TemporalTransformer(
                        out_dim,
                        out_dim // head_dim,
                        head_dim,
                        depth=transformer_depth,
                        context_dim=context_dim,
                        disable_self_attn=disabled_sa,
                        use_linear=use_linear_in_temporal,
                        multiply_zero=use_image_dataset,
                    )
                )
            else:
                middle_block.append(
                    TemporalAttentionMultiBlock(
                        out_dim,
                        num_heads,
                        head_dim,
                        rotary_emb=self.rotary_emb,
                        use_image_dataset=use_image_dataset,
                        use_sim_mask=use_sim_mask,
                        temporal_attn_times=temporal_attn_times,
                    )
                )

        # self.middle.append(ResidualBlock(out_dim, embed_dim, out_dim, use_scale_shift_norm, 'none'))
        middle_block.append(ResBlock(out_dim, embed_dim, dropout, use_scale_shift_norm=False))
        # self.middle.append(TemporalConvBlockV1(out_dim,dropout=dropout,use_image_dataset=use_image_dataset))
        self.middle_block = nn.CellList(middle_block)

        # decoder
        output_blocks = []
        for i, (in_dim, out_dim) in enumerate(zip(dec_dims[:-1], dec_dims[1:])):
            for j in range(num_res_blocks + 1):
                # residual (+attention) blocks
                # block = [
                #     ResidualBlock(in_dim + shortcut_dims.pop(), embed_dim, out_dim, use_scale_shift_norm, 'none')
                # ]
                block = [
                    ResBlock(
                        in_dim + shortcut_dims.pop(),
                        embed_dim,
                        dropout,
                        out_dim,
                        use_scale_shift_norm=False,
                        use_image_dataset=use_image_dataset,
                    )
                ]
                # block.append(TemporalConvBlockV1(out_dim,dropout=dropout,use_image_dataset=use_image_dataset))
                if scale in attn_scales:
                    # block.append(FlashAttentionBlock(out_dim, context_dim, num_heads, head_dim))
                    block.append(
                        SpatialTransformer(
                            out_dim,
                            out_dim // head_dim,
                            head_dim,
                            depth=1,
                            context_dim=1024,
                            disable_self_attn=False,
                            use_linear=True,
                        )
                    )
                    if self.temporal_attention:
                        if USE_TEMPORAL_TRANSFORMER:
                            block.append(
                                TemporalTransformer(
                                    out_dim,
                                    out_dim // head_dim,
                                    head_dim,
                                    depth=transformer_depth,
                                    context_dim=context_dim,
                                    disable_self_attn=disabled_sa,
                                    use_linear=use_linear_in_temporal,
                                    multiply_zero=use_image_dataset,
                                )
                            )
                        else:
                            block.append(
                                TemporalAttentionMultiBlock(
                                    out_dim,
                                    num_heads,
                                    head_dim,
                                    rotary_emb=self.rotary_emb,
                                    use_image_dataset=use_image_dataset,
                                    use_sim_mask=use_sim_mask,
                                    temporal_attn_times=temporal_attn_times,
                                )
                            )
                in_dim = out_dim

                # upsample
                if i != len(dim_mult) - 1 and j == num_res_blocks:
                    # upsample = ResidualBlock(out_dim, embed_dim, out_dim, use_scale_shift_norm, 'upsample')
                    upsample = Upsample(out_dim, True, dims=2, out_channels=out_dim)
                    scale *= 2.0
                    block.append(upsample)
                    # block.append(TemporalConvBlockV1(out_dim,dropout=dropout,use_image_dataset=use_image_dataset))
                block = nn.CellList(block)
                output_blocks.append(block)

        self.output_blocks = nn.CellList(output_blocks)

        # head
        self.out = nn.SequentialCell(
            GroupNorm(32, out_dim),
            nn.SiLU(),
            nn.Conv2d(out_dim, self.out_dim, 3, pad_mode="pad", padding=1, has_bias=True),
        )

        # zero out the last layer params
        self.out[-1].weight.set_data(initializer("zeros", self.out[-1].weight.shape, self.out[-1].weight.dtype))

    def load_state_dict(self, path, text_to_video_pretrain):
        def prune_weights(sd):
            return {key: p for key, p in sd.items() if "input_blocks.0.0" not in key}

        def fix_typo(sd):
            return {k.replace("temopral_conv", "temporal_conv"): v for k, v in sd.items()}

        if text_to_video_pretrain:
            load_pt_weights_in_model(self, path, (prune_weights, fix_typo))
        else:
            load_pt_weights_in_model(self, path, (fix_typo,))

    def construct(
        self,
        x,
        t,
        y=None,
        depth=None,
        image=None,
        motion=None,
        local_image=None,
        single_sketch=None,
        masked=None,
        canny=None,
        sketch=None,
        histogram=None,
        fps=None,
        video_mask=None,
        focus_present_mask=None,
        # probability at which a given batch sample will focus on the present
        # (0. is all off, 1. is completely arrested attention across time)
        prob_focus_present=0.0,
        mask_last_frame_num=0,
    ):
        assert self.inpainting or masked is None, "inpainting is not supported"

        batch, c, f, h, w = x.shape
        self.batch = batch

        # image and video joint training, if mask_last_frame_num is set, prob_focus_present will be ignored
        if mask_last_frame_num > 0:
            focus_present_mask = None
            video_mask[-mask_last_frame_num:] = False
        else:
            focus_present_mask = default(focus_present_mask, lambda: prob_mask_like((batch,), prob_focus_present))
            # if focus_present_mask.all():
            #     print(focus_present_mask)

        if self.temporal_attention and not USE_TEMPORAL_TRANSFORMER:
            time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2])
        else:
            time_rel_pos_bias = None

        # all-zero and all-keep masks
        zero = ops.zeros(batch, dtype=ms.bool_)
        keep = ops.zeros(batch, dtype=ms.bool_)
        if self.training:
            nzero = (ops.rand(batch) < self.p_all_zero).sum()
            nkeep = (ops.rand(batch) < self.p_all_keep).sum()
            index = ops.randperm(batch)
            zero[index[0:nzero]] = True
            keep[index[nzero : nzero + nkeep]] = True
        assert not zero.any() and not keep.any()
        misc_dropout = partial(self.misc_dropout, zero=zero, keep=keep)

        concat = x.new_zeros((batch, self.concat_dim, f, h, w))
        if depth is not None:
            # DropPath mask
            # b c f h w -> b f c h w -> (b f) c h w
            depth = ops.transpose(depth, (0, 2, 1, 3, 4))
            depth = ops.reshape(depth, (-1, *depth.shape[2:]))
            depth = self.depth_embedding(depth)
            h = depth.shape[2]
            # (b f) c h w -> b f c h w -> b h w f c -> (b h w) f c
            depth = ops.reshape(depth, (batch, depth.shape[0] // batch, *depth.shape[1:]))
            depth = ops.transpose(depth, (0, 3, 4, 1, 2))
            depth = ops.reshape(depth, (-1, *depth.shape[3:]))
            depth = self.depth_embedding_after(depth)

            # (b h w) f c -> b h w f c -> b c f h w
            depth = ops.reshape(depth, (batch, h, depth.shape[0] // (batch * h), *depth.shape[1:]))
            depth = ops.transpose(depth, (0, 4, 3, 1, 2))
            concat = concat + misc_dropout(depth)

        # local_image_embedding
        if local_image is not None:
            # b c f h w -> b f c h w -> (b f) c h w
            local_image = ops.transpose(local_image, (0, 2, 1, 3, 4))
            local_image = ops.reshape(local_image, (-1, *local_image.shape[2:]))
            local_image = self.local_image_embedding(local_image)

            h = local_image.shape[2]
            # (b f) c h w -> b f c h w -> b h w f c -> (b h w) f c
            local_image = ops.reshape(local_image, (batch, local_image.shape[0] // batch, *local_image.shape[1:]))
            local_image = ops.transpose(local_image, (0, 3, 4, 1, 2))
            local_image = ops.reshape(local_image, (-1, *local_image.shape[3:]))
            local_image = self.local_image_embedding_after(local_image)
            # (b h w) f c -> b h w f c -> b c f h w
            local_image = ops.reshape(
                local_image, (batch, h, local_image.shape[0] // (batch * h), *local_image.shape[1:])
            )
            local_image = ops.transpose(local_image, (0, 4, 3, 1, 2))
            concat = concat + misc_dropout(local_image)

        if motion is not None:
            # b c f h w -> b f c h w -> (b f) c h w
            motion = ops.transpose(motion, (0, 2, 1, 3, 4))
            motion = ops.reshape(motion, (-1, *motion.shape[2:]))
            motion = self.motion_embedding(motion)

            h = motion.shape[2]
            # (b f) c h w -> b f c h w -> b h w f c -> (b h w) f c
            motion = ops.reshape(motion, (batch, motion.shape[0] // batch, *motion.shape[1:]))
            motion = ops.transpose(motion, (0, 3, 4, 1, 2))
            motion = ops.reshape(motion, (-1, *motion.shape[3:]))
            motion = self.motion_embedding_after(motion)
            # (b h w) f c -> b h w f c -> b c f h w
            motion = ops.reshape(motion, (batch, h, motion.shape[0] // (batch * h), *motion.shape[1:]))
            motion = ops.transpose(motion, (0, 4, 3, 1, 2))

            if hasattr(self.cfg, "p_zero_motion_alone") and self.cfg.p_zero_motion_alone and self.training:
                motion_d = ops.rand(batch) < self.cfg.p_zero_motion
                motion_d = motion_d[:, None, None, None, None]
                motion = motion.masked_fill(motion_d, 0)
                concat = concat + motion
            else:
                concat = concat + misc_dropout(motion)

        if canny is not None:
            # DropPath mask
            # b c f h w -> b f c h w -> (b f) c h w
            canny = ops.transpose(canny, (0, 2, 1, 3, 4))
            canny = ops.reshape(canny, (-1, *canny.shape[2:]))
            canny = self.canny_embedding(canny)

            h = canny.shape[2]
            # (b f) c h w -> b f c h w -> b h w f c -> (b h w) f c
            canny = ops.reshape(canny, (batch, canny.shape[0] // batch, *canny.shape[1:]))
            canny = ops.transpose(canny, (0, 3, 4, 1, 2))
            canny = ops.reshape(canny, (-1, *canny.shape[3:]))
            canny = self.canny_embedding_after(canny)
            # (b h w) f c -> b h w f c -> b c f h w
            canny = ops.reshape(canny, (batch, h, canny.shape[0] // (batch * h), *canny.shape[1:]))
            canny = ops.transpose(canny, (0, 4, 3, 1, 2))
            concat = concat + misc_dropout(canny)

        if sketch is not None:
            # DropPath mask
            # b c f h w -> b f c h w -> (b f) c h w
            sketch = ops.transpose(sketch, (0, 2, 1, 3, 4))
            sketch = ops.reshape(sketch, (-1, *sketch.shape[2:]))
            sketch = self.sketch_embedding(sketch)

            h = sketch.shape[2]
            # (b f) c h w -> b f c h w -> b h w f c -> (b h w) f c
            sketch = ops.reshape(sketch, (batch, sketch.shape[0] // batch, *sketch.shape[1:]))
            sketch = ops.transpose(sketch, (0, 3, 4, 1, 2))
            sketch = ops.reshape(sketch, (-1, *sketch.shape[3:]))
            sketch = self.sketch_embedding_after(sketch)
            # (b h w) f c -> b h w f c -> b c f h w
            sketch = ops.reshape(sketch, (batch, h, sketch.shape[0] // (batch * h), *sketch.shape[1:]))
            sketch = ops.transpose(sketch, (0, 4, 3, 1, 2))
            concat = concat + misc_dropout(sketch)

        if single_sketch is not None:
            # DropPath mask
            # b c f h w -> b f c h w -> (b f) c h w
            single_sketch = ops.transpose(single_sketch, (0, 2, 1, 3, 4))
            single_sketch = ops.reshape(single_sketch, (-1, *single_sketch.shape[2:]))
            single_sketch = self.single_sketch_embedding(single_sketch)

            h = single_sketch.shape[2]
            # (b f) c h w -> b f c h w -> b h w f c -> (b h w) f c
            single_sketch = ops.reshape(
                single_sketch, (batch, single_sketch.shape[0] // batch, *single_sketch.shape[1:])
            )
            single_sketch = ops.transpose(single_sketch, (0, 3, 4, 1, 2))
            single_sketch = ops.reshape(single_sketch, (-1, *single_sketch.shape[3:]))
            single_sketch = self.single_sketch_embedding_after(single_sketch)
            # (b h w) f c -> b h w f c -> b c f h w
            single_sketch = ops.reshape(
                single_sketch, (batch, h, single_sketch.shape[0] // (batch * h), *single_sketch.shape[1:])
            )
            single_sketch = ops.transpose(single_sketch, (0, 4, 3, 1, 2))
            concat = concat + misc_dropout(single_sketch)

        if masked is not None:
            # DropPath mask
            # b c f h w -> b f c h w -> (b f) c h w
            masked = ops.transpose(masked, (0, 2, 1, 3, 4))
            masked = ops.reshape(masked, (-1, *masked.shape[2:]))
            masked = self.masked_embedding(masked)

            h = masked.shape[2]
            # (b f) c h w -> b f c h w -> b h w f c -> (b h w) f c
            masked = ops.reshape(masked, (batch, masked.shape[0] // batch, *masked.shape[1:]))
            masked = ops.transpose(masked, (0, 3, 4, 1, 2))
            masked = ops.reshape(masked, (-1, *masked.shape[3:]))
            masked = self.mask_embedding_after(masked)
            # (b h w) f c -> b h w f c -> b c f h w
            masked = ops.reshape(masked, (batch, h, masked.shape[0] // (batch * h), *masked.shape[1:]))
            masked = ops.transpose(masked, (0, 4, 3, 1, 2))
            concat = concat + misc_dropout(masked)

        x = ops.cat([x, concat], axis=1)
        # b c f h w -> b f c h w -> (b f) c h w
        x = ops.transpose(x, (0, 2, 1, 3, 4))
        x = ops.reshape(x, (-1, *x.shape[2:]))
        x = self.pre_image(x)
        # (b f) c h w -> b f c h w -> b c f h w
        x = ops.reshape(x, (batch, x.shape[0] // batch, *x.shape[1:]))
        x = ops.transpose(x, (0, 2, 1, 3, 4))

        # embeddings
        if self.use_fps_condition and fps is not None:
            e = self.time_embed(sinusoidal_embedding(t, self.dim)) + self.fps_embedding(
                sinusoidal_embedding(fps, self.dim)
            )
        else:
            e = self.time_embed(sinusoidal_embedding(t, self.dim))

        # context = x.new_zeros((batch, 0, self.context_dim))
        # we don't need context with shape (0,)
        if y is not None:
            y_context = misc_dropout(y)
            # context = ops.cat([context, y_context], axis=1)
            context = y_context
        else:
            y_context = self.zero_y.tile((batch, 1, 1))
            # context = ops.cat([context, y_context], axis=1)
            context = y_context

        if image is not None:
            image_context = misc_dropout(self.pre_image_condition(image))
            context = ops.cat([context, image_context], axis=1)

        # repeat f times for spatial e and context
        e = e.repeat_interleave(repeats=f, dim=0)
        context = context.repeat_interleave(repeats=f, dim=0)

        # always in shape (b f) c h w, except for temporal layer
        # b c f h w -> b f c h w -> (b f) c h w
        x = ops.transpose(x, (0, 2, 1, 3, 4))
        x = ops.reshape(x, (-1, *x.shape[2:]))
        # encoder
        xs = []
        for block in self.input_blocks:
            x = self._forward_single(block, x, e, context, time_rel_pos_bias, focus_present_mask, video_mask)
            xs.append(x)

        # middle
        for block in self.middle_block:
            x = self._forward_single(block, x, e, context, time_rel_pos_bias, focus_present_mask, video_mask)

        # decoder
        for block in self.output_blocks:
            x = ops.cat([x, xs.pop()], axis=1)
            x = self._forward_single(
                block,
                x,
                e,
                context,
                time_rel_pos_bias,
                focus_present_mask,
                video_mask,
                reference=xs[-1] if len(xs) > 0 else None,
            )

        # head
        x = self.out(x)

        # reshape back to (b c f h w)
        # (b f) c h w -> b f c h w -> b c f h w
        x = ops.reshape(x, (batch, x.shape[0] // batch, *x.shape[1:]))
        x = ops.transpose(x, (0, 2, 1, 3, 4))
        return x

    def _forward_single(self, module, x, e, context, time_rel_pos_bias, focus_present_mask, video_mask, reference=None):
        if self.use_checkpoint:
            raise NotImplementedError("Activation checkpointing is not supported for now!")
        if isinstance(module, ResidualBlock):
            x = module(x, e, reference)
        elif isinstance(module, ResBlock):
            x = module(x, e, self.batch)
        elif isinstance(module, SpatialTransformer):
            x = module(x, context)
        elif isinstance(module, TemporalTransformer):
            # (b f) c h w -> b f c h w -> b c f h w
            x = ops.reshape(x, (self.batch, x.shape[0] // self.batch, *x.shape[1:]))
            x = ops.transpose(x, (0, 2, 1, 3, 4))
            x = module(x, context)
            # b c f h w -> b f c h w -> (b f) c h w
            x = ops.transpose(x, (0, 2, 1, 3, 4))
            x = ops.reshape(x, (-1, *x.shape[2:]))
        elif isinstance(module, CrossAttention):
            x = module(x, context)
        elif isinstance(module, MemoryEfficientCrossAttention):
            x = module(x, context)
        elif isinstance(module, BasicTransformerBlock):
            x = module(x, context)
        elif isinstance(module, FeedForward):
            x = module(x, context)
        elif isinstance(module, Upsample):
            x = module(x)
        elif isinstance(module, Downsample):
            x = module(x)
        elif isinstance(module, Resample):
            x = module(x, reference)
        elif isinstance(module, TemporalAttentionBlock):
            # (b f) c h w -> b f c h w -> b c f h w
            x = ops.reshape(x, (self.batch, x.shape[0] // self.batch, *x.shape[1:]))
            x = ops.transpose(x, (0, 2, 1, 3, 4))
            x = module(x, time_rel_pos_bias, focus_present_mask, video_mask)
            # b c f h w -> b f c h w -> (b f) c h w
            x = ops.transpose(x, (0, 2, 1, 3, 4))
            x = ops.reshape(x, (-1, *x.shape[2:]))
        elif isinstance(module, TemporalAttentionMultiBlock):
            # (b f) c h w -> b f c h w -> b c f h w
            x = ops.reshape(x, (self.batch, x.shape[0] // self.batch, *x.shape[1:]))
            x = ops.transpose(x, (0, 2, 1, 3, 4))
            x = module(x, time_rel_pos_bias, focus_present_mask, video_mask)
            # b c f h w -> b f c h w -> (b f) c h w
            x = ops.transpose(x, (0, 2, 1, 3, 4))
            x = ops.reshape(x, (-1, *x.shape[2:]))
        elif isinstance(module, TemporalConvBlockV0):
            # (b f) c h w -> b f c h w -> b c f h w
            x = ops.reshape(x, (self.batch, x.shape[0] // self.batch, *x.shape[1:]))
            x = ops.transpose(x, (0, 2, 1, 3, 4))
            x = module(x)
            # b c f h w -> b f c h w -> (b f) c h w
            x = ops.transpose(x, (0, 2, 1, 3, 4))
            x = ops.reshape(x, (-1, *x.shape[2:]))
        elif isinstance(module, TemporalConvBlockV1):
            # (b f) c h w -> b f c h w -> b c f h w
            x = ops.reshape(x, (self.batch, x.shape[0] // self.batch, *x.shape[1:]))
            x = ops.transpose(x, (0, 2, 1, 3, 4))
            x = module(x)
            # b c f h w -> b f c h w -> (b f) c h w
            x = ops.transpose(x, (0, 2, 1, 3, 4))
            x = ops.reshape(x, (-1, *x.shape[2:]))
        elif isinstance(module, nn.CellList):
            for block in module:
                x = self._forward_single(
                    block, x, e, context, time_rel_pos_bias, focus_present_mask, video_mask, reference
                )
        else:
            x = module(x)
        return x


if __name__ == "__main__":
    from vc.config.base import cfg

    # [model] unet
    model = UNetSD_temporal(
        cfg,
        in_dim=cfg.unet_in_dim,
        dim=cfg.unet_dim,
        y_dim=cfg.unet_y_dim,
        context_dim=cfg.unet_context_dim,
        out_dim=cfg.unet_out_dim,
        dim_mult=cfg.unet_dim_mult,
        num_heads=cfg.unet_num_heads,
        head_dim=cfg.unet_head_dim,
        num_res_blocks=cfg.unet_res_blocks,
        attn_scales=cfg.unet_attn_scales,
        dropout=cfg.unet_dropout,
        temporal_attn_times=0,
        use_checkpoint=cfg.use_checkpoint,
        use_image_dataset=True,
        use_fps_condition=cfg.use_fps_condition,
    )  # .to(gpu)

    print(int(sum(p.numel() for k, p in model.named_parameters()) / (1024**2)), "M parameters")
