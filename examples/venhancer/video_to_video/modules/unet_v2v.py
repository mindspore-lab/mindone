import sys
from abc import abstractmethod
from typing import Callable, List, Optional, Tuple, Union

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, mint
from mindspore.ops.operations.nn_ops import FlashAttentionScore

sys.path.append("../../")


from ._layers import GELU, GroupNorm, LayerNorm

USE_TEMPORAL_TRANSFORMER = True


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
        mask = x.new_ones(b, dtype=ms.bool)
        if keep is not None:
            mask[keep] = False
        if zero is not None:
            mask[zero] = False

        # drop-path index
        index = mint.nonzero(mask, as_tuple=True)[0]
        index = index[ops.randperm(len(index))[:n]]
        if zero is not None:
            index = mint.cat([index, mint.nonzero(zero, as_tuple=True)[0]], dim=0)

        # drop-path multiplier
        multiplier = x.new_ones(b)
        multiplier[index] = 0.0
        output = tuple(u * self.broadcast(multiplier, u) for u in args)
        return output[0] if len(args) == 1 else output

    def broadcast(self, src, dst):
        assert src.shape[0] == dst.shape[0]
        shape = (dst.shape[0],) + (1,) * (dst.ndim - 1)
        return src.view(shape)


def sinusoidal_embedding(timesteps: Tensor, dim: int) -> Tensor:
    # check input
    half = dim // 2
    timesteps = timesteps.float()

    # compute sinusoidal embedding
    sinusoid = ops.outer(timesteps, mint.pow(10000, -mint.arange(half, dtype=ms.float32).div(half)))
    x = mint.cat([mint.cos(sinusoid), mint.sin(sinusoid)], dim=1)
    if dim % 2 != 0:
        x = mint.cat([x, mint.zeros_like(x[:, :1])], dim=1)
    return x


def exists(x: Optional[Tensor]) -> bool:
    return x is not None


def default(val: Optional[Tensor], d: Union[Tensor, Callable[..., Tensor]]) -> Tensor:
    if exists(val):
        return val
    return d() if callable(d) else d


def prob_mask_like(shape, prob):
    if prob == 1:
        return mint.ones(shape, dtype=ms.bool)
    elif prob == 0:
        return mint.zeros(shape, dtype=ms.bool)
    else:
        mask = mint.zeros(shape).float().uniform_(0, 1) < prob
        # aviod mask all, which will cause find_unused_parameters error
        if mask.all():
            mask[0] = False
        return mask


class CrossAttention(nn.Cell):
    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        max_bs: int = 16384,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.max_bs = max_bs
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = mint.nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = mint.nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = mint.nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.SequentialCell(mint.nn.Linear(inner_dim, query_dim), mint.nn.Dropout(p=dropout))

        self.attention = FlashAttentionScore(1, scale_value=self.dim_head**-0.5, input_layout="BSH")

    def _rearange_in(self, x: Tensor, b: int) -> Tensor:
        # (b, n, h*d) -> (b*h, n, d)
        shape = x.shape[1]
        x = x.unsqueeze(3)
        x = x.reshape(b, shape, self.heads, self.dim_head)
        x = x.transpose(0, 2, 1, 3)
        x = x.reshape(b * self.heads, shape, self.dim_head)
        x = x.contiguous()
        return x

    def _rearange_out(self, x: Tensor, b: int) -> Tensor:
        # (b*h, n, d) -> (b, n, h*d)
        shape = x.shape[1]
        x = x.unsqueeze(0)
        x = x.reshape(b, self.heads, shape, self.dim_head)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(b, shape, self.heads * self.dim_head)
        return x

    def construct(self, x: Tensor, context: Optional[Tensor] = None, mask: Optional[Tensor] = None):
        assert mask is None

        q = self.to_q(x)
        context = default(context, x)

        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q = self._rearange_in(q, b)
        k = self._rearange_in(k, b)
        v = self._rearange_in(v, b)

        # actually compute the attention, what we cannot get enough of.
        if q.shape[0] > self.max_bs:
            q_list = mint.chunk(q, q.shape[0] // self.max_bs, dim=0)
            k_list = mint.chunk(k, k.shape[0] // self.max_bs, dim=0)
            v_list = mint.chunk(v, v.shape[0] // self.max_bs, dim=0)
            out_list = []
            for q_1, k_1, v_1 in zip(q_list, k_list, v_list):
                _, _, _, out = self.attention(q_1, k_1, v_1, None, None, None, None)
                out_list.append(out)
            out = mint.cat(out_list, dim=0)
        else:
            _, _, _, out = self.attention(q, k, v, None, None, None, None)

        out = self._rearange_out(out, b)
        return self.to_out(out)


class RelativePositionBias(nn.Cell):
    def __init__(self, heads=8, num_buckets=32, max_distance=128):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = mint.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = (
            max_exact
            + (
                mint.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)  # noqa
            ).long()
        )
        val_if_large = mint.min(val_if_large, ops.full_like(val_if_large, num_buckets - 1))

        ret += mint.where(is_small, n, val_if_large)
        return ret

    def construct(self, n):
        q_pos = mint.arange(n, dtype=ms.float32)
        k_pos = mint.arange(n, dtype=ms.float32)
        rel_pos = ops.reshape(k_pos, (1, -1)) - ops.reshape(q_pos, (-1, 1))
        rp_bucket = self._relative_position_bucket(
            rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance
        )
        values = self.relative_attention_bias(rp_bucket)
        return ops.transpose(values, (2, 0, 1))


class SpatialTransformer(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        n_heads: int,
        d_head: int,
        depth: int = 1,
        dropout: float = 0.0,
        context_dim: Union[None, int, List[int]] = None,
        disable_self_attn: bool = False,
        use_linear: bool = False,
    ) -> None:
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        if not use_linear:
            self.proj_in = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0, pad_mode="pad", has_bias=True
            )
        else:
            self.proj_in = mint.nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.CellList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    disable_self_attn=disable_self_attn,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0, pad_mode="pad", has_bias=True)
            )
        else:
            self.proj_out = zero_module(mint.nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

    def construct(self, x: Tensor, context: Union[None, Tensor, List[Tensor]] = None) -> Tensor:
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        # b c h w -> b (h w) c
        x = ops.transpose(x, (0, 2, 3, 1))
        x = ops.reshape(x, (b, -1, c)).contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i])
        if self.use_linear:
            x = self.proj_out(x)
        # b (h w) c -> b c h w
        x = ops.reshape(x, (b, h, w, c))
        x = ops.transpose(x, (0, 3, 1, 2)).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


class BasicTransformerBlock(nn.Cell):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_head: int,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,
        gated_ff: bool = True,
        disable_self_attn: bool = False,
    ) -> None:
        super().__init__()
        attn_cls = CrossAttention
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None,
        )
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)

        attn_cls2 = CrossAttention

        self.attn2 = attn_cls2(query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        self.norm1 = LayerNorm(dim, elementwise_affine=True)
        self.norm2 = LayerNorm(dim, elementwise_affine=True)
        self.norm3 = LayerNorm(dim, elementwise_affine=True)

    def construct(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class GEGLU(nn.Cell):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.proj = mint.nn.Linear(dim_in, dim_out * 2)
        self.gelu = GELU()

    def construct(self, x: Tensor) -> Tensor:
        x, gate = self.proj(x).chunk(2, axis=-1)
        return x * self.gelu(gate)


def zero_module(module: nn.Cell):
    """
    Zero out the parameters of a module and return it.
    """
    for _, p in module.parameters_and_names():
        ops.assign(p, mint.zeros_like(p))
    return module


def avg_pool_nd(dims: int, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return mint.nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, has_bias=kwargs.pop("has_bias", True), pad_mode=kwargs.pop("pad_mode", "pad"), **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, has_bias=kwargs.pop("has_bias", True), pad_mode=kwargs.pop("pad_mode", "pad"), **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, has_bias=kwargs.pop("has_bias", True), pad_mode=kwargs.pop("pad_mode", "pad"), **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class FeedForward(nn.Cell):
    def __init__(
        self, dim: int, dim_out: Optional[int] = None, mult: int = 4, glu: bool = False, dropout: float = 0.0
    ) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.SequentialCell(mint.nn.Linear(dim, inner_dim), GELU()) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.SequentialCell(project_in, mint.nn.Dropout(p=dropout), mint.nn.Linear(inner_dim, dim_out))

    def construct(self, x: Tensor) -> Tensor:
        return self.net(x)


class Upsample(nn.Cell):
    def __init__(
        self, channels: int, use_conv: bool, dims: int = 2, out_channels: Optional[int] = None, padding: int = 1
    ) -> None:
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=padding, pad_mode="pad", has_bias=True)

    def construct(self, x: Tensor) -> Tensor:
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = ops.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        else:
            x = ops.interpolate(x, scale_factor=2.0, recompute_scale_factor=True, mode="nearest")
            x = x[..., 1:-1, :]
        if self.use_conv:
            x = self.conv(x)
        return x


class ResBlock(nn.Cell):
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float,
        out_channels: Optional[int] = None,
        use_conv: bool = False,
        use_scale_shift_norm: bool = False,
        dims: int = 2,
        up: bool = False,
        down: bool = False,
        use_temporal_conv: bool = True,
        use_image_dataset: bool = False,
    ) -> None:
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
            nn.Conv2d(channels, self.out_channels, 3, padding=1, pad_mode="pad", has_bias=True),
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
            mint.nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.SequentialCell(
            GroupNorm(32, self.out_channels),
            nn.SiLU(),
            mint.nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1, pad_mode="pad", has_bias=True)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1, pad_mode="pad", has_bias=True)

        if self.use_temporal_conv:
            self.temopral_conv = TemporalConvBlock_v2(
                self.out_channels, self.out_channels, dropout=0.1, use_image_dataset=use_image_dataset
            )

    def construct(self, x: Tensor, emb: Tensor, batch_size: int) -> Tensor:
        return self._construct(x, emb, batch_size)

    def _construct(self, x: Tensor, emb: Tensor, batch_size: int) -> Tensor:
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = mint.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        h = self.skip_connection(x) + h

        if self.use_temporal_conv:
            # (b f) c h w -> b c f h w
            h = ops.reshape(h, (batch_size, -1, *h.shape[1:]))
            h = ops.transpose(h, (0, 2, 1, 3, 4))
            h = self.temopral_conv(h)
            # b c f h w -> (b f) c h w
            h = ops.transpose(h, (0, 2, 1, 3, 4))
            h = ops.reshape(h, (-1, *h.shape[2:]))
        return h


class Downsample(nn.Cell):
    def __init__(
        self,
        channels: int,
        use_conv: bool,
        dims: int = 2,
        out_channels: Optional[int] = None,
        padding: Tuple[int, int] = (2, 2, 1, 1),
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = nn.Conv2d(
                self.channels, self.out_channels, 3, stride=stride, padding=padding, pad_mode="pad", has_bias=True
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def construct(self, x: Tensor):
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
            GroupNorm(32, in_dim), nn.SiLU(), nn.Conv2d(in_dim, out_dim, 3, padding=1, pad_mode="pad", has_bias=True)
        )
        self.resample = Resample(in_dim, in_dim, mode)
        self.embedding = nn.SequentialCell(
            nn.SiLU(), mint.nn.Linear(embed_dim, out_dim * 2 if use_scale_shift_norm else out_dim)
        )
        self.layer2 = nn.SequentialCell(
            GroupNorm(32, out_dim),
            nn.SiLU(),
            mint.nn.Dropout(dropout),
            nn.Conv2d(out_dim, out_dim, 3, padding=1, pad_mode="pad", has_bias=True),
        )
        self.shortcut = (
            nn.Identity() if in_dim == out_dim else nn.Conv2d(in_dim, out_dim, 1, pad_mode="pad", has_bias=True)
        )

        # zero out the last layer params
        ops.assign(self.layer2[-1].weight, mint.zeros_like(self.layer2[-1].weight))

    def construct(self, x, e, reference=None):
        identity = self.resample(x, reference)
        x = self.layer1[-1](self.resample(self.layer1[:-1](x), reference))
        e = self.embedding(e).unsqueeze(-1).unsqueeze(-1).type(x.dtype)
        if self.use_scale_shift_norm:
            scale, shift = e.chunk(2, dim=1)
            x = self.layer2[0](x) * (1 + scale) + shift
            x = self.layer2[1:](x)
        else:
            x = x + e
            x = self.layer2(x)
        x = x + self.shortcut(identity)
        return x


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
        dropout=1.0,
        context_dim=None,
        disable_self_attn=False,
        use_linear=False,
        only_self_att=True,
        multiply_zero=False,
        causal_attention=False,
        causal_block_size=1,
        relative_position=False,
        temporal_length=None,
        dtype=ms.float32,
    ):
        super().__init__()
        self.multiply_zero = multiply_zero
        self.only_self_att = only_self_att
        self.relative_position = relative_position
        self.causal_attention = causal_attention
        self.causal_block_size = causal_block_size
        self.dtype = dtype
        if self.only_self_att:
            context_dim = None
        if not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        if not use_linear:
            self.proj_in = nn.Conv1d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0, has_bias=True, pad_mode="pad"
            ).to_float(self.dtype)
        else:
            self.proj_in = mint.nn.Linear(in_channels, inner_dim).to_float(self.dtype)

        self.transformer_blocks = nn.CellList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv1d(
                    inner_dim, in_channels, kernel_size=1, stride=1, padding=0, has_bias=True, pad_mode="pad"
                ).to_float(self.dtype)
            )
        else:
            self.proj_out = zero_module(nn.Dense(in_channels, inner_dim).to_float(self.dtype))
        self.use_linear = use_linear

    def construct(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if self.only_self_att:
            context = None
        if not isinstance(context, list):
            context = [context]
        b, c, t, h, w = x.shape
        x_in = x
        x = self.norm(x)

        # b c t h w -> (b h w) c t
        x = ops.transpose(x, (0, 3, 4, 1, 2))
        x = ops.reshape(x, (-1, x.shape[3], x.shape[4]))
        if not self.use_linear:
            x = self.proj_in(x)
        # bhw c t -> bhw t c
        x = ops.transpose(x, (0, 2, 1))
        if self.use_linear:
            x = self.proj_in(x)

        if self.only_self_att:
            # x = ops.transpose(x, (0, 2, 1))
            for i, block in enumerate(self.transformer_blocks):
                x = block(x)
            # (b hw) f c -> b hw f c
            x = ops.reshape(x, (b, x.shape[0] // b, x.shape[1], x.shape[2]))
        else:
            # (b hw) c f -> b hw f c
            x = ops.reshape(x, (b, x.shape[0] // b, x.shape[1], x.shape[2]))
            x = ops.transpose(x, (0, 1, 3, 2))
            for i, block in enumerate(self.transformer_blocks):
                # (b f) l con -> b f l con
                context[i] = ops.reshape(
                    context[i],
                    (context[i].shape[0] // self.frames, self.frames, context[i].shape[1], context[i].shape[2]),
                )  # todo: wtf frames
                # calculate each batch one by one (since number in shape could not greater then 65,535 for some package)
                for j in range(b):
                    context_i_j = mint.tile(context[i][j], ((h * w) // self.frames, 1, 1))  # todo: wtf frames
                    x[j] = block(x[j], context=context_i_j)

        if self.use_linear:
            x = self.proj_out(x)
            # b (h w) t c -> b h w t c -> b c t h w
            x = ops.reshape(x, (x.shape[0], h, w, x.shape[2], x.shape[3]))
            # x = ops.transpose(x, (0, 3, 4, 1, 2))
            x = ops.transpose(x, (0, 4, 3, 1, 2))
        if not self.use_linear:
            # b hw t c -> (b hw) t c -> (b hw) c t
            x = ops.reshape(x, (-1, x.shape[2], x.shape[3]))
            x = ops.transpose(x, (0, 2, 1))
            x = self.proj_out(x)
            # (b h w) c t -> b h w c t -> b c t h w
            x = ops.reshape(x, (b, h, w, x.shape[1], x.shape[2]))
            x = ops.transpose(x, (0, 3, 4, 1, 2))

        if self.multiply_zero:
            x = 0.0 * x + x_in
        else:
            x = x + x_in
        return x


class TemporalConvBlock_v2(nn.Cell):
    def __init__(
        self, in_dim: int, out_dim: Optional[int] = None, dropout: float = 0.0, use_image_dataset: bool = False
    ) -> None:
        super(TemporalConvBlock_v2, self).__init__()
        if out_dim is None:
            out_dim = in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_image_dataset = use_image_dataset

        # conv layers
        self.conv1 = nn.SequentialCell(
            GroupNorm(32, in_dim),
            nn.SiLU(),
            nn.Conv3d(in_dim, out_dim, (3, 1, 1), padding=(1, 1, 0, 0, 0, 0), pad_mode="pad", has_bias=True),
        )
        self.conv2 = nn.SequentialCell(
            GroupNorm(32, out_dim),
            nn.SiLU(),
            mint.nn.Dropout(p=dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 1, 0, 0, 0, 0), pad_mode="pad", has_bias=True),
        )
        self.conv3 = nn.SequentialCell(
            GroupNorm(32, out_dim),
            nn.SiLU(),
            mint.nn.Dropout(p=dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 1, 0, 0, 0, 0), pad_mode="pad", has_bias=True),
        )
        self.conv4 = nn.SequentialCell(
            GroupNorm(32, out_dim),
            nn.SiLU(),
            mint.nn.Dropout(p=dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 1, 0, 0, 0, 0), pad_mode="pad", has_bias=True),
        )

        # zero out the last layer params,so the conv block is identity
        ops.assign(self.conv4[-1].weight, mint.zeros_like(self.conv4[-1].weight))
        ops.assign(self.conv4[-1].bias, mint.zeros_like(self.conv4[-1].bias))

    def construct(self, x: Tensor) -> Tensor:
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


class Vid2VidSDUNet(nn.Cell):
    def __init__(
        self,
        in_dim=4,
        dim=320,
        y_dim=1024,
        context_dim=1024,
        out_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_heads=8,
        head_dim=64,
        num_res_blocks=2,
        attn_scales=[1 / 1, 1 / 2, 1 / 4],
        use_scale_shift_norm=True,
        dropout=0.1,
        temporal_attn_times=1,
        temporal_attention=True,
        use_image_dataset=False,
        use_fps_condition=False,
        use_sim_mask=False,
        training=False,
        inpainting=True,
    ):
        embed_dim = dim * 4
        num_heads = num_heads if num_heads else dim // 32
        super(Vid2VidSDUNet, self).__init__()
        self.in_dim = in_dim
        self.dim = dim
        self.y_dim = y_dim
        self.context_dim = context_dim
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
        self.use_image_dataset = use_image_dataset
        self.use_fps_condition = use_fps_condition
        self.use_sim_mask = use_sim_mask
        self.training = training
        self.inpainting = inpainting

        use_linear_in_temporal = False
        transformer_depth = 1
        disabled_sa = False
        # params
        enc_dims = [dim * u for u in [1] + dim_mult]
        dec_dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        shortcut_dims = []
        scale = 1.0

        # embeddings
        self.time_embed = nn.SequentialCell(
            mint.nn.Linear(dim, embed_dim), nn.SiLU(), mint.nn.Linear(embed_dim, embed_dim)
        )

        if self.use_fps_condition:
            self.fps_embedding = nn.SequentialCell(
                mint.nn.Linear(dim, embed_dim), nn.SiLU(), mint.nn.Linear(embed_dim, embed_dim)
            )
            ops.assign(self.fps_embedding[-1].weight, mint.zeros_like(self.fps_embedding[-1].weight))
            ops.assign(self.fps_embedding[-1].bias, mint.zeros_like(self.fps_embedding[-1].bias))

        # encoder
        self.input_blocks = nn.CellList()
        init_block = nn.CellList([nn.Conv2d(self.in_dim, dim, 3, padding=1, pad_mode="pad", has_bias=True)])
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
                raise NotImplementedError
        self.input_blocks.append(init_block)
        shortcut_dims.append(dim)
        for i, (in_dim, out_dim) in enumerate(zip(enc_dims[:-1], enc_dims[1:])):
            for j in range(num_res_blocks):
                block = nn.CellList(
                    [
                        ResBlock(
                            in_dim,
                            embed_dim,
                            dropout,
                            out_channels=out_dim,
                            use_scale_shift_norm=False,
                            use_image_dataset=use_image_dataset,
                        )
                    ]
                )
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
                            raise NotImplementedError
                in_dim = out_dim
                self.input_blocks.append(block)
                shortcut_dims.append(out_dim)

                # downsample
                if i != len(dim_mult) - 1 and j == num_res_blocks - 1:
                    downsample = Downsample(out_dim, True, dims=2, out_channels=out_dim)
                    shortcut_dims.append(out_dim)
                    scale /= 2.0
                    self.input_blocks.append(downsample)

        self.middle_block = nn.CellList(
            [
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
        )

        if self.temporal_attention:
            if USE_TEMPORAL_TRANSFORMER:
                self.middle_block.append(
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
                raise NotImplementedError

        self.middle_block.append(ResBlock(out_dim, embed_dim, dropout, use_scale_shift_norm=False))

        # decoder
        self.output_blocks = nn.CellList()
        for i, (in_dim, out_dim) in enumerate(zip(dec_dims[:-1], dec_dims[1:])):
            for j in range(num_res_blocks + 1):
                block = nn.CellList(
                    [
                        ResBlock(
                            in_dim + shortcut_dims.pop(),
                            embed_dim,
                            dropout,
                            out_dim,
                            use_scale_shift_norm=False,
                            use_image_dataset=use_image_dataset,
                        )
                    ]
                )
                if scale in attn_scales:
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
                            raise NotImplementedError
                in_dim = out_dim

                # upsample
                if i != len(dim_mult) - 1 and j == num_res_blocks:
                    upsample = Upsample(out_dim, True, dims=2.0, out_channels=out_dim)
                    scale *= 2.0
                    block.append(upsample)
                self.output_blocks.append(block)

        # head
        self.out = nn.SequentialCell(
            GroupNorm(32, out_dim),
            nn.SiLU(),
            nn.Conv2d(out_dim, self.out_dim, 3, padding=1, pad_mode="pad", has_bias=True),
        )

        # zero out the last layer params
        ops.assign(self.out[-1].weight, mint.zeros_like(self.out[-1].weight))

    def construct(self, x, t, y, x_lr=None, fps=None, mask_last_frame_num=0):
        batch, c, f, h, w = x.shape
        self.batch = batch

        # embeddings
        e = self.time_embed(sinusoidal_embedding(t, self.dim))
        context = y

        # repeat f times for spatial e and context
        e = e.repeat_interleave(repeats=f, dim=0)
        context = context.repeat_interleave(repeats=f, dim=0)

        # always in shape (b f) c h w, except for temporal layer
        # b c f h w -> (b f) c h w
        x = ops.transpose(x, (0, 2, 1, 3, 4))
        x = ops.reshape(x, (-1, *x.shape[2:]))
        # encoder
        xs = []
        for ind, block in enumerate(self.input_blocks):
            x = self._construct_single(block, x, e, context)
            xs.append(x)

        # middle
        for block in self.middle_block:
            x = self._construct_single(block, x, e, context)

        # decoder
        for block in self.output_blocks:
            x = mint.cat([x, xs.pop()], dim=1)
            x = self._construct_single(block, x, e, context, reference=xs[-1] if len(xs) > 0 else None)

        # head
        x = self.out(x)

        # (b f) c h w -> (b c f h w)
        x = ops.reshape(x, (batch, -1, *x.shape[1:]))
        x = ops.transpose(x, (0, 2, 1, 3, 4))

        return x

    def _construct_single(self, module, x, e, context, reference=None):
        if isinstance(module, ResidualBlock):
            x = x.contiguous()
            x = module(x, e, reference)
        elif isinstance(module, ResBlock):
            x = x.contiguous()
            x = module(x, e, self.batch)
        elif isinstance(module, SpatialTransformer):
            x = module(x, context)
        elif isinstance(module, TemporalTransformer):
            # (b f) c h w -> (b c f h w)
            x = ops.reshape(x, (self.batch, -1, *x.shape[1:]))
            x = ops.transpose(x, (0, 2, 1, 3, 4))
            x = module(x, context)
            # b c f h w -> (b f) c h w
            x = ops.transpose(x, (0, 2, 1, 3, 4))
            x = ops.reshape(x, (-1, *x.shape[2:]))
            x = module(x, context)
        elif isinstance(module, CrossAttention):
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
        elif isinstance(module, nn.CellList):
            for block in module:
                x = self._construct_single(block, x, e, context, reference)
        else:
            x = module(x)
        return x


class ControlledV2VUNet(Vid2VidSDUNet):
    def __init__(self):
        super(ControlledV2VUNet, self).__init__()
        self.VideoControlNet = VideoControlNet()

    def construct(
        self,
        x,
        t,
        y,
        hint=None,
        t_hint=None,
        s_cond=None,
        mask_cond=None,
        x_lr=None,
        fps=None,
        mask_last_frame_num=0,
    ):
        batch, c, f, h, w = x.shape
        self.batch = batch

        control = self.VideoControlNet(x, t, y, hint=hint, t_hint=t_hint, mask_cond=mask_cond, s_cond=s_cond)

        e = self.time_embed(sinusoidal_embedding(t, self.dim))
        e = e.repeat_interleave(repeats=f, dim=0)

        context = y
        context = context.repeat_interleave(repeats=f, dim=0)

        # always in shape (b f) c h w, except for temporal layer
        # b c f h w -> (b f) c h w
        x = ops.transpose(x, (0, 2, 1, 3, 4))
        x = ops.reshape(x, (-1, *x.shape[2:]))
        # encoder
        xs = []
        for block in self.input_blocks:
            x = self._construct_single(block, x, e, context)
            xs.append(x)
        # middle
        for block in self.middle_block:
            x = self._construct_single(block, x, e, context)

        if control is not None:
            x = control.pop() + x

        # decoder
        for block in self.output_blocks:
            if control is None:
                x = mint.cat([x, xs.pop()], dim=1)
            else:
                x = mint.cat([x, xs.pop() + control.pop()], dim=1)
            x = self._construct_single(block, x, e, context, reference=xs[-1] if len(xs) > 0 else None)

        # head
        x = self.out(x)

        # reshape back to (b c f h w)
        # (b f) c h w -> (b c f h w)
        x = ops.reshape(x, (batch, -1, *x.shape[1:]))
        x = ops.transpose(x, (0, 2, 1, 3, 4))
        return x

    def _construct_single(self, module, x, e, context, reference=None):
        if isinstance(module, ResidualBlock):
            x = x.contiguous()
            x = module(x, e, reference)
        elif isinstance(module, ResBlock):
            x = x.contiguous()
            x = module(x, e, self.batch)
        elif isinstance(module, SpatialTransformer):
            x = module(x, context)
        elif isinstance(module, TemporalTransformer):
            # (b f) c h w -> (b c f h w)
            x = ops.reshape(x, (self.batch, -1, *x.shape[1:]))
            x = ops.transpose(x, (0, 2, 1, 3, 4))
            x = module(x, context)
            # b c f h w -> (b f) c h w
            x = ops.transpose(x, (0, 2, 1, 3, 4))
            x = ops.reshape(x, (-1, *x.shape[2:]))
        elif isinstance(module, CrossAttention):
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
        elif isinstance(module, nn.CellList):
            for block in module:
                x = self._construct_single(block, x, e, context, reference)
        else:
            x = module(x)
        return x


class VideoControlNet(nn.CellList):
    def __init__(
        self,
        in_dim=4,
        dim=320,
        y_dim=1024,
        context_dim=1024,
        out_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_heads=8,
        head_dim=64,
        num_res_blocks=2,
        attn_scales=[1 / 1, 1 / 2, 1 / 4],
        use_scale_shift_norm=True,
        dropout=0.1,
        temporal_attn_times=1,
        temporal_attention=True,
        use_image_dataset=False,
        use_fps_condition=False,
        use_sim_mask=False,
        training=False,
        inpainting=True,
    ):
        embed_dim = dim * 4
        num_heads = num_heads if num_heads else dim // 32
        super(VideoControlNet, self).__init__()
        self.in_dim = in_dim
        self.dim = dim
        self.y_dim = y_dim
        self.context_dim = context_dim
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
        self.use_image_dataset = use_image_dataset
        self.use_fps_condition = use_fps_condition
        self.use_sim_mask = use_sim_mask
        self.training = training
        self.inpainting = inpainting

        use_linear_in_temporal = False
        transformer_depth = 1
        disabled_sa = False
        # params
        enc_dims = [dim * u for u in [1] + dim_mult]
        shortcut_dims = []
        scale = 1.0

        # embeddings
        self.time_embed = nn.SequentialCell(
            mint.nn.Linear(dim, embed_dim), nn.SiLU(), mint.nn.Linear(embed_dim, embed_dim)
        )

        self.hint_time_zero_linear = zero_module(mint.nn.Linear(embed_dim, embed_dim))

        # scale prompt
        self.scale_cond = nn.SequentialCell(
            mint.nn.Linear(dim, embed_dim), nn.SiLU(), zero_module(mint.nn.Linear(embed_dim, embed_dim))
        )

        if self.use_fps_condition:
            self.fps_embedding = nn.SequentialCell(
                mint.nn.Linear(dim, embed_dim), nn.SiLU(), mint.nn.Linear(embed_dim, embed_dim)
            )
            ops.assign(self.fps_embedding[-1].weight, mint.zeros_like(self.fps_embedding[-1].weight))
            ops.assign(self.fps_embedding[-1].bias, mint.zeros_like(self.fps_embedding[-1].bias))

        # encoder
        self.input_blocks = nn.CellList()
        init_block = nn.CellList([nn.Conv2d(self.in_dim, dim, 3, padding=1, pad_mode="pad", has_bias=True)])
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
                raise NotImplementedError
        self.input_blocks.append(init_block)
        self.zero_convs = nn.CellList([self.make_zero_conv(dim)])
        shortcut_dims.append(dim)
        for i, (in_dim, out_dim) in enumerate(zip(enc_dims[:-1], enc_dims[1:])):
            for j in range(num_res_blocks):
                block = nn.CellList(
                    [
                        ResBlock(
                            in_dim,
                            embed_dim,
                            dropout,
                            out_channels=out_dim,
                            use_scale_shift_norm=False,
                            use_image_dataset=use_image_dataset,
                        )
                    ]
                )
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
                            raise NotImplementedError
                in_dim = out_dim
                self.input_blocks.append(block)
                self.zero_convs.append(self.make_zero_conv(out_dim))
                shortcut_dims.append(out_dim)

                # downsample
                if i != len(dim_mult) - 1 and j == num_res_blocks - 1:
                    downsample = Downsample(out_dim, True, dims=2, out_channels=out_dim)
                    shortcut_dims.append(out_dim)
                    scale /= 2.0
                    self.input_blocks.append(downsample)
                    self.zero_convs.append(self.make_zero_conv(out_dim))

        self.middle_block = nn.CellList(
            [
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
        )

        if self.temporal_attention:
            if USE_TEMPORAL_TRANSFORMER:
                self.middle_block.append(
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
                raise NotImplementedError
        self.middle_block.append(ResBlock(out_dim, embed_dim, dropout, use_scale_shift_norm=False))

        self.middle_block_out = self.make_zero_conv(embed_dim)

        """
        add prompt
        """
        add_dim = 320
        self.add_dim = add_dim

        self.input_hint_block = zero_module(nn.Conv2d(4, add_dim, 3, padding=1, pad_mode="pad", has_bias=True))

    def make_zero_conv(self, in_channels, out_channels=None):
        out_channels = in_channels if out_channels is None else out_channels
        return TimestepEmbedSequential(
            zero_module(nn.Conv2d(in_channels, out_channels, 1, padding=0, pad_mode="pad", has_bias=True))
        )

    def construct(
        self,
        x,
        t,
        y,
        s_cond=None,
        hint=None,
        t_hint=None,
        mask_cond=None,
    ):
        batch, x_c, f, h, w = x.shape
        self.batch = batch

        if hint is not None:
            add = x.new_zeros((batch, self.add_dim, f, h, w))
            # b c f h w -> (b f) c h w
            hints = ops.transpose(hint, (0, 2, 1, 3, 4))
            hints = ops.reshape(hints, (-1, *hints.shape[2:]))
            hints = self.input_hint_block(hints)
            # (b f) c h w -> (b c f h w)
            hints = ops.reshape(hints, (batch, -1, *hints.shape[1:]))
            hints = ops.transpose(hints, (0, 2, 1, 3, 4))
            if mask_cond is not None:
                for i in range(batch):
                    mask_cond_per_batch = mask_cond[i]
                    inds = mint.nonzero(mask_cond_per_batch >= 0, as_tuple=True)[0]
                    hint_inds = mask_cond_per_batch[inds]
                    add[i, :, inds] += hints[i, :, hint_inds]
                    # add[i,:,inds] += hints[i]
            # b c f h w -> (b f) c h w
            add = ops.transpose(add, (0, 2, 1, 3, 4))
            add = ops.reshape(add, (-1, *add.shape[2:]))

        e = self.time_embed(sinusoidal_embedding(t, self.dim))
        e = e.repeat_interleave(repeats=f, dim=0)

        if t_hint is not None:
            e_cond = self.hint_time_zero_linear(self.time_embed(sinusoidal_embedding(t_hint, self.dim)))
            if mask_cond is not None:
                # (b f) d -> (b f d)
                e = ops.reshape(e, (batch, -1, e.shape[-1]))
                for i in range(batch):
                    mask_cond_per_batch = mask_cond[i]
                    inds = mint.nonzero(mask_cond_per_batch >= 0, as_tuple=True)[0]
                    e[i, inds] += e_cond[i]
                # (b f d) -> (b f) d
                e = ops.reshape(e, (-1, e.shape[-1]))
            else:
                e_cond = e_cond.repeat_interleave(repeats=f, dim=0)
                e += e_cond

        if s_cond is not None:
            e_scale = self.scale_cond(sinusoidal_embedding(s_cond, self.dim))
            if mask_cond is not None:
                # (b f) d -> (b f d)
                e = ops.reshape(e, (batch, -1, e.shape[-1]))
                for i in range(batch):
                    mask_cond_per_batch = mask_cond[i]
                    inds = mint.nonzero(mask_cond_per_batch >= 0, as_tuple=True)[0]
                    e[i, inds] += e_scale[i]
                    # (b f d) -> (b f) d
                    e = ops.reshape(e, (-1, e.shape[-1]))
            else:
                e_scale = e_scale.repeat_interleave(repeats=f, dim=0)
                e += e_scale

        context = y.repeat_interleave(repeats=f, dim=0)

        # always in shape (b f) c h w, except for temporal layer
        # b c f h w -> (b f) c h w
        x = ops.transpose(x, (0, 2, 1, 3, 4))
        x = ops.reshape(x, (-1, *x.shape[2:]))

        # encoder
        xs = []
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if hint is not None:
                for block in module:
                    x = self._construct_single(block, x, e, context)
                    if not isinstance(block, TemporalTransformer):
                        if hint is not None:
                            x += add
                            hint = None
            else:
                x = self._construct_single(module, x, e, context)
            xs.append(zero_conv(x, e, context))

        # middle
        for block in self.middle_block:
            x = self._construct_single(block, x, e, context)
        xs.append(self.middle_block_out(x, e, context))

        return xs

    def _construct_single(self, module, x, e, context, reference=None):
        if isinstance(module, ResidualBlock):
            x = x.contiguous()
            x = module(x, e, reference)
        elif isinstance(module, ResBlock):
            x = x.contiguous()
            x = module(x, e, self.batch)
        elif isinstance(module, SpatialTransformer):
            x = module(x, context)
        elif isinstance(module, TemporalTransformer):
            # (b f) c h w -> (b c f h w)
            x = ops.reshape(x, (self.batch, -1, *x.shape[1:]))
            x = ops.transpose(x, (0, 2, 1, 3, 4))
            x = module(x, context)
            # b c f h w -> (b f) c h w
            x = ops.transpose(x, (0, 2, 1, 3, 4))
            x = ops.reshape(x, (-1, *x.shape[2:]))
        elif isinstance(module, CrossAttention):
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
        elif isinstance(module, nn.CellList):
            for block in module:
                x = self._construct_single(block, x, e, context, reference)
        else:
            x = module(x)
        return x


class TimestepBlock(nn.Cell):
    """
    Any module where construct() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def construct(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.SequentialCell, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def construct(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x
