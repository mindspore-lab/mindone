from typing import Callable, List, Optional, Tuple, Union

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, mint

from mindone.models.modules.flash_attention import MSFlashAttention

from ._layers import GELU, GroupNorm, LayerNorm


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
        n = (mint.rand(b) < self.p).sum()

        # non-zero and non-keep mask
        mask = x.new_ones(b, dtype=ms.bool)
        if keep is not None:
            mask[keep] = False
        if zero is not None:
            mask[zero] = False

        # drop-path index
        index = mint.where(mask)[0]
        index = index[ops.randperm(len(index))[:n]]
        if zero is not None:
            index = mint.cat([index, mint.where(zero)[0]], dim=0)

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

class MemoryEfficientCrossAttention(nn.Cell):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        max_bs=16384,
        dropout=0.0,
    ):
        super().__init__()

        assert FLASH_IS_AVAILABLE

        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = mint.nn.Linear(query_dim, inner_dim, has_bias=False)
        self.to_k = mint.nn.Linear(context_dim, inner_dim, has_bias=False)
        self.to_v = mint.nn.Linear(context_dim, inner_dim, has_bias=False)

        self.to_out = nn.SequentialCell(nn.Dense(inner_dim, query_dim), mint.nn.Dropout(p=dropout))

        self.flash_attention = FlashAttention(head_dim=dim_head, head_num=heads, high_precision=True)

    def construct(self, x, context=None, mask=None, additional_tokens=None):
        h = self.heads

        n_tokens_to_mask = 0
        if additional_tokens is not None:
            # get the number of masked tokens at the beginning of the output sequence
            n_tokens_to_mask = additional_tokens.shape[1]
            # add additional token
            x = ops.concat((additional_tokens, x), axis=1)

        q = self.to_q(x)
        if context is None:
            context = x
        k = self.to_k(context)
        v = self.to_v(context)

        # rearange_in, "b n (h d) -> b h n d"
        q_b, q_n, _ = q.shape
        q = q.view(q_b, q_n, h, -1).transpose(0, 2, 1, 3)
        k_b, k_n, _ = k.shape
        k = k.view(k_b, k_n, h, -1).transpose(0, 2, 1, 3)
        v_b, v_n, _ = v.shape
        v = v.view(v_b, v_n, h, -1).transpose(0, 2, 1, 3)

        head_dim = q.shape[-1]
        if q_n % 16 == 0 and k_n % 16 == 0 and head_dim <= 256:
            if mask is None:
                mask = mint.zeros((q_b, q_n, q_n), ms.uint8)
            out = self.flash_attention(q.to(ms.float16), k.to(ms.float16), v.to(ms.float16), mask.to(ms.uint8))
        else:
            out = scaled_dot_product_attention(q, k, v, attn_mask=mask)  # scale is dim_head ** -0.5 per default

        # rearange_out, "b h n d -> b n (h d)"
        b, h, n, d = out.shape
        out = out.transpose(0, 2, 1, 3).view(b, n, -1)
        dtype = q.dtype
        out = out.to(dtype)

        if additional_tokens is not None:
            # remove additional token
            out = out[:, n_tokens_to_mask:]

        return self.to_out(out)




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
        return nn.AvgPool2d(*args, **kwargs)
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


class GEGLU(nn.Cell):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.proj = nn.Dense(dim_in, dim_out * 2)
        self.gelu = GELU()

    def construct(self, x: Tensor) -> Tensor:
        x, gate = self.proj(x).chunk(2, axis=-1)
        return x * self.gelu(gate)


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

        self.to_q = nn.Dense(query_dim, inner_dim, has_bias=False)
        self.to_k = nn.Dense(context_dim, inner_dim, has_bias=False)
        self.to_v = nn.Dense(context_dim, inner_dim, has_bias=False)
        self.to_out = nn.SequentialCell(nn.Dense(inner_dim, query_dim), nn.Dropout(p=dropout))

        self.attention = MSFlashAttention(self.dim_head, self.heads)

    def _rearange_in(self, x: Tensor, b: int) -> Tensor:
        x = x.unsqueeze(3)
        x = x.reshape(b, x.shape[1], self.heads, self.dim_head)
        x = x.transpose(0, 2, 1, 3)
        x = x.reshape(b * self.heads, x.shape[1], self.dim_head)
        x = x.contiguous()
        return x

    def _rearange_out(self, x: Tensor, b: int) -> Tensor:
        x = x.unsqueeze(0)
        x = x.reshape(b, self.heads, x.shape[1], self.dim_head)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(b, x.shape[1], self.heads * self.dim_head)
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
                out = self.attention(q_1, k_1, v_1)
                out_list.append(out)
            out = mint.cat(out_list, dim=0)
        else:
            out = self.attention(q, k, v)

        out = self._rearange_out(out)
        return self.to_out(out)


class FeedForward(nn.Cell):
    def __init__(
        self, dim: int, dim_out: Optional[int] = None, mult: int = 4, glu: bool = False, dropout: float = 0.0
    ) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.SequentialCell(nn.Dense(dim, inner_dim), GELU()) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.SequentialCell(project_in, nn.Dropout(p=dropout), nn.Dense(inner_dim, dim_out))

    def construct(self, x: Tensor) -> Tensor:
        return self.net(x)


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
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.norm3 = LayerNorm(dim)

    def construct(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


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
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0, pad_mode="pad", has_bias=True)
            )
        else:
            self.proj_out = zero_module(nn.Dense(in_channels, inner_dim))
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
            x = ops.interpolate(x, scale_factor=2, mode="nearest")
            x = x[..., 1:-1, :]
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Cell):
    def __init__(
        self,
        channels: int,
        use_conv: bool,
        dims: int = 2,
        out_channels: Optional[int] = None,
        padding: Tuple[int, int] = (2, 1),
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
            nn.Conv3d(in_dim, out_dim, (3, 1, 1), padding=(1, 0, 0), pad_mode="pad", has_bias=True),
        )
        self.conv2 = nn.SequentialCell(
            GroupNorm(32, out_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0), pad_mode="pad", has_bias=True),
        )
        self.conv3 = nn.SequentialCell(
            GroupNorm(32, out_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0), pad_mode="pad", has_bias=True),
        )
        self.conv4 = nn.SequentialCell(
            GroupNorm(32, out_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0), pad_mode="pad", has_bias=True),
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
            nn.Dense(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.SequentialCell(
            GroupNorm(32, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
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
            h = ops.reshape(h, (batch_size, -1, *h.shape[2:]))
            h = ops.transpose(h, (0, 2, 1, 3, 4))
            h = self.temopral_conv(h)
            # b c f h w -> (b f) c h w
            h = ops.transpose(h, (0, 2, 1, 3, 4))
            h = ops.reshape(h, (-1.0 * h.shape[2:]))
        return h
