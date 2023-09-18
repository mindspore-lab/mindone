import math
import types

import numpy as np
from packaging import version

import mindspore as ms
from mindspore import nn, ops
from mindspore.common.initializer import initializer

##################################################
# Spatial Attention and Transformer
##################################################


def is_old_ms_version(last_old_version="1.10.1"):
    # some APIs are changed after ms 1.10.1 version, such as dropout
    return version.parse(ms.__version__) <= version.parse(last_old_version)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isinstance(d, types.FunctionType) else d


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.get_parameters():
        p.set_data(initializer("zeros", p.shape, p.dtype))
    return module


class GEGLU(nn.Cell):
    def __init__(self, dim_in, dim_out, dtype=ms.float32):
        super().__init__()
        self.proj = nn.Dense(dim_in, dim_out * 2).to_float(dtype)
        self.split = ops.Split(-1, 2)
        self.gelu = ops.GeLU()

    def construct(self, x):
        x, gate = self.split(self.proj(x))

        return x * self.gelu(gate)


class FeedForward(nn.Cell):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0, dtype=ms.float32):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.SequentialCell(nn.Dense(dim, inner_dim).to_float(dtype), nn.GELU().to_float(dtype))
            if not glu
            else GEGLU(dim, inner_dim, dtype=dtype)
        )
        self.net = nn.SequentialCell(
            project_in,
            nn.Dropout(1 - dropout) if is_old_ms_version() else nn.Dropout(p=dropout),
            nn.Dense(inner_dim, dim_out).to_float(dtype),
        )

    def construct(self, x):
        return self.net(x)


class CrossAttention(nn.Cell):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, dtype=ms.float32):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.reshape = ops.Reshape()
        self.softmax = ops.Softmax(axis=-1)
        self.transpose = ops.Transpose()
        self.to_q = nn.Dense(query_dim, inner_dim, has_bias=False).to_float(dtype)
        self.to_k = nn.Dense(context_dim, inner_dim, has_bias=False).to_float(dtype)
        self.to_v = nn.Dense(context_dim, inner_dim, has_bias=False).to_float(dtype)
        self.to_out = nn.SequentialCell(
            nn.Dense(inner_dim, query_dim).to_float(dtype),
            nn.Dropout(1 - dropout) if is_old_ms_version() else nn.Dropout(p=dropout),
        )

    def construct(self, x, context=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        def rearange_in(x):
            # (b, n, h*d) -> (b*h, n, d)
            h = self.heads
            b, n, d = x.shape
            d = d // h

            x = self.reshape(x, (b, n, h, d))
            x = self.transpose(x, (0, 2, 1, 3))
            x = self.reshape(x, (b * h, n, d))
            return x

        q = rearange_in(q)
        k = rearange_in(k)
        v = rearange_in(v)

        sim = ops.matmul(q, self.transpose(k, (0, 2, 1))) * self.scale

        if exists(mask):
            mask = self.reshape(mask, (mask.shape[0], -1))
            if sim.dtype == ms.float16:
                finfo_type = np.float16
            else:
                finfo_type = np.float32
            max_neg_value = -np.finfo(finfo_type).max
            mask = mask.repeat(self.heads, axis=0)
            mask = ops.expand_dims(mask, axis=1)
            sim.masked_fill(mask, max_neg_value)

        attn = self.softmax(sim)
        out = ops.matmul(attn, v)

        def rearange_out(x):
            # (b*h, n, d) -> (b, n, h*d)
            h = self.heads
            b, n, d = x.shape
            b = b // h

            x = self.reshape(x, (b, h, n, d))
            x = self.transpose(x, (0, 2, 1, 3))
            x = self.reshape(x, (b, n, h * d))
            return x

        out = rearange_out(out)
        return self.to_out(out)


class GroupNorm(nn.GroupNorm):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__(num_groups=num_groups, num_channels=num_channels, eps=eps, affine=affine)

    def construct(self, x):
        x_shape = x.shape
        if x.ndim >= 3:
            x = x.view(x_shape[0], x_shape[1], x_shape[2], -1)
        y = super().construct(x)
        return y.view(x_shape)


class BasicTransformerBlock(nn.Cell):
    # ATTENTION_MODES = {
    #     "softmax": CrossAttention,  # vanilla attention
    #     "softmax-xformers" (not supported for now): MemoryEfficientCrossAttention
    # }
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        disable_self_attn=False,
        dtype=ms.float32,
    ):
        super().__init__()
        self.dtype = dtype
        self.disable_self_attn = disable_self_attn
        self.attn1 = CrossAttention(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None,
            dtype=self.dtype,
        )  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff, dtype=self.dtype)
        self.attn2 = CrossAttention(
            query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout, dtype=self.dtype
        )  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(
            [
                dim,
            ],
            epsilon=1e-05,
        ).to_float(ms.float32)
        self.norm2 = nn.LayerNorm(
            [
                dim,
            ],
            epsilon=1e-05,
        ).to_float(ms.float32)
        self.norm3 = nn.LayerNorm(
            [
                dim,
            ],
            epsilon=1e-05,
        ).to_float(ms.float32)
        self.checkpoint = checkpoint

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
        dtype=ms.float32,
    ):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.dtype = dtype
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True).to_float(ms.float32)
        if not use_linear:
            self.proj_in = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0, has_bias=True
            ).to_float(self.dtype)
        else:
            self.proj_in = nn.Dense(in_channels, inner_dim).to_float(self.dtype)

        self.transformer_blocks = nn.CellList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    disable_self_attn=disable_self_attn,
                    checkpoint=use_checkpoint,
                    dtype=self.dtype,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0, has_bias=True).to_float(
                    self.dtype
                )
            )
        else:
            self.proj_out = zero_module(nn.Dense(in_channels, inner_dim).to_float(self.dtype))
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
        x = ops.reshape(x, (x.shape[0], x.shape[1], h, w))
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


##################################################
# Temporal Attention and Transformer #######
##################################################


class TemporalAttentionBlock(nn.Cell):
    def __init__(
        self, dim, heads=4, dim_head=32, rotary_emb=None, use_image_dataset=False, use_sim_mask=False, dtype=ms.float32
    ):
        super().__init__()
        # consider num_heads first, as pos_bias needs fixed num_heads
        # heads = dim // dim_head if dim_head else heads
        dim_head = dim // heads
        assert heads * dim_head == dim
        self.use_image_dataset = use_image_dataset
        self.use_sim_mask = use_sim_mask
        self.dtype = dtype

        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = GroupNorm(32, dim).to_float(ms.float32)
        self.rotary_emb = rotary_emb.to_float(self.dtype)
        self.to_qkv = nn.Dense(dim, hidden_dim * 3).to_float(self.dtype)  # , bias = False)
        self.to_out = nn.Dense(hidden_dim, dim).to_float(self.dtype)  # , bias = False)

    def construct(self, x, pos_bias=None, focus_present_mask=None, video_mask=None):
        identity = x
        n, height = x.shape[2], x.shape[-2]
        b, f, c, h, w = x.shape
        # b c f h w -> b f c h w -> (b f) c h w
        x = x.transpose((0, 2, 1, 3, 4)).reshape((b * f, c, h, w))
        x = self.norm(x)
        # (b f) c h w -> b f c h w -> b c f h w
        x = x.reshape((b, f, c, h, w)).transpose((0, 2, 1, 3, 4))

        # b c f h w -> b c f (h w) -> b (h w) f c
        x = ops.reshape(x, (x.shape[0], x.shape[1], x.shape[2], -1))
        x = ops.transpose(x, (0, 3, 2, 1))

        qkv = self.to_qkv(x).chunk(3, axis=-1)

        if exists(focus_present_mask) and focus_present_mask.all():
            # if all batch samples are focusing on present
            # it would be equivalent to passing that token's values （v=qkv[-1]） through to the output
            values = qkv[-1]
            out = self.to_out(values)
            # b (h w) f c -> b h w f c -> b c f h w
            out = ops.reshape(out, (out.shape[0], height, out.shape[1] // height, out.shape[2], out.shape[3]))
            out = ops.transpose(out, (0, 4, 3, 1, 2))

            return out + identity

        # split out heads
        # ... n (h d) -> ... n h d -> ... h n d
        q, k, v = qkv[0], qkv[1], qkv[2]
        permute_idx = tuple(range(q.ndim - 3)) + (q.ndim - 2, q.ndim - 3, q.ndim - 1)
        q = ops.reshape(q, (q.shape[0], q.shape[1], q.shape[2], self.heads, q.shape[-1] // self.heads))
        q = ops.transpose(q, permute_idx)
        k = ops.reshape(k, (k.shape[0], k.shape[1], k.shape[2], self.heads, k.shape[-1] // self.heads))
        k = ops.transpose(k, permute_idx)
        v = ops.reshape(v, (v.shape[0], v.shape[1], v.shape[2], self.heads, v.shape[-1] // self.heads))
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
            # print(sim.shape,pos_bias.shape)
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
                ops.reshape(attend_self_mask, [1, 1, 1] + attend_self_mask.shape),
                ops.reshape(attend_all_mask, [1, 1, 1] + attend_all_mask.shape),
            )

            sim = sim.masked_fill(~mask, -np.finfo(ms.dtype_to_nptype(sim.dtype)).max)

        if self.use_sim_mask:
            sim_mask = ops.tril(ops.ones((n, n), dtype=ms.bool_), diagonal=0)
            sim = sim.masked_fill(~sim_mask, -np.finfo(ms.dtype_to_nptype(sim.dtype)).max)

        # numerical stability
        sim = sim - sim.amax(axis=-1, keepdims=True)
        attn = sim.float().softmax(axis=-1)

        # aggregate values
        out = ops.bmm(attn, v)

        # ... h n d -> ... n h d -> ... n (h d)
        permute_idx = tuple(range(out.ndim - 3)) + (out.ndim - 2, out.ndim - 3, out.ndim - 1)
        out = ops.transpose(out, (0, 1, 3, 2, 4))
        out = ops.reshape(out, (out.shape[0], out.shape[1], out.shape[2], -1))
        out = self.to_out(out)

        # b (h w) f c -> b h w f c -> b c f h w
        out = ops.reshape(out, (out.shape[0], height, out.shape[1] // height, out.shape[2], out.shape[3]))
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
        dtype=ms.float32,
    ):
        super().__init__()
        self.dtype = dtype
        self.att_layers = nn.CellList(
            [
                TemporalAttentionBlock(dim, heads, dim_head, rotary_emb, use_image_dataset, use_sim_mask, self.dtype)
                for _ in range(temporal_attn_times)
            ]
        )

    def construct(self, x, pos_bias=None, focus_present_mask=None, video_mask=None):
        for layer in self.att_layers:
            x = layer(x, pos_bias, focus_present_mask, video_mask)
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
        dropout=0.0,
        context_dim=None,
        disable_self_attn=False,
        use_linear=False,
        use_checkpoint=True,
        only_self_att=True,
        multiply_zero=False,
        dtype=ms.float32,
    ):
        super().__init__()
        self.multiply_zero = multiply_zero
        self.only_self_att = only_self_att
        self.use_adaptor = False
        self.dtype = dtype
        if self.only_self_att:
            context_dim = None
        if not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True).to_float(ms.float32)
        if not use_linear:
            self.proj_in = nn.Conv1d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0, has_bias=True
            ).to_float(self.dtype)
        else:
            self.proj_in = nn.Dense(in_channels, inner_dim).to_float(self.dtype)
            if self.use_adaptor:
                frames = None
                self.adaptor_in = nn.Dense(frames, frames).to_float(self.dtype)  # todo: what frames

        self.transformer_blocks = nn.CellList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    checkpoint=use_checkpoint,
                    dtype=self.dtype,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv1d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0, has_bias=True).to_float(
                    self.dtype
                )
            )
        else:
            self.proj_out = zero_module(nn.Dense(in_channels, inner_dim).to_float(self.dtype))
            if self.use_adaptor:
                self.adaptor_out = nn.Dense(frames, frames).to_float(self.dtype)  # todo: what frames
        self.use_linear = use_linear

    def construct(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if self.only_self_att:
            context = None
        if not isinstance(context, list):
            context = [context]
        b, c, f, h, w = x.shape
        x_in = x
        # b c f h w -> b f c h w -> (b f) c h w
        x = x.transpose((0, 2, 1, 3, 4)).reshape((b * f, c, h, w))
        x = self.norm(x)
        # (b f) c h w -> b f c h w -> b c f h w
        x = x.reshape((b, f, c, h, w)).transpose((0, 2, 1, 3, 4))

        if not self.use_linear:
            # b c f h w -> b h w c f -> (b h w) c f
            x = ops.transpose(x, (0, 3, 4, 1, 2))
            x = ops.reshape(x, (-1, x.shape[3], x.shape[4]))
            x = self.proj_in(x)
        # [16384, 16, 320]
        if self.use_linear:
            # (b f) c h w -> b f c h w -> b h w f c -> b (h w) f c
            x = ops.reshape(x, (x.shape[0] // self.frames, self.frames, x.shape[1], x.shape[2], x.shape[3]))
            x = ops.transpose(x, (0, 3, 4, 1, 2))
            x = ops.reshape(x, (x.shape[0], -1, x.shape[3], x.shape[4]))  # todo: what frames
            x = self.proj_in(x)

        if self.only_self_att:
            x = ops.transpose(x, (0, 2, 1))
            for i, block in enumerate(self.transformer_blocks):
                x = block(x)
            # (b hw) f c -> b hw f c
            x = ops.reshape(x, (b, x.shape[0] // b, x.shape[1], x.shape[2]))
        else:
            # (b hw) c f -> b hw f c
            x = ops.reshape(x, (b, x.shape[0] // b, x.shape[1], x.shape[2]))
            x = ops.transpose(x, (0, 1, 3, 2))
            for i, block in enumerate(self.transformer_blocks):
                # context[i] = repeat(context[i], '(b f) l con -> b (f r) l con', r=(h*w)//self.frames, f=self.frames).contiguous()
                # (b f) l con -> b f l con
                context[i] = ops.reshape(
                    context[i],
                    (context[i].shape[0] // self.frames, self.frames, context[i].shape[1], context[i].shape[2]),
                )  # todo: wtf frames
                # calculate each batch one by one (since number in shape could not greater then 65,535 for some package)
                for j in range(b):
                    context_i_j = ops.tile(context[i][j], ((h * w) // self.frames, 1, 1))  # todo: wtf frames
                    x[j] = block(x[j], context=context_i_j)

        if self.use_linear:
            x = self.proj_out(x)
            # b (h w) f c -> b h w f c -> b f c h w
            x = ops.reshape(x, (x.shape[0], h, w, x.shape[2:]))
            x = ops.transpose(x, (0, 3, 4, 1, 2))
        if not self.use_linear:
            # b hw f c -> (b hw) f c -> (b hw) c f
            x = ops.reshape(x, (-1, x.shape[2], x.shape[3]))
            x = ops.transpose(x, (0, 2, 1))
            x = self.proj_out(x)
            # (b h w) c f -> b h w c f -> b c f h w
            x = ops.reshape(x, (b, h, w, x.shape[1], x.shape[2]))
            x = ops.transpose(x, (0, 3, 4, 1, 2))

        if self.multiply_zero:
            x = 0.0 * x + x_in
        else:
            x = x + x_in
        return x


class TemporalConvBlock_v2(nn.Cell):
    def __init__(self, in_dim, out_dim=None, dropout=0.0, use_image_dataset=False, dtype=ms.float32):
        super(TemporalConvBlock_v2, self).__init__()
        if out_dim is None:
            out_dim = in_dim  # int(1.5*in_dim)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_image_dataset = use_image_dataset
        self.dtype = dtype

        # conv layers
        self.conv1 = nn.SequentialCell(
            GroupNorm(32, in_dim).to_float(ms.float32),
            nn.SiLU().to_float(self.dtype),
            nn.Conv3d(in_dim, out_dim, (3, 1, 1), pad_mode="pad", padding=(1, 1, 0, 0, 0, 0), has_bias=True).to_float(
                self.dtype
            ),
        )
        self.conv2 = nn.SequentialCell(
            GroupNorm(32, out_dim).to_float(ms.float32),
            nn.SiLU().to_float(self.dtype),
            nn.Dropout(1 - dropout) if is_old_ms_version() else nn.Dropout(p=dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), pad_mode="pad", padding=(1, 1, 0, 0, 0, 0), has_bias=True).to_float(
                self.dtype
            ),
        )
        self.conv3 = nn.SequentialCell(
            GroupNorm(32, out_dim).to_float(ms.float32),
            nn.SiLU().to_float(self.dtype),
            nn.Dropout(1 - dropout) if is_old_ms_version() else nn.Dropout(p=dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), pad_mode="pad", padding=(1, 1, 0, 0, 0, 0), has_bias=True).to_float(
                self.dtype
            ),
        )
        self.conv4 = nn.SequentialCell(
            GroupNorm(32, out_dim).to_float(ms.float32),
            nn.SiLU().to_float(self.dtype),
            nn.Dropout(1 - dropout) if is_old_ms_version() else nn.Dropout(p=dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), pad_mode="pad", padding=(1, 1, 0, 0, 0, 0), has_bias=True).to_float(
                self.dtype
            ),
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


class RelativePositionBias(nn.Cell):
    def __init__(self, heads=8, num_buckets=32, max_distance=128, dtype=ms.float32):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads).to_float(dtype)
        self.dtype = dtype

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
