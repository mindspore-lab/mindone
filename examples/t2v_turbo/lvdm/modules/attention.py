from functools import partial

import numpy as np

import mindspore as ms
from mindspore import mint, nn, ops, recompute

try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILBLE = True
except Exception:
    XFORMERS_IS_AVAILBLE = False
from lvdm.basics import zero_module
from lvdm.common import GroupNormExtend, LayerNorm, default, exists


class RelativePosition(nn.Cell):
    """https://github.com/evelinehong/Transformer_Relative_Position_PyTorch/blob/master/relative_position.py"""

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = ms.Parameter(ms.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def construct(self, length_q, length_k):
        device = self.embeddings_table.device
        range_vec_q = mint.arange(length_q, device=device)
        range_vec_k = mint.arange(length_k, device=device)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = mint.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = final_mat.long()
        embeddings = self.embeddings_table[final_mat]
        return embeddings


class CrossAttention(nn.Cell):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        relative_position=False,
        temporal_length=None,
        img_cross_attention=False,
        dtype=ms.float32,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Dense(query_dim, inner_dim, has_bias=False).to_float(dtype)
        self.to_k = nn.Dense(context_dim, inner_dim, has_bias=False).to_float(dtype)
        self.to_v = nn.Dense(context_dim, inner_dim, has_bias=False).to_float(dtype)
        self.to_out = nn.SequentialCell(nn.Dense(inner_dim, query_dim).to_float(dtype), nn.Dropout(p=dropout))

        self.image_cross_attention_scale = 1.0
        self.text_context_len = 77
        self.img_cross_attention = img_cross_attention
        if self.img_cross_attention:
            self.to_k_ip = nn.Dense(context_dim, inner_dim, has_bias=False).to_float(dtype)
            self.to_v_ip = nn.Dense(context_dim, inner_dim, has_bias=False).to_float(dtype)
        else:
            self.to_k_ip = nn.Identity()
            self.to_v_ip = nn.Identity()

        self.relative_position = relative_position
        if self.relative_position:
            assert temporal_length is not None
            self.relative_position_k = RelativePosition(num_units=dim_head, max_relative_position=temporal_length)
            self.relative_position_v = RelativePosition(num_units=dim_head, max_relative_position=temporal_length)
        else:
            # only used for spatial attention, while NOT for temporal attention
            if XFORMERS_IS_AVAILBLE and temporal_length is None:
                self.forward = self.efficient_forward

    @staticmethod
    def _rearrange_in(x, h):
        # (b, n, h*d) -> (b*h, n, d)
        b, n, d = x.shape
        d = d // h

        x = ops.reshape(x, (b, n, h, d))
        x = ops.transpose(x, (0, 2, 1, 3))
        x = ops.reshape(x, (b * h, n, d))
        return x

    @staticmethod
    def _rearrange_out(x, h):
        # (b*h, n, d) -> (b, n, h*d)
        b, n, d = x.shape
        b = b // h

        x = ops.reshape(x, (b, h, n, d))
        x = ops.transpose(x, (0, 2, 1, 3))
        x = ops.reshape(x, (b, n, h * d))
        return x

    def construct(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)

        k_ip, v_ip, out_ip = None, None, None

        # considering image token additionally
        if context is not None and self.img_cross_attention:
            context, context_img = (
                context[:, : self.text_context_len, :],
                context[:, self.text_context_len :, :],
            )
            k = self.to_k(context)
            v = self.to_v(context)
            k_ip = self.to_k_ip(context_img)
            v_ip = self.to_v_ip(context_img)
        else:
            k = self.to_k(context)
            v = self.to_v(context)

        # q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        # (b, n, h*d) -> (b*h, n, d)
        q = self._rearrange_in(q, h)
        k = self._rearrange_in(k, h)
        v = self._rearrange_in(v, h)

        sim = mint.matmul(q, ops.transpose(k, (0, 2, 1))) * self.scale
        # sim = ops.einsum("b i d, b j d -> b i j", q, k) * self.scale

        if self.relative_position:
            len_q, len_k, len_v = q.shape[1], k.shape[1], v.shape[1]
            k2 = self.relative_position_k(len_q, len_k)
            # sim2 = einsum("b t d, t s d -> b t s", q, k2) * self.scale  # TODO check
            sim2 = mint.matmul(q, ops.transpose(k2, (0, 2, 1))) * self.scale
            sim += sim2

        if exists(mask):
            # feasible for causal attention mask only
            max_neg_value = -np.finfo(sim.dtype).max
            # mask = repeat(mask, "b i j -> (b h) i j", h=h)
            mask = mask.repeat_interleave(h, 0)
            sim = sim.masked_fill(~(mask > 0.5), max_neg_value)

        # attention, what we cannot get enough of
        sim = ops.softmax(sim, axis=-1)
        # out = einsum("b i j, b j d -> b i d", sim, v)
        out = mint.matmul(sim, v)

        if self.relative_position:
            v2 = self.relative_position_v(len_q, len_v)
            # out2 = einsum("b t s, t s d -> b t d", sim, v2)  # TODO check
            out2 = mint.matmul(sim, v2)  # TODO check
            out += out2

        # out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        out = self._rearrange_out(out, h)

        # considering image token additionally
        if context is not None and self.img_cross_attention:
            # k_ip, v_ip = map(
            #     lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (k_ip, v_ip)
            # )
            k_ip = self._rearrange_in(k_ip, h)
            v_ip = self._rearrange_in(v_ip, h)
            # sim_ip = einsum("b i d, b j d -> b i j", q, k_ip) * self.scale
            sim_ip = mint.matmul(q, ops.transpose(k_ip, (0, 2, 1))) * self.scale
            sim_ip = mint.nn.functional.softmax(sim_ip, dim=-1)
            # out_ip = einsum("b i j, b j d -> b i d", sim_ip, v_ip)
            out_ip = mint.matmul(sim_ip, v_ip)
            # out_ip = rearrange(out_ip, "(b h) n d -> b n (h d)", h=h)
            out_ip = self._rearrange_out(out_ip, h)
            out = out + self.image_cross_attention_scale * out_ip

        return self.to_out(out)

    def efficient_forward(self, x, context=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)

        # considering image token additionally
        if context is not None and self.img_cross_attention:
            context, context_img = (
                context[:, : self.text_context_len, :],
                context[:, self.text_context_len :, :],
            )
            k = self.to_k(context)
            v = self.to_v(context)
            k_ip = self.to_k_ip(context_img)
            v_ip = self.to_v_ip(context_img)
        else:
            k = self.to_k(context)
            v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )
        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=None)

        # considering image token additionally
        if context is not None and self.img_cross_attention:
            k_ip, v_ip = map(
                lambda t: t.unsqueeze(3)
                .reshape(b, t.shape[1], self.heads, self.dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b * self.heads, t.shape[1], self.dim_head)
                .contiguous(),
                (k_ip, v_ip),
            )
            out_ip = xformers.ops.memory_efficient_attention(q, k_ip, v_ip, attn_bias=None, op=None)
            out_ip = (
                out_ip.unsqueeze(0)
                .reshape(b, self.heads, out.shape[1], self.dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b, out.shape[1], self.heads * self.dim_head)
            )

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        if context is not None and self.img_cross_attention:
            out = out + self.image_cross_attention_scale * out_ip
        return self.to_out(out)


class BasicTransformerBlock(nn.Cell):
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
        attention_cls=None,
        img_cross_attention=False,
    ):
        super().__init__()
        attn_cls = CrossAttention if attention_cls is None else attention_cls
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None,
        )
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            img_cross_attention=img_cross_attention,
        )
        self.norm1 = LayerNorm((dim,))
        self.norm2 = LayerNorm((dim,))
        self.norm3 = LayerNorm((dim,))
        self.checkpoint = checkpoint

    def construct(self, x, context=None, mask=None):
        x = (
            self.attn1(
                self.norm1(x),
                context=context if self.disable_self_attn else None,
                mask=mask,
            )
            + x
        )
        x = self.attn2(self.norm2(x), context=context, mask=mask) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Cell):
    """
    Transformer block for image-like data in spatial axis.
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
        use_checkpoint=True,
        disable_self_attn=False,
        use_linear=False,
        img_cross_attention=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = GroupNormExtend(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
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
                    context_dim=context_dim,
                    img_cross_attention=img_cross_attention,
                    disable_self_attn=disable_self_attn,
                    checkpoint=use_checkpoint,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0, has_bias=True)
            )
        else:
            self.proj_out = zero_module(nn.Dense(inner_dim, in_channels))
        self.use_linear = use_linear

    def construct(self, x, context=None):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)

        # x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        x = ops.reshape(x, (b, c, h * w))  # (b, c, h*w)
        x = ops.transpose(x, (0, 2, 1))  # (b, h*w, c)

        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            if block.checkpoint and self.training:
                x = recompute(block, x, context=context)
            else:
                x = block(x, context=context)
        if self.use_linear:
            x = self.proj_out(x)

        # x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        x = ops.reshape(x, (b, h, w, c))  # (b, h, w, c)
        x = ops.transpose(x, (0, 3, 1, 2))  # (b, c, h, w)

        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


class TemporalTransformer(nn.Cell):
    """
    Transformer block for image-like data in temporal axis.
    First, reshape to b, t, d.
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
        use_checkpoint=True,
        use_linear=False,
        only_self_att=True,
        causal_attention=False,
        relative_position=False,
        temporal_length=None,
    ):
        super().__init__()
        self.only_self_att = only_self_att
        self.relative_position = relative_position
        self.causal_attention = causal_attention
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = GroupNormExtend(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Conv1d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0, has_bias=True)
        if not use_linear:
            self.proj_in = nn.Conv1d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0, has_bias=True)
        else:
            self.proj_in = nn.Dense(in_channels, inner_dim)

        if relative_position:
            assert temporal_length is not None
            attention_cls = partial(CrossAttention, relative_position=True, temporal_length=temporal_length)
        else:
            attention_cls = None
        if self.causal_attention:
            assert temporal_length is not None
            self.mask = ops.tril(mint.ones([1, temporal_length, temporal_length]))

        if self.only_self_att:
            context_dim = None
        self.transformer_blocks = nn.CellList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim,
                    attention_cls=attention_cls,
                    checkpoint=use_checkpoint,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv1d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0, has_bias=True)
            )
        else:
            self.proj_out = zero_module(nn.Dense(inner_dim, in_channels))
        self.use_linear = use_linear

    def construct(self, x, context=None):
        b, c, t, h, w = x.shape
        x_in = x
        x = self.norm(x)

        # x = rearrange(x, "b c t h w -> (b h w) c t").contiguous()
        x = x.permute(0, 3, 4, 1, 2)
        x = x.reshape(-1, x.shape[3], x.shape[4])

        if not self.use_linear:
            x = self.proj_in(x)

        # x = rearrange(x, "bhw c t -> bhw t c").contiguous()
        x = x.permute(0, 2, 1)

        if self.use_linear:
            x = self.proj_in(x)

        mask = None
        if self.causal_attention:
            mask = self.mask
            # mask = repeat(mask, "l i j -> (l bhw) i j", bhw=b * h * w)
            mask = mask.repeat_interleave(b * h * w, 0)

        if self.only_self_att:
            # note: if no context is given, cross-attention defaults to self-attention
            for i, block in enumerate(self.transformer_blocks):
                if block.checkpoint and self.training:
                    x = recompute(block, x, mask=mask)
                else:
                    x = block(x, mask=mask)
            # x = rearrange(x, "(b hw) t c -> b hw t c", b=b).contiguous()
            x = x.reshape(b, -1, x.shape[1], x.shape[2])
        else:
            # x = rearrange(x, "(b hw) t c -> b hw t c", b=b).contiguous()
            x = ops.reshape(x, (b, x.shape[0] // b, x.shape[1], x.shape[2]))
            x = ops.transpose(x, (0, 1, 3, 2))

            # context = rearrange(context, "(b t) l con -> b t l con", t=t).contiguous()
            _, l, con = context.shape
            context = context.reshape(b, t, l, con)

            for i, block in enumerate(self.transformer_blocks):
                # calculate each batch one by one (since number in shape could not greater then 65,535 for some package)
                for j in range(b):
                    rep = (h * w) // context[j].shape[0]
                    context_j = context[j].repeat_interleave(rep, 0)
                    # context_j = repeat(
                    #     context[j], "t l con -> (t r) l con", r=(h * w) // t, t=t
                    # ).contiguous()
                    # note: causal mask will not applied in cross-attention case
                    if block.checkpoint and self.training:
                        x[j] = recompute(block, x[j], context=context_j)
                    else:
                        x[j] = block(x[j], context=context_j)

        if self.use_linear:
            x = self.proj_out(x)
            # x = rearrange(x, "b (h w) t c -> b c t h w", h=h, w=w).contiguous()
            x = ops.reshape(x, (x.shape[0], h, w, x.shape[2], x.shape[3]))
            x = ops.transpose(x, (0, 4, 3, 1, 2))
        else:
            # x = rearrange(x, "b hw t c -> (b hw) c t").contiguous()
            x = ops.reshape(x, (-1, x.shape[2], x.shape[3]))
            x = ops.transpose(x, (0, 2, 1))
            x = self.proj_out(x)
            # x = rearrange(x, "(b h w) c t -> b c t h w", b=b, h=h, w=w).contiguous()
            x = ops.reshape(x, (b, h, w, x.shape[1], x.shape[2]))
            x = ops.transpose(x, (0, 3, 4, 1, 2))

        return x + x_in


class GEGLU(nn.Cell):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Dense(dim_in, dim_out * 2)

    def construct(self, x):
        x, gate = self.proj(x).chunk(2, axis=-1)
        return x * ops.gelu(gate)


class FeedForward(nn.Cell):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.SequentialCell(nn.Dense(dim, inner_dim), nn.GELU()) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.SequentialCell(project_in, nn.Dropout(p=dropout), nn.Dense(inner_dim, dim_out))

    def construct(self, x):
        return self.net(x)


class LinearAttention(nn.Cell):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, has_bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)


class SpatialSelfAttention(nn.Cell):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = GroupNormExtend(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def construct(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        # q = rearrange(q, "b c h w -> b (h w) c")
        q = q.reshape(b, -1, c)
        # k = rearrange(k, "b c h w -> b c (h w)")
        k = k.reshape(b, c, -1)
        # w_ = ops.einsum("bij,bjk->bik", q, k)
        w_ = mint.matmul(q, k)  # TODO: check!

        w_ = w_ * (int(c) ** (-0.5))
        w_ = mint.nn.functional.softmax(w_, dim=2)

        # attend to values
        # v = rearrange(v, "b c h w -> b c (h w)")
        b, c, h, w = v.shape
        v = v.reshape(b, c, -1)
        # w_ = rearrange(w_, "b i j -> b j i")
        w_ = w_.permute(0, 2, 1)
        # h_ = ops.einsum("bij,bjk->bik", v, w_)
        h_ = mint.matmul(v, w_)  # TODO: check!!
        # h_ = rearrange(h_, "b c (h w) -> b c h w", h=h)
        h_ = h_.reshape(h_.shape[0], h_.shape[1], -1)
        h_ = self.proj_out(h_)

        return x + h_
