# reference to https://github.com/Stability-AI/generative-models

from gm.modules.diffusionmodules.util import normalization, zero_module
from gm.modules.transformers import scaled_dot_product_attention
from gm.util import default, exists

import mindspore as ms
from mindspore import nn, ops

try:
    # FIXME: some error with mindspore.nn.layer.flash_attention.FlashAttention in mindspore 2.1.0
    # from mindspore.nn.layer.flash_attention import FlashAttention
    from mindspore.ops._op_impl._custom_op.flash_attention.flash_attention_impl import get_flash_attention

    FLASH_IS_AVAILABLE = True
    print("flash attention is available.")
except ImportError:
    FLASH_IS_AVAILABLE = False
    print("flash attention is unavailable.")


# feedforward
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
        project_in = nn.SequentialCell([nn.Dense(dim, inner_dim), nn.GELU()]) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.SequentialCell([project_in, nn.Dropout(p=dropout), nn.Dense(inner_dim, dim_out)])

    def construct(self, x):
        return self.net(x)


# TODO: Add Flash Attention Support
class LinearAttention(nn.Cell):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, has_bias=False, pad_mode="valid")
        self.to_out = nn.Conv2d(hidden_dim, dim, 1, has_bias=True, pad_mode="valid")

    def construct(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)

        # shape, "b (qkv heads c) h w -> qkv b heads c (h w)"
        qkv = qkv.view(b, 3, self.heads, c, -1).swapaxes(
            0, 1
        )  # b (qkv heads c) h w -> b qkv heads c (h w) -> qkv b heads c (h w)
        q, k, v = ops.split(qkv, 1)
        q, k, v = q.squeeze(0), k.squeeze(0), v.squeeze(0)

        k = ops.softmax(k, axis=-1)

        # context = ops.einsum("bhdn,bhen->bhde", k, v)
        context = ops.BatchMatMul(transpose_b=True)(k, v)  # bhdn  # bhen  # bhde

        # out = ops.einsum("bhde,bhdn->bhen", context, q)
        out = ops.BatchMatMul(transpose_a=True)(context, q)  # bhde  # bhdn  # bhen

        # shape, b heads c (h w) -> b (heads c) h w
        out = out.view(out.shape[0], -1, h, w)

        return self.to_out(out)


# reference to https://arxiv.org/abs/2205.14135
class FlashAttention(nn.Cell):
    def __init__(self):
        super(FlashAttention, self).__init__()
        self.flash_attention = get_flash_attention(tiling_stgy_name="sparse")

    def construct(self, q, k, v, attention_mask=None, dropout_mask=None, alibi_mask=None):
        # ALiBi, reference to https://arxiv.org/abs/2108.12409
        _, h, Nq, d = q.shape
        dim_mask = ops.ones((d,), dtype=ms.int8)
        scale = d**-0.25
        q = q * scale
        k = k * scale
        o, l, m = self.flash_attention(q, k, v, dim_mask, attention_mask, dropout_mask, alibi_mask)

        return o  # (b, h, n, d)


class MemoryEfficientCrossAttention(nn.Cell):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
    ):
        super().__init__()

        assert FLASH_IS_AVAILABLE

        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Dense(query_dim, inner_dim, has_bias=False)
        self.to_k = nn.Dense(context_dim, inner_dim, has_bias=False)
        self.to_v = nn.Dense(context_dim, inner_dim, has_bias=False)

        self.to_out = nn.SequentialCell(nn.Dense(inner_dim, query_dim), nn.Dropout(p=dropout))

        self.flash_attention = FlashAttention()

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
            out = self.flash_attention(q, k, v, mask)
        else:
            out = scaled_dot_product_attention(q, k, v, attn_mask=mask)  # scale is dim_head ** -0.5 per default

        # rearange_out, "b h n d -> b n (h d)"
        b, h, n, d = out.shape
        out = out.transpose(0, 2, 1, 3).view(b, n, -1)

        if additional_tokens is not None:
            # remove additional token
            out = out[:, n_tokens_to_mask:]

        return self.to_out(out)


class CrossAttention(nn.Cell):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Dense(query_dim, inner_dim, has_bias=False)
        self.to_k = nn.Dense(context_dim, inner_dim, has_bias=False)
        self.to_v = nn.Dense(context_dim, inner_dim, has_bias=False)

        self.to_out = nn.SequentialCell(nn.Dense(inner_dim, query_dim), nn.Dropout(p=dropout))

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

        # b n (h d) -> b h n d
        q_b, q_n, _ = q.shape
        q = q.view(q_b, q_n, h, -1).transpose(0, 2, 1, 3)
        k_b, k_n, _ = k.shape
        k = k.view(k_b, k_n, h, -1).transpose(0, 2, 1, 3)
        v_b, v_n, _ = v.shape
        v = v.view(v_b, v_n, h, -1).transpose(0, 2, 1, 3)

        out = scaled_dot_product_attention(q, k, v, attn_mask=mask)  # scale is dim_head ** -0.5 per default

        # b h n d -> b n (h d)
        b, h, n, d = out.shape
        out = out.transpose(0, 2, 1, 3).view(b, n, -1)

        if additional_tokens is not None:
            # remove additional token
            out = out[:, n_tokens_to_mask:]

        return self.to_out(out)


class BasicTransformerBlock(nn.Cell):
    ATTENTION_MODES = {
        "vanilla": CrossAttention,  # vanilla attention
        "flash-attention": MemoryEfficientCrossAttention,  # flash attention
    }

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        disable_self_attn=False,
        attn_mode="vanilla",  # ["vanilla", "flash-attention"]
    ):
        super().__init__()
        assert attn_mode in self.ATTENTION_MODES
        if attn_mode != "vanilla" and not FLASH_IS_AVAILABLE:
            print(
                f"Attention mode '{attn_mode}' is not available. Falling back to native attention. "
                f"This is not a problem in MindSpore >= 2.0.1 on Ascend devices; "
                f"FYI, you are running with MindSpore version {ms.__version__}"
            )
            attn_mode = "vanilla"
        attn_cls = self.ATTENTION_MODES[attn_mode]
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
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )  # is self-attn if context is none
        self.norm1 = nn.LayerNorm([dim], epsilon=1e-5)
        self.norm2 = nn.LayerNorm([dim], epsilon=1e-5)
        self.norm3 = nn.LayerNorm([dim], epsilon=1e-5)

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
        attn_type="vanilla",
    ):
        super().__init__()
        print(f"constructing {self.__class__.__name__} of depth {depth} w/ {in_channels} channels and {n_heads} heads")
        from omegaconf import ListConfig

        if exists(context_dim) and not isinstance(context_dim, (list, ListConfig)):
            context_dim = [context_dim]
        if exists(context_dim) and isinstance(context_dim, list):
            if depth != len(context_dim):
                print(
                    f"WARNING: {self.__class__.__name__}: Found context dims {context_dim} of depth {len(context_dim)}, "
                    f"which does not match the specified 'depth' of {depth}. Setting context_dim to {depth * [context_dim[0]]} now."
                )
                # depth does not match context dims.
                assert all(
                    map(lambda x: x == context_dim[0], context_dim)
                ), "need homogenous context_dim to match depth automatically"
                context_dim = depth * [context_dim[0]]
        elif context_dim is None:
            context_dim = [None] * depth
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = normalization(in_channels, eps=1e-6)
        if not use_linear:
            self.proj_in = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0, has_bias=True, pad_mode="valid"
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
                    attn_mode=attn_type,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0, has_bias=True, pad_mode="valid")
            )
        else:
            self.proj_out = zero_module(nn.Dense(inner_dim, in_channels))
        self.use_linear = use_linear

    def construct(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, (list, tuple)):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)

        # b c h w -> b (h w) c
        x = x.view(*x.shape[:2], -1).transpose(0, 2, 1)

        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            if i > 0 and len(context) == 1:
                i = 0  # use same context for each block
            x = block(x, context=context[i])
        if self.use_linear:
            x = self.proj_out(x)

        # b (h w) c -> b c h w
        x = x.transpose(0, 2, 1)
        x = x.view(*x.shape[:2], h, w)

        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in
