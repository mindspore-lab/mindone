import mindspore as ms
from mindspore import nn, ops
from mindcv.models.layers import DropPath
from .dit import Attention, SelfAttention, Mlp, LayerNorm, GELU


class MultiHeadCrossAttention(nn.Cell):
    """
    Flash attention doesnot work well (leading to noisy images) for SD1.5-based models on 910B up to MS2.2.1-20231122 version,
    due to the attention head dimension is 40, num heads=5. Require test on future versions
    """
    def __init__(self, d_model, num_heads, attn_drop=0.0, proj_drop=0.0, has_bias=True, use_FA=False):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # TODO: it's better to remove bias 
        self.q_linear = nn.Dense(d_model, d_model, has_bias=has_bias)
        self.kv_linear = nn.Dense(d_model, d_model * 2, has_bias=has_bias)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Dense(d_model, d_model, has_bias=has_bias)
        self.proj_drop = nn.Dropout(p=proj_drop)

        self.attention = Attention(self.head_dim, attn_drop=attn_drop)
        self.use_FA = use_FA

    @staticmethod
    def _rearange_in(x):
        # (b, n, h, d) -> (b h n d) -> (b*h, n, d)
        b, n, h, d = x.shape
        x = ops.transpose(x, (0, 2, 1, 3))
        x = ops.reshape(x, (b * h, n, d))
        return x

    @staticmethod
    def _rearange_out(x, h):
        # (b*h, n, d) -> (b h n d) -> (b n h d) ->  (b n h*d)
        bh, n, d = x.shape
        b = bh // h

        x = ops.reshape(x, (b, h, n, d))
        x = ops.transpose(x, (0, 2, 1, 3))
        x = ops.reshape(x, (b, n, h*d))
        return x

    def construct(self, x, cond, mask=None):
        # C = head_dim * num_heads
        # query/value: img tokens; key: condition; mask: if padding tokens
        B, N, C = x.shape

        q = self.q_linear(x).reshape((1, -1, self.num_heads, self.head_dim))
        kv = self.kv_linear(cond).reshape((1, -1, 2, self.num_heads, self.head_dim))

        k, v = kv.unbind(2)

        attn_bias = None
        # print('D--', q.shape, k.shape, v.shape)
        if self.use_FA:
            # x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)
            # x = x.reshape((B, -1, C))
            raise NotImplementedError
            # (b, n, h, d) -> (b, n, h*d)
        else:
            # (b, n, h, d) -> (b h n d) -> (b*h, n, d)
            q = self._rearange_in(q)
            k = self._rearange_in(k)
            v = self._rearange_in(v)

            x = self.attention(q, k, v, mask)
            # (b*h, n, d) -> (b h n d) -> (b n h d) ->  (b n h*d)
            x = self._rearange_out(x, self.num_heads)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift


class STDiTBlock(nn.Cell):
    def __init__(
        self,
        hidden_size,
        num_heads,
        d_s=None,
        d_t=None,
        mlp_ratio=4.0,
        drop_path=0.0,
        enable_flashattn=False,
        enable_layernorm_kernel=False,
        enable_sequence_parallelism=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        assert not enable_layernorm_kernel, "Not implemented" 
        # self.enable_flashattn = enable_flashattn
        # self._enable_sequence_parallelism = enable_sequence_parallelism

        self.attn_cls = SelfAttention
        self.mha_cls = MultiHeadCrossAttention

        self.norm1 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.attn = self.attn_cls(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
        )
        self.cross_attn = self.mha_cls(hidden_size, num_heads)
        self.norm2 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        approx_gelu = lambda: GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale_shift_table = ms.Parameter(ops.randn(6, hidden_size) / hidden_size**0.5)

        # temporal attention
        self.d_s = d_s
        self.d_t = d_t

        self.attn_temp = self.attn_cls(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
        )

    @staticmethod
    def _rearrange_in_S(x, T):
        # x_s = rearrange(x_m, "B (T S) C -> (B T) S C", T=self.d_t, S=self.d_s)
        B, TS, C = x.shape
        S = TS // T
        x = ops.reshape(x, (B*T, S, C))
        return x

    @staticmethod
    def _rearrange_out_S(x, T):
        # x_s = rearrange(x_s, "(B T) S C -> B (T S) C", T=self.d_t, S=self.d_s)
        BT, S, C = x.shape
        B = BT // T
        x = ops.reshape(x, (B, T*S, C))
        return x

    @staticmethod
    def _rearrange_in_T(x, T):
        # x_t = rearrange(x, "B (T S) C -> (B S) T C", T=self.d_t, S=self.d_s)
        B, TS, C = x.shape
        S = TS // T
        x = ops.reshape(x, (B, T, S, C))
        x = ops.transpose(x, (0, 2, 1, 3))
        x = ops.reshape(x, (B*S, T, C))
        return x

    @staticmethod
    def _rearrange_out_T(x, S):
        # x_t = rearrange(x_t, "(B S) T C -> B (T S) C", T=self.d_t, S=self.d_s)
        BS, T, C = x.shape
        B = BS // S
        x = ops.reshape(x, (B, S, T, C))
        x = ops.transpose(x, (0, 2, 1, 3))
        x = ops.reshape(x, (B, T*S, C))
        return x

    def construct(self, x, y, t, mask=None, tpe=None):
        B, N, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).chunk(6, axis=1)
        x_m = t2i_modulate(self.norm1(x), shift_msa, scale_msa)

        # spatial branch
        # x_s = rearrange(x_m, "B (T S) C -> (B T) S C", T=self.d_t, S=self.d_s)
        x_s = self._rearrange_in_S(x_m, T=self.d_t)
        x_s = self.attn(x_s)

        # x_s = rearrange(x_s, "(B T) S C -> B (T S) C", T=self.d_t, S=self.d_s)
        x_s = self._rearrange_out_S(x_s, T=self.d_t)
        x = x + self.drop_path(gate_msa * x_s)

        # temporal branch
        # x_t = rearrange(x, "B (T S) C -> (B S) T C", T=self.d_t, S=self.d_s)
        x_t = self._rearrange_in_T(x, T=self.d_t)
        if tpe is not None:
            x_t = x_t + tpe
        x_t = self.attn_temp(x_t)

        # x_t = rearrange(x_t, "(B S) T C -> B (T S) C", T=self.d_t, S=self.d_s)
        x_t = self._rearrange_out_T(x_t, S=self.d_s)
        x = x + self.drop_path(gate_msa * x_t)

        # cross attn
        x = x + self.cross_attn(x, y, mask)

        # mlp
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))

        return x


