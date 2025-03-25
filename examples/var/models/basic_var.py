import mindspore as ms
import mindspore.mint.nn.functional as F
from mindspore import Parameter, mint, nn
from mindspore.ops.operations.nn_ops import FlashAttentionScore

from .helpers import DropPath

# this file only provides the 3 blocks used in VAR transformer
__all__ = ["FFN", "AdaLNSelfAttn", "AdaLNBeforeHead"]


class FFN(nn.Cell):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = mint.nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate=True)
        self.fc2 = mint.nn.Linear(hidden_features, out_features)
        self.drop = mint.nn.Dropout(drop, inplace=True) if drop > 0 else mint.nn.Identity()

    def construct(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class SelfAttention(nn.Cell):
    def __init__(
        self,
        block_idx,
        embed_dim=768,
        num_heads=12,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_l2_norm=False,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.block_idx, self.num_heads, self.head_dim = block_idx, num_heads, embed_dim // num_heads  # =64
        self.attn_l2_norm = attn_l2_norm
        if self.attn_l2_norm:
            self.scale = 1.0
            self.scale_mul_1H11 = Parameter(
                mint.full(size=(1, self.num_heads, 1, 1), fill_value=4.0).log(), requires_grad=True
            )
            self.max_scale_mul = mint.log(ms.tensor(100)).item()
        else:
            self.scale = 0.25 / mint.sqrt(self.head_dim)

        self.mat_qkv = mint.nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.q_bias, self.v_bias = Parameter(mint.zeros(embed_dim)), Parameter(mint.zeros(embed_dim))
        self.register_buffer("zero_k_bias", mint.zeros(embed_dim))

        self.proj = mint.nn.Linear(embed_dim, embed_dim)
        self.proj_drop = mint.nn.Dropout(proj_drop, inplace=True) if proj_drop > 0 else mint.nn.Identity()
        self.attn_drop: float = attn_drop

        # only used during inference
        self.caching, self.cached_k, self.cached_v = False, None, None
        self.attention = FlashAttentionScore(
            head_num=self.num_heads, scale_value=self.scale, input_layout="BNSD", keep_prob=1 - self.attn_drop
        )

    def register_buffer(self, name, attr):
        setattr(self, name, Parameter(default_input=attr, requires_grad=False))

    def kv_caching(self, enable: bool):
        self.caching, self.cached_k, self.cached_v = enable, None, None

    # NOTE: attn_bias is None during inference because kv cache is enabled
    def construct(self, x, attn_bias):
        B, L, C = x.shape

        qkv = F.linear(
            input=x, weight=self.mat_qkv.weight, bias=mint.cat((self.q_bias, self.zero_k_bias, self.v_bias))
        ).view((B, L, 3, self.num_heads, self.head_dim))
        # qkv: BL3Hc

        q, k, v = qkv.unbind(dim=2)  # q or k or v: BLHc
        dim_cat = 1

        if self.attn_l2_norm:
            scale_mul = self.scale_mul_1H11.clamp(max=self.max_scale_mul).exp()
            scale_mul = mint.transpose(scale_mul, 1, 2)  # 1H11 to 11H1
            q = F.normalize(q, dim=-1).mul(scale_mul)
            k = F.normalize(k, dim=-1)

        if self.caching:
            if self.cached_k is None:
                self.cached_k = k
                self.cached_v = v
            else:
                k = self.cached_k = mint.cat((self.cached_k, k), dim=dim_cat)
                v = self.cached_v = mint.cat((self.cached_v, v), dim=dim_cat)
        attention_mask = None if attn_bias is None else attn_bias.bool().expand((B, self.num_heads, -1, -1))
        # dropout_p = self.attn_drop if self.training else 0.0
        q = q.swapaxes(1, 2)
        k = k.swapaxes(1, 2)
        v = v.swapaxes(1, 2)
        out = self.attention(q, k, v, None, None, None, attention_mask)[3].swapaxes(1, 2).view((B, L, C))

        return self.proj_drop(self.proj(out))


class AdaLNSelfAttn(nn.Cell):
    def __init__(
        self,
        block_idx,
        last_drop_p,
        embed_dim,
        cond_dim,
        shared_aln: bool,
        norm_layer,
        num_heads,
        mlp_ratio=4.0,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        attn_l2_norm=False,
    ):
        super(AdaLNSelfAttn, self).__init__()
        self.block_idx, self.last_drop_p, self.C = block_idx, last_drop_p, embed_dim
        self.C, self.D = embed_dim, cond_dim
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else mint.nn.Identity()
        self.attn = SelfAttention(
            block_idx=block_idx,
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=drop,
            attn_l2_norm=attn_l2_norm,
        )
        self.ffn = FFN(in_features=embed_dim, hidden_features=round(embed_dim * mlp_ratio), drop=drop)

        self.ln_wo_grad = norm_layer(embed_dim, elementwise_affine=False)
        self.shared_aln = shared_aln
        if self.shared_aln:
            self.ada_gss = Parameter(mint.randn(1, 1, 6, embed_dim) / embed_dim**0.5)
        else:
            lin = mint.nn.Linear(cond_dim, 6 * embed_dim)
            self.ada_lin = nn.SequentialCell(mint.nn.SiLU(), lin)

        self.fused_add_norm_fn = None

    # NOTE: attn_bias is None during inference because kv cache is enabled
    def construct(self, x, cond_BD, attn_bias):  # C: embed_dim, D: cond_dim
        if self.shared_aln:
            gamma1, gamma2, scale1, scale2, shift1, shift2 = (self.ada_gss + cond_BD).unbind(
                2
            )  # 116C + B16C =unbind(2)=> 6 B1C
        else:
            gamma1, gamma2, scale1, scale2, shift1, shift2 = self.ada_lin(cond_BD).view((-1, 1, 6, self.C)).unbind(2)
        x = x + self.drop_path(
            self.attn(self.ln_wo_grad(x).mul(scale1.add(1)).add(shift1), attn_bias=attn_bias).mul(gamma1)
        )
        x = x + self.drop_path(
            self.ffn(self.ln_wo_grad(x).mul(scale2.add(1)).add(shift2)).mul(gamma2)
        )  # this mul(gamma2) cannot be in-placed when FusedMLP is used
        return x

    def extra_repr(self) -> str:
        return f"shared_aln={self.shared_aln}"


class AdaLNBeforeHead(nn.Cell):
    def __init__(self, C, D, norm_layer):  # C: embed_dim, D: cond_dim
        super().__init__()
        self.C, self.D = C, D
        self.ln_wo_grad = norm_layer(C, elementwise_affine=False)
        self.ada_lin = nn.SequentialCell(mint.nn.SiLU(), mint.nn.Linear(D, 2 * C))

    def construct(self, x_BLC: ms.Tensor, cond_BD: ms.Tensor):
        scale, shift = self.ada_lin(cond_BD).view((-1, 1, 2, self.C)).unbind(2)
        return self.ln_wo_grad(x_BLC).mul(scale.add(1)).add(shift)
