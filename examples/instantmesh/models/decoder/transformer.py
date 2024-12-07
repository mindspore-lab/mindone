import mindspore as ms
import mindspore.nn as nn
from mindspore import mint


class BasicTransformerBlock(nn.Cell):
    """
    Transformer block that takes in a cross-attention condition and another modulation vector applied to sub-blocks.
    """

    # use attention from torch.nn.MultiHeadAttention
    # Block contains a cross-attention layer, a self-attention layer, and a MLP
    def __init__(
        self,
        inner_dim: int,
        cond_dim: int,
        num_heads: int,
        eps: float,
        attn_drop: float = 0.0,
        attn_bias: bool = False,
        mlp_ratio: float = 4.0,
        mlp_drop: float = 0.0,
        dtype: ms.dtype = ms.float32,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm([inner_dim])
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=inner_dim,
            num_heads=num_heads,
            kdim=cond_dim,
            vdim=cond_dim,
            dropout=attn_drop,
            has_bias=attn_bias,
            batch_first=True,
            dtype=dtype,
        )
        self.norm2 = nn.LayerNorm([inner_dim])
        self.self_attn = nn.MultiheadAttention(
            embed_dim=inner_dim,
            num_heads=num_heads,
            dropout=attn_drop,
            has_bias=attn_bias,
            batch_first=True,
            dtype=dtype,
        )
        self.norm3 = nn.LayerNorm([inner_dim])
        self.mlp = nn.SequentialCell(
            nn.Dense(inner_dim, int(inner_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(p=mlp_drop),
            nn.Dense(int(inner_dim * mlp_ratio), inner_dim),
            nn.Dropout(p=mlp_drop),
        )

    def construct(self, x, cond):
        # x: [N, L, D]
        # cond: [N, L_cond, D_cond]
        x = x + self.cross_attn(self.norm1(x), cond, cond)[0]
        before_sa = self.norm2(x)
        x = x + self.self_attn(before_sa, before_sa, before_sa)[0]
        x = x + self.mlp(self.norm3(x))
        return x


class TriplaneTransformer(nn.Cell):
    """
    Transformer with condition that generates a triplane representation.

    Reference:
    Timm: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L486
    """

    def __init__(
        self,
        inner_dim: int,
        image_feat_dim: int,
        triplane_low_res: int,
        triplane_high_res: int,
        triplane_dim: int,
        num_layers: int,
        num_heads: int,
        eps: float = 1e-6,
        dtype: ms.dtype = ms.float32,
        use_recompute: bool = False,
    ):
        super().__init__()

        # attributes
        self.triplane_low_res = triplane_low_res
        self.triplane_high_res = triplane_high_res
        self.triplane_dim = triplane_dim

        # modules
        # initialize pos_embed with 1/sqrt(dim) * N(0, 1)
        self.pos_embed = ms.Parameter(
            mint.normal(size=(1, 3 * triplane_low_res**2, inner_dim)) * (1.0 / inner_dim) ** 0.5
        )  # [L, D]
        self.layers = nn.CellList(
            [
                BasicTransformerBlock(
                    inner_dim=inner_dim, cond_dim=image_feat_dim, num_heads=num_heads, eps=eps, dtype=dtype
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm([inner_dim], epsilon=eps)
        self.deconv = nn.Conv2dTranspose(
            inner_dim, triplane_dim, kernel_size=2, stride=2, padding=0, pad_mode="valid", has_bias=True
        )

        if use_recompute:
            for b in self.layers:
                b.recompute()
            self.norm.recompute()
            self.deconv.recompute()

    # @ms.jit
    def construct(self, image_feats):
        # image_feats: [N, L_cond, D_cond]

        N = image_feats.shape[0]
        H = W = self.triplane_low_res
        # L = 3 * H * W

        x = self.pos_embed.tile((N, 1, 1))  # [N, L, D]
        for layer in self.layers:
            x = layer(x, image_feats)
        x = self.norm(x)

        # separate each plane and apply deconv
        x = x.view(N, 3, H, W, -1)
        x = mint.permute(x, (1, 0, 4, 2, 3))  # [3, N, D, H, W]
        x = x.contiguous().view(3 * N, -1, H, W)  # [3*N, D, H, W]
        x = self.deconv(x)  # [3*N, D', H', W']
        x = x.view(3, N, *x.shape[-3:])  # [3, N, D', H', W']
        x = mint.permute(x, (1, 0, 2, 3, 4))  # [N, 3, D', H', W']
        x = x.contiguous()

        return x
