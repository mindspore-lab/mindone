import mindspore as ms
from mindspore import nn, ops

from .attention import FeedForward, LayerNorm, is_old_ms_version


class PreNormattention(nn.Cell):
    def __init__(self, dim, fn, dtype=ms.float32):
        super().__init__()
        self.dtype = dtype
        self.norm = LayerNorm(
            [
                dim,
            ],
            epsilon=1e-05,
        )
        self.fn = fn

    def construct(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs) + x


class PostNormattention(nn.Cell):
    def __init__(self, dim, fn, dtype=ms.float32):
        super().__init__()
        self.dtype = dtype
        self.norm = LayerNorm(
            [
                dim,
            ],
            epsilon=1e-05,
        )
        self.fn = fn

    def construct(self, x, **kwargs):
        return self.norm(self.fn(x, **kwargs) + x)


class Attention(nn.Cell):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, dtype=ms.float32):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5
        self.dtype = dtype

        self.to_qkv = nn.Dense(dim, inner_dim * 3, has_bias=False).to_float(self.dtype)

        self.to_out = (
            nn.SequentialCell(
                nn.Dense(inner_dim, dim).to_float(self.dtype),
                nn.Dropout(1 - dropout) if is_old_ms_version() else nn.Dropout(p=dropout),
            )
            if project_out
            else nn.Identity()
        )

    def rearrange_qkv(self, tensor: ms.Tensor):
        # b n (h d) -> b n h d -> b h n d
        tensor = ops.reshape(tensor, (tensor.shape[0], tensor.shape[1], self.heads, tensor.shape[2] // self.heads))
        tensor = ops.transpose(tensor, (0, 2, 1, 3))
        return tensor

    def construct(self, x):
        qkv = self.to_qkv(x).chunk(3, axis=-1)
        q, k, v = [self.rearrange_qkv(i) for i in qkv]
        dots = ops.bmm(q, k.transpose(0, 1, 3, 2)) * self.scale
        attn = ops.softmax(dots.to(ms.float32)).to(self.dtype)
        out = ops.bmm(attn, v)
        # b h n d -> b n h d -> b n (h d)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (out.shape[0], out.shape[1], -1))
        return self.to_out(out)


class Transformer_v2(nn.Cell):
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
        dtype=ms.float32,
    ):
        super().__init__()
        layers = []
        self.depth = depth
        self.dtype = dtype
        for _ in range(depth):
            layers.append(
                nn.CellList(
                    [
                        PreNormattention(
                            dim,
                            Attention(dim, heads=heads, dim_head=dim_head_k, dropout=dropout_atte, dtype=self.dtype),
                            dtype=self.dtype,
                        ),
                        FeedForward(dim, mlp_dim, dropout=dropout_ffn, dtype=self.dtype),
                    ]
                )
            )
        self.layers = nn.CellList(layers)

    def construct(self, x):
        # if self.depth
        for attn, ff in self.layers[:1]:
            x = attn(x)
            x = ff(x) + x
        if self.depth > 1:
            for attn, ff in self.layers[1:]:
                x = attn(x)
                x = ff(x) + x
        return x
