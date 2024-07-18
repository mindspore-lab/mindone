# modified from https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py
# and https://github.com/lucidrains/imagen-pytorch/blob/main/imagen_pytorch/imagen_pytorch.py
# and https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/resampler.py
import math

from mindspore import Parameter, nn, ops


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.SequentialCell(
        nn.LayerNorm([dim], epsilon=1e-05),
        nn.Dense(dim, inner_dim, has_bias=False),
        nn.GELU(),
        nn.Dense(inner_dim, dim, has_bias=False),
    )


def reshape_tensor(x, heads):
    bs, length, width = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.swapaxes(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x


class PerceiverAttention(nn.Cell):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm([dim], epsilon=1e-05)
        self.norm2 = nn.LayerNorm([dim], epsilon=1e-05)

        self.to_q = nn.Dense(dim, inner_dim, has_bias=False)
        self.to_kv = nn.Dense(dim, inner_dim * 2, has_bias=False)
        self.to_out = nn.Dense(inner_dim, dim, has_bias=False)

    def construct(self, x, latents):
        """
        Args:
            x (ms.Tensor): image features
                shape (b, n1, D)
            latent (ms.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = ops.cat((x, latents), axis=-2)
        k, v = self.to_kv(kv_input).chunk(2, axis=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).swapaxes(-2, -1)  # More stable with f16 than dividing afterwards
        weight = ops.softmax(weight.float(), axis=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)


class Resampler(nn.Cell):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
        video_length=None,  # using frame-wise version or not
    ):
        super().__init__()
        # queries for a single frame / image
        self.num_queries = num_queries
        self.video_length = video_length

        # <num_queries> queries for each frame
        if video_length is not None:
            num_queries = num_queries * video_length

        self.latents = Parameter(ops.randn(1, num_queries, dim) / dim**0.5)
        self.proj_in = nn.Dense(embedding_dim, dim)
        self.proj_out = nn.Dense(dim, output_dim)
        self.norm_out = nn.LayerNorm([output_dim], epsilon=1e-05)

        self.layers = nn.CellList([])
        for _ in range(depth):
            self.layers.append(
                nn.CellList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def construct(self, x):
        latents = self.latents.tile((x.shape[0], 1, 1))  # B (T L) C
        x = self.proj_in(x)

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        latents = self.proj_out(latents)
        latents = self.norm_out(latents)  # B L C or B (T L) C

        return latents
