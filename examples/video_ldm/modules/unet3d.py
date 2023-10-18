from mindspore import nn, Parameter, Tensor
from ldm.modules.diffusionmodules.openaimodel import UNetModel


class Conv3DLayer(nn.Cell):
    def __init__(self, in_channels: int, out_channels: int, num_frames: int):
        super().__init__()
        self._num_frames = num_frames

        self.conv3d = nn.SequentialCell(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=(3, 1, 1),
                padding=(1, 1, 0, 0, 0, 0),
                pad_mode="pad",
                has_bias=True,
            ),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=(3, 1, 1),
                padding=(1, 1, 0, 0, 0, 0),
                pad_mode="pad",
                has_bias=True,
            ),
        )

        self.alpha = Parameter(1.0)

    def construct(self, x: Tensor) -> Tensor:
        h = x.reshape(-1, self._num_frames, x.shape[1], x.shape[2], x.shape[3])  # (b t) c h w -> b c t h w
        h = self.conv3d(h)
        h = h.reshape(-1, x.shape[1], x.shape[2], x.shape[3])  # b c t h w -> (b t) c h w

        # TODO: limit alpha with no grad
        return self.alpha * x + (1 - self.alpha) * h


class TemporalAttentionLayer(nn.Cell):
    # TODO: replace with ldm.modules.attention.CrossAttention
    def __init__(self, dim, num_frames: int, num_heads: int = 8, kv_dim=None):
        super().__init__()
        self.n_frames = num_frames
        self.n_heads = num_heads

        head_dim = dim // num_heads
        proj_dim = head_dim * num_heads
        self.q_proj = nn.Dense(dim, proj_dim, has_bias=False)

        kv_dim = kv_dim or dim
        self.k_proj = nn.Dense(kv_dim, proj_dim, has_bias=False)
        self.v_proj = nn.Dense(kv_dim, proj_dim, has_bias=False)
        self.o_proj = nn.Dense(proj_dim, dim, has_bias=False)

        # self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True, has_bias=True)
        # TODO: self.pos_enc = PositionalEncoding(dim)
        self.alpha = Parameter(1.0)

    def forward(self, q: Tensor, kv=None, mask=None):
        skip = q

        bt, c, h, w = q.shape
        # (b t) c h w -> b (h w) t c
        q = q.reshape(-1, self.n_frames, c, h, w)
        q = q.traspose(0, 3, 4, 1, 2).reshape(q.shape[0], h * w, q.shape[1], q.shape[2])

        # q = q + self.pos_enc(self.n_frames)

        kv = kv[:: self.n_frames] if kv is not None else q
        q = self.q_proj(q)
        k = self.k_proj(kv)
        v = self.v_proj(kv)

        # q = rearrange(q, "b hw t (heads d) -> b hw heads t d", heads=self.n_heads)
        # k = rearrange(k, "b s (heads d) -> b heads s d", heads=self.n_heads)
        # v = rearrange(v, "b s (heads d) -> b heads s d", heads=self.n_heads)

        # out = F.scaled_dot_product_attention(q, k, v, mask)
        # out = rearrange(out, "b hw heads t d -> b hw t (heads d)")
        # out = self.o_proj(out)

        # out = rearrange(out, "b (h w) t c -> (b t) c h w", h=h, w=w)

        # TODO: limit alpha with no grad
        out = q
        return self.alpha * skip + (1 - self.alpha) * out


class VideoLDMUNetModel(UNetModel):
    def __init__(self, num_frames=5, **kwargs):
        super().__init__(**kwargs)

        # inject 3D convolutions and temporal attention into input and output blocks
        for blocks in [self.input_blocks, self.output_blocks]:
            for i in range(len(blocks)):
                names = [block.cls_name for block in blocks[i]]
                if "ResBlock" in names and "SpatialTransformer" in names:
                    out_channels = blocks[i][0].out_channels
                    blocks[i].insert(1, Conv3DLayer(out_channels, out_channels, num_frames))
                    blocks[i].append(TemporalAttentionLayer(out_channels, num_frames, kv_dim=kwargs["context_dim"]))
