from stepvideo.parallel import get_sequence_parallel_rank, get_sequence_parallel_world_size

from mindspore import mint, ops


class RoPE1D:
    def __init__(self, freq=1e4, F0=1.0, scaling_factor=1.0):
        self.base = freq
        self.F0 = F0
        self.scaling_factor = scaling_factor
        # self.cache = {}

    def get_cos_sin(self, D, seq_len, dtype):
        # w/ cache
        # if (D, seq_len, dtype) not in self.cache:
        #     inv_freq = 1.0 / (self.base ** (ops.arange(0, D, 2).float() / D))
        #     t = ops.arange(seq_len, dtype=inv_freq.dtype)
        #     freqs = ops.einsum("i,j->ij", t, inv_freq).to(dtype)
        #     freqs = ops.cat((freqs, freqs), axis=-1)
        #     cos = freqs.cos()  # (Seq, Dim)
        #     sin = freqs.sin()
        #     self.cache[D, seq_len, dtype] = (cos, sin)
        # return self.cache[D, seq_len, dtype]

        # w/o cache
        inv_freq = 1.0 / (self.base ** (mint.arange(0, D, 2).float() / D))
        t = mint.arange(seq_len, dtype=inv_freq.dtype)
        # freqs = ops.einsum("i,j->ij", t, inv_freq).to(dtype)
        freqs = (t[:, None] * inv_freq[None, :]).to(dtype)

        freqs = mint.cat((freqs, freqs), dim=-1)
        cos = freqs.cos()  # (Seq, Dim)
        sin = freqs.sin()
        return (cos, sin)

    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return mint.cat((-x2, x1), dim=-1)

    def apply_rope1d(self, tokens, pos1d, cos, sin):
        assert pos1d.ndim == 2
        cos = mint.nn.functional.embedding(pos1d, cos)[:, :, None, :]
        sin = mint.nn.functional.embedding(pos1d, sin)[:, :, None, :]
        return (tokens * cos) + (self.rotate_half(tokens) * sin)

    def __call__(self, tokens, positions):
        """
        input:
            * tokens: batch_size x ntokens x nheads x dim
            * positions: batch_size x ntokens (t position of each token)
        output:
            * tokens after appplying RoPE2D (batch_size x ntokens x nheads x dim)
        """
        D = tokens.shape[3]
        assert positions.ndim == 2  # Batch, Seq
        cos, sin = self.get_cos_sin(D, int(positions.max()) + 1, tokens.dtype)
        tokens = self.apply_rope1d(tokens, positions, cos, sin)
        return tokens


class RoPE3D(RoPE1D):
    def __init__(self, freq=1e4, F0=1.0, scaling_factor=1.0):
        super(RoPE3D, self).__init__(freq, F0, scaling_factor)
        # self.position_cache = {}

    def get_mesh_3d(self, rope_positions, bsz):
        f, h, w = rope_positions

        # w/ cache
        # if f"{f}-{h}-{w}" not in self.position_cache:
        #     x = torch.arange(f)
        #     y = torch.arange(h)
        #     z = torch.arange(w)
        #     self.position_cache[f"{f}-{h}-{w}"] = torch.cartesian_prod(x, y, z).view(1, f*h*w, 3).expand(bsz, -1, 3)
        # return self.position_cache[f"{f}-{h}-{w}"]

        # w/o cache
        x = mint.arange(f)
        y = mint.arange(h)
        z = mint.arange(w)
        return mint.broadcast_to(
            ops.cartesian_prod(x, y, z).view(1, f * h * w, 3), (bsz, -1, 3)
        )  # FIXME: ops.cartesian_prod

    def __call__(self, tokens, rope_positions, ch_split, parallel=False):
        """
        input:
            * tokens: batch_size x ntokens x nheads x dim
            * rope_positions: list of (f, h, w)
        output:
            * tokens after appplying RoPE2D (batch_size x ntokens x nheads x dim)
        """
        assert sum(ch_split) == tokens.shape[-1]

        mesh_grid = self.get_mesh_3d(rope_positions, bsz=tokens.shape[0])
        out = []
        for i, (D, x) in enumerate(zip(ch_split, mint.split(tokens, ch_split, dim=-1))):
            cos, sin = self.get_cos_sin(D, int(mesh_grid.max()) + 1, tokens.dtype)

            if parallel:
                mesh = mint.chunk(mesh_grid[:, :, i], get_sequence_parallel_world_size(), dim=1)[
                    get_sequence_parallel_rank()
                ]
            else:
                mesh = mesh_grid[:, :, i]
            x = self.apply_rope1d(x, mesh, cos, sin)
            out.append(x)

        tokens = mint.cat(out, dim=-1)
        return tokens
