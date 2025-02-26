import mindspore as ms
from mindspore import mint, nn, ops

# ----------------------------------------------------------
# RoPE2D: RoPE implementation in 2D
# ----------------------------------------------------------

try:
    from .curope import cuRoPE2D

    RoPE2D = cuRoPE2D
except ImportError:
    print("Warning, cannot find compiled version of RoPE2D, using a slow version instead")

    class RoPE2D(nn.Cell):
        def __init__(self, freq=10000.0, F0=1.0, scaling_factor=1.0):
            super().__init__()
            self.base = freq
            self.F0 = F0
            self.scaling_factor = scaling_factor
            self.cache = {}

        def get_cos_sin(self, D, seq_len, dtype):
            if (D, seq_len, dtype) not in self.cache:
                inv_freq = 1.0 / (self.base ** (ops.arange(0, D, 2).float() / D))
                t = ops.arange(0, seq_len, dtype=inv_freq.dtype)
                # freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
                freqs = ops.outer(t, inv_freq).to(dtype)
                freqs = mint.cat((freqs, freqs), dim=-1)
                cos = freqs.cos()  # (Seq, Dim)
                sin = freqs.sin()
                self.cache[D, seq_len, dtype] = (cos, sin)
            return self.cache[D, seq_len, dtype]

        @staticmethod
        def rotate_half(x):
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
            return mint.cat((-x2, x1), dim=-1)

        def apply_rope1d(self, tokens, pos1d, cos, sin):
            assert pos1d.ndim == 2

            cos = ops.embedding(pos1d, cos)[:, None, :, :]
            sin = ops.embedding(pos1d, sin)[:, None, :, :]
            return (tokens * cos) + (self.rotate_half(tokens) * sin)

        def construct(self, tokens, positions):
            """
            input:
                * tokens: batch_size x nheads x ntokens x dim
                * positions: batch_size x ntokens x 2 (y and x position of each token)
            output:
                * tokens after appplying RoPE2D (batch_size x nheads x ntokens x dim)
            """
            assert tokens.size(3) % 2 == 0, "number of dimensions should be a multiple of two"
            D = tokens.size(3) // 2
            assert positions.ndim == 3 and positions.shape[-1] == 2  # Batch, Seq, 2
            cos, sin = self.get_cos_sin(D, int(positions.max()) + 1, tokens.dtype)
            # split features into two along the feature dimension, and apply rope1d on each half
            y, x = tokens.chunk(2, axis=-1)
            y = self.apply_rope1d(y, positions[:, :, 0], cos, sin)
            x = self.apply_rope1d(x, positions[:, :, 1], cos, sin)
            tokens = mint.cat((y, x), dim=-1)
            return tokens


class LinearScalingRoPE2D(RoPE2D):
    """Code from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L148"""

    def construct(self, tokens, positions):
        # difference to the original RoPE: a scaling factor is aplied to the position ids
        dtype = positions.dtype
        positions = positions.float() / self.scaling_factor
        positions = positions.to(dtype)
        tokens = super().construct(tokens, positions)
        return tokens


try:
    from .curope import cuRoPE1D

    RoPE1D = cuRoPE1D
except ImportError:
    print("Warning, cannot find compiled version of RoPE2D, using a slow version instead")

    class RoPE1D(nn.Cell):
        def __init__(self, freq=10000.0, F0=1.0, scaling_factor=1.0):
            super().__init__()
            self.base = freq
            self.F0 = F0
            self.scaling_factor = scaling_factor
            self.cache = {}

        def get_cos_sin(self, D, seq_len, dtype):
            if (D, seq_len, dtype) not in self.cache:
                inv_freq = 1.0 / (self.base ** (ops.arange(0, D, 2).float() / D))
                t = ops.arange(0, seq_len, dtype=inv_freq.dtype)
                # freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
                freqs = ops.outer(t, inv_freq).to(dtype)
                freqs = mint.cat((freqs, freqs), dim=-1)
                cos = freqs.cos()  # (Seq, Dim)
                sin = freqs.sin()
                self.cache[D, seq_len, dtype] = (cos, sin)
            return self.cache[D, seq_len, dtype]

        @staticmethod
        def rotate_half(x):
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
            return mint.cat((-x2, x1), dim=-1)

        def apply_rope1d(self, tokens, pos1d, cos, sin):
            assert pos1d.ndim == 2
            cos = ops.embedding(pos1d, cos)[:, None, :, :]
            sin = ops.embedding(pos1d, sin)[:, None, :, :]
            return (tokens * cos) + (self.rotate_half(tokens) * sin)

        def construct(self, tokens, positions):
            """
            input:
                * tokens: batch_size x nheads x ntokens x dim
                * positions: batch_size x ntokens (t position of each token)
            output:
                * tokens after appplying RoPE2D (batch_size x nheads x ntokens x dim)
            """
            D = tokens.size(3)
            assert positions.ndim == 2  # Batch, Seq
            cos, sin = self.get_cos_sin(D, int(positions.max()) + 1, tokens.dtype)
            tokens = self.apply_rope1d(tokens, positions, cos, sin)
            return tokens


class LinearScalingRoPE1D(RoPE1D):
    """Code from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L148"""

    def construct(self, tokens, positions):
        # difference to the original RoPE: a scaling factor is aplied to the position ids
        dtype = positions.dtype
        positions = positions.float() / self.scaling_factor
        positions = positions.to(dtype)
        tokens = super().construct(tokens, positions)
        return tokens


class PositionGetter2D(object):
    """return positions of patches"""

    def __init__(self):
        self.cache_positions = {}

    def __call__(self, b, h, w):
        if not (h, w) in self.cache_positions:
            x = ops.arange(0, w)
            y = ops.arange(0, h)
            self.cache_positions[h, w] = ms_cartesian_prod(y, x)  # (h, w, 2)
        pos = self.cache_positions[h, w].reshape(1, h * w, 2)
        pos = ops.repeat_interleave(pos, b, axis=0).copy()
        return pos


def ms_cartesian_prod(*tensors):
    for tensor in tensors:
        assert len(tensor.shape) == 1, f"Accept 1-dim tensor as input! But got {len(tensor.shape)}-dim input"
    pools = [tuple(pool) for pool in tensors]
    result = [[]]
    for pool in pools:
        result = [x + [y] for x in result for y in pool]
    return ms.Tensor(result)


class PositionGetter1D(object):
    """return positions of patches"""

    def __init__(self):
        self.cache_positions = {}

    def __call__(self, b, length):
        if not (length) in self.cache_positions:
            x = ops.arange(0, length)
            self.cache_positions[length] = x  # (l, )
        pos = self.cache_positions[length].reshape(1, length)
        pos = ops.repeat_interleave(pos, b, axis=0).copy()
        return pos
