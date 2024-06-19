"""
Source: https://github.com/lucidrains/rotary-embedding-torch/
"""
from math import pi
from typing import Optional, Tuple

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # FIXME: python 3.7

import numpy as np

import mindspore as ms
from mindspore import Parameter, Tensor, dtype, nn, ops


def rotate_half(x: Tensor) -> Tensor:
    x = x.reshape(x.shape[:-1] + (-1, 2))  # ... (d r) -> ... d r, r = 2
    x1, x2 = x.chunk(2, axis=-1)
    x = ops.concat((-x2, x1), axis=-1)
    return x.reshape(x.shape[:-2] + (-1,))  # '... d r -> ... (d r)'


def apply_rotary_emb(
    freqs: Parameter, t: Tensor, start_index: int = 0, scale: float = 1.0, seq_dim: int = -2
) -> Tensor:
    if t.ndim == 3:
        seq_len = t.shape[seq_dim]
        freqs = freqs[-seq_len:].dtype(t.dtype)

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    assert (
        rot_dim <= t.shape[-1]
    ), f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"

    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t = (t * freqs.cos().astype(t.dtype) * scale) + (rotate_half(t) * freqs.sin().astype(t.dtype) * scale)
    return ops.cat((t_left, t, t_right), axis=-1)


class RotaryEmbedding(nn.Cell):
    """
    Rotary Position Embedding (RoPE).
    """

    def __init__(
        self,
        dim: int,
        custom_freqs: Optional[Tensor] = None,
        freqs_for: Literal["lang", "pixel", "constant"] = "lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
        learned_freq=False,
        use_xpos=False,
        xpos_scale_base=512,
        interpolate_factor=1.0,
        theta_rescale_factor=1.0,
        seq_before_head_dim=False,
        cache_if_possible=False,
    ):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/

        theta *= theta_rescale_factor ** (dim / (dim - 2))

        self.freqs_for = freqs_for

        if custom_freqs is not None:
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)].astype(np.float32) / dim))
        elif freqs_for == "pixel":
            freqs = np.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "constant":
            freqs = np.ones(num_freqs)
        else:
            raise ValueError(f"Invalid freqs_for: {freqs_for}")

        if cache_if_possible:
            raise NotImplementedError("Cache is not supported")

        self.freqs = Parameter(Tensor(freqs, dtype=dtype.float32), requires_grad=learned_freq)
        self.learned_freq = learned_freq

        # default sequence dimension
        self.seq_before_head_dim = seq_before_head_dim
        self.default_seq_dim = -3 if seq_before_head_dim else -2

        # interpolation factors
        assert interpolate_factor >= 1.0
        self.interpolate_factor = interpolate_factor

        # xpos
        self.use_xpos = use_xpos
        self.scale = None
        if use_xpos:
            self.scale = Tensor((np.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim), dtype=dtype.float32)
            self.scale_base = xpos_scale_base

    def get_seq_pos(self, seq_len, dtype, offset=0):
        return (ops.arange(seq_len, dtype=dtype) + offset) / self.interpolate_factor

    def rotate_queries_or_keys(self, t: Tensor, seq_dim=None, offset=0, freq_seq_len=None) -> Tensor:
        """
        Args:
            t: tensor of shape (b n h d)
        """
        t = t.swapaxes(1, 2)  # the expected tensor shape is (b h n d), but the input shape is (b n h d)
        seq_dim = seq_dim or self.default_seq_dim

        if self.use_xpos:
            raise ValueError(
                "you must use `.rotate_queries_and_keys` method instead and pass in both queries and keys,"
                " for length extrapolatable rotary embeddings"
            )

        dtype, seq_len = t.dtype, t.shape[seq_dim]

        if freq_seq_len is not None:
            assert freq_seq_len >= seq_len
            seq_len = freq_seq_len

        freqs = self.construct(self.get_seq_pos(seq_len, dtype=dtype, offset=offset), seq_len=seq_len, offset=offset)

        if seq_dim == -3:
            freqs = freqs.unsqueeze(1)  # n d -> n 1 d

        return apply_rotary_emb(freqs, t, seq_dim=seq_dim).swapaxes(1, 2)  # (b h n d) -> (b n h d)

    def rotate_queries_with_cached_keys(self, q, k, seq_dim=None, offset=0):
        raise NotImplementedError

    def rotate_queries_and_keys(self, q, k, seq_dim=None):
        raise NotImplementedError

    def get_scale(self, t: Tensor, seq_len: Optional[int] = None, offset=0):
        raise NotImplementedError

    def get_axial_freqs(self, *dims):
        raise NotImplementedError

    def construct(self, t: Tensor, seq_len=None, offset=0) -> Tensor:
        freqs = t.astype(self.freqs.dtype)[..., None] * self.freqs
        return freqs.repeat(2, axis=-1)  # ... n -> ... (n r), r = 2


def rope_1d(x: Tensor, freqs_cis: Tensor) -> Tensor:
    dtype = x.dtype
    x = x.to(ms.float32)
    x = ops.transpose(x, (0, 2, 1, 3))  # b h n d
    freqs_cis = freqs_cis[:, None, ...]  # b(1) 1 n d
    sin_matrix = ops.sin(freqs_cis)
    cos_matrix = ops.cos(freqs_cis)
    cos_part = ops.mul(x, cos_matrix)
    sin_part = ops.mul(rotate_half(x), sin_matrix)

    x = cos_part + sin_part
    x = ops.transpose(x, (0, 2, 1, 3))  # b n h d
    return x.to(dtype)


def precompute_freqs_cis(seq_len: int, dim: int, theta: float = 10000.0) -> np.ndarray:
    positional_ids = np.arange(seq_len, dtype=np.float32)
    indices = 1.0 / np.power(theta, 2 * np.arange(dim // 2, dtype=np.float32) / dim)
    embeddings = np.outer(positional_ids, indices)
    embeddings = np.repeat(embeddings, 2, axis=-1)
    return embeddings

class RotaryEmbeddingSP(nn.Cell):
    def __init__(
        self,
        dim: int,
        theta: float = 10000.0,
        ratio: float = 1.0,
        k: float = 1.0,
        dp: int = 1,
        mp: int = 1,
        sp: int = 1,
    ):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.ratio = ratio
        self.k = k

        self.dp = dp
        self.mp = mp
        self.sp = sp

        self.transpose = ops.Transpose()
        self.mul = ops.Mul()
        self.split = ops.Split(axis=-1, output_num=2)
        self.concat = ops.Concat(axis=-1)
        self.neg = ops.Neg()
        self.add = ops.Add()
        self.transpose_2 = ops.Transpose()
        self.shard()

    def rotate_half(self, x: Tensor, shape: Tuple[int, ...]):
        x = ops.reshape(x, (x.shape[0], x.shape[1], x.shape[2], -1, 2))
        x0, x1 = self.split(x)
        x = self.concat([self.neg(x1), x0])
        x = ops.reshape(x, shape)
        return x

    def construct(self, x: Tensor):
        dtype = x.dtype
        x = x.to(ms.float32)
        x = self.transpose(x, (0, 2, 1, 3))  # b h n d
        shape = x.shape
        _, _, n, d = shape
        sin_matrix, cos_matrix = self.cal_matrix(n, d)
        cos_part = self.mul(x, cos_matrix)
        sin_part = self.mul(self.rotate_half(x, shape), sin_matrix)
        x = self.add(cos_part, sin_part)
        x = self.transpose_2(x, (0, 2, 1, 3))  # b n h d
        return x.to(dtype)

    def shard(self):
        self.transpose.shard(((self.dp, self.sp, self.mp, 1),))
        self.mul.shard(((self.dp, self.mp, self.sp, 1), (1, 1, self.sp, 1)))
        self.split.shard(((self.dp, self.mp, self.sp, 1, 1),))
        self.concat.shard(((self.dp, self.mp, self.sp, 1, 1), (self.dp, self.mp, self.sp, 1, 1)))
        self.neg.shard(((self.dp, self.mp, self.sp, 1, 1),))
        self.add.shard(((self.dp, self.mp, self.sp, 1), (self.dp, self.mp, self.sp, 1)))
        self.transpose_2.shard(((self.dp, self.mp, self.sp, 1),))

    def cal_matrix(self, seq_len: int, dim: int):
        scaled_seq_len = int(seq_len * self.ratio)

        positional_ids = ops.arange(scaled_seq_len, dtype=ms.float32) / self.ratio
        indices = 1.0 / ops.pow(self.theta * self.k, 2 * ops.arange(dim // 2, dtype=ms.float32) / dim)

        embeddings = ops.outer(positional_ids, indices)
        embeddings = ops.repeat_interleave(embeddings, 2, axis=-1)
        embeddings = ops.reshape(embeddings, (1, 1, scaled_seq_len, dim))

        sin_matrix = ops.sin(embeddings)
        cos_matrix = ops.cos(embeddings)
        return sin_matrix, cos_matrix
