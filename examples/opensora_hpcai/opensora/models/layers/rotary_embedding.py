"""
Source: https://github.com/lucidrains/rotary-embedding-torch/
"""
from math import pi
from typing import Optional

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # FIXME: python 3.7

import numpy as np

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
