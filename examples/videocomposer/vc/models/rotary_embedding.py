from math import pi

import mindspore as ms
from mindspore import nn, ops

# helper functions


def exists(val):
    return val is not None


def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
    shape_len = list(shape_lens)[0]

    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))

    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all(
        [*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]
    ), "invalid dimensions for broadcastable concatentation"
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return ops.cat(tensors, axis=dim)


# rotary embedding helper functions


def rotate_half(x):
    b, dr = x.shape[:-1], x.shape[-1]
    d, r = dr // 2, 2
    x = ops.reshape(x, (*b, d, r))
    x1, x2 = x.unbind(dim=-1)
    x = ops.stack((-x2, x1), axis=-1)
    return ops.reshape(x, (*b, dr))


def apply_rotary_emb(freqs, t, start_index=0, scale=1.0):
    freqs = freqs.to(t.dtype)
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    assert (
        rot_dim <= t.shape[-1]
    ), f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"
    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    return ops.cat((t_left, t, t_right), axis=-1)


# learned rotation helpers


def apply_learned_rotations(rotations, t, start_index=0, freq_ranges=None):
    if exists(freq_ranges):
        rotations = ops.einsum("..., f -> ... f", rotations, freq_ranges)
        b, r, f = rotations.shape[:-2], rotations.shape[-2], rotations.shape[-1]
        rotations = ops.reshape(rotations, (*b, r * f))

    rotations = ops.tile(rotations, (*(1,) * (rotations.ndim - 1), 2))
    return apply_rotary_emb(rotations, t, start_index=start_index)


# classes


class RotaryEmbedding(nn.Cell):
    def __init__(
        self,
        dim,
        custom_freqs=None,
        freqs_for="lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
        learned_freq=False,
        use_xpos=False,
        xpos_scale_base=512,
        interpolate_factor=1.0,
        theta_rescale_factor=1.0,
    ):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
        theta *= theta_rescale_factor ** (dim / (dim - 2))

        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (theta ** (ops.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        elif freqs_for == "pixel":
            freqs = ops.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "constant":
            freqs = ops.ones(num_freqs).float()
        else:
            raise ValueError(f"unknown modality {freqs_for}")

        self.cache = dict()
        self.cache_scale = dict()
        self.freqs = ms.Parameter(freqs, requires_grad=learned_freq)

        # interpolation factors
        assert interpolate_factor >= 1.0
        self.interpolate_factor = interpolate_factor

        # xpos
        self.use_xpos = use_xpos
        if not use_xpos:
            # todo: self.register_buffer("scale", None), mindspore.Parameter does not support NoneType.
            self.scale = None
            return

        scale = (ops.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.scale_base = xpos_scale_base
        self.scale = ms.Parameter(scale, requires_grad=False)

    def get_seq_pos(self, seq_len, dtype, offset=0):
        return (ops.arange(seq_len, dtype=dtype) + offset) / self.interpolate_factor

    def rotate_queries_or_keys(self, t, seq_dim=-2, offset=0):
        assert (
            not self.use_xpos
        ), "you must use `.rotate_queries_and_keys` method instead and pass in both queries and keys, for length extrapolatable rotary embeddings"
        dtype, seq_len = t.dtype, t.shape[seq_dim]
        freqs = self.construct(
            lambda: self.get_seq_pos(seq_len, dtype=dtype, offset=offset),
            cache_key=f"freqs:{seq_len}|offset:{offset}",
        )
        return apply_rotary_emb(freqs, t)

    def rotate_queries_and_keys(self, q, k, seq_dim=-2):
        assert self.use_xpos
        dtype, seq_len = q.dtype, q.shape[seq_dim]
        seq = self.get_seq_pos(seq_len, dtype=dtype)
        freqs = self.construct(lambda: seq, cache_key=f"freqs:{seq_len}")
        scale = self.get_scale(lambda: seq, cache_key=f"scale:{seq_len}").to(dtype)
        rotated_q = apply_rotary_emb(freqs, q, scale=scale)
        rotated_k = apply_rotary_emb(freqs, k, scale=scale**-1)
        return rotated_q, rotated_k

    def get_scale(self, t, cache_key=None):
        assert self.use_xpos

        if exists(cache_key) and cache_key in self.cache:
            return self.cache[cache_key]

        if callable(t):
            t = t()

        scale = 1.0
        if self.use_xpos:
            power = (t - len(t) // 2) / self.scale_base
            scale = self.scale ** ops.expand_dims(power, -1)
            scale = ops.cat((scale, scale), axis=-1)

        if exists(cache_key):
            self.cache[cache_key] = scale

        return scale

    def construct(self, t, cache_key=None):
        if exists(cache_key) and cache_key in self.cache:
            return self.cache[cache_key]

        if callable(t):
            t = t()

        freqs = self.freqs

        freqs = ops.einsum("..., f -> ... f", t.astype(freqs.dtype), freqs)
        freqs = ops.tile(freqs, (*(1,) * (freqs.ndim - 1), 2))

        if exists(cache_key):
            self.cache[cache_key] = freqs

        return freqs
