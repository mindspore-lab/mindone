"""
Finite Scalar Quantization: VQ-VAE Made Simple - https://arxiv.org/abs/2309.15505
Code adapted from Jax version in Appendix A.1
"""

from __future__ import annotations

from functools import wraps
from typing import List, Tuple

from einops import pack, unpack

import mindspore as ms
from mindspore import mint, nn

# helper functions


def exists(v):
    return v is not None


def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None


def maybe(fn):
    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)

    return inner


def pack_one(t, pattern):
    t = t.numpy()
    return ms.tensor(pack([t], pattern))


def unpack_one(t, ps, pattern):
    t = t.numpy()
    ps = ps.numpy()
    return ms.tensor(unpack(t, ps, pattern)[0])


# tensor helpers


def round_ste(z: ms.Tensor) -> ms.Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z)


# main class


class FSQ(nn.Cell):
    def __init__(
        self,
        levels: List[int],
        dim: int | None = None,
        num_codebooks=1,
        keep_num_codebooks_dim: bool | None = None,
        scale: float | None = None,
        allowed_dtypes: Tuple[ms.dtype, ...] = (ms.float32, ms.float64),
        channel_first: bool = False,
        projection_has_bias: bool = True,
        return_indices=True,
        force_quantization_f32=True,
    ):
        super().__init__()
        _levels = ms.tensor(list(levels), dtype=ms.int32)
        self._levels = _levels

        _basis = mint.cumprod(ms.tensor([1] + levels[:-1]), dim=0, dtype=ms.int32)
        self._basis = _basis

        self.scale = scale

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)

        self.channel_first = channel_first

        has_projections = self.dim != effective_codebook_dim
        self.project_in = (
            mint.nn.Linear(self.dim, effective_codebook_dim, bias=projection_has_bias)
            if has_projections
            else nn.Identity()
        )
        self.project_out = (
            mint.nn.Linear(effective_codebook_dim, self.dim, bias=projection_has_bias)
            if has_projections
            else nn.Identity()
        )

        self.has_projections = has_projections

        self.return_indices = return_indices
        if return_indices:
            self.codebook_size = self._levels.prod().item()
            implicit_codebook = self._indices_to_codes(mint.arange(self.codebook_size))
            self.implicit_codebook = implicit_codebook

        self.allowed_dtypes = allowed_dtypes
        self.force_quantization_f32 = force_quantization_f32

    def bound(self, z, eps: float = 1e-3):
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = mint.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z):
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2  # Renormalize to [-1, 1].
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized):
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat):
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def _indices_to_codes(self, indices):
        level_indices = self.indices_to_level_indices(indices)
        codes = self._scale_and_shift_inverse(level_indices)
        return codes

    def codes_to_indices(self, zhat):
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(ms.int32)

    def indices_to_level_indices(self, indices):
        """Converts indices to indices at each level, perhaps needed for a transformer with factorized embeddings"""
        indices = indices.unsqueeze(-1)
        codes_non_centered = (indices // self._basis) % self._levels
        return codes_non_centered

    def indices_to_codes(self, indices):
        """Inverse of `codes_to_indices`."""
        assert exists(indices)

        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        codes = self._indices_to_codes(indices)

        if self.keep_num_codebooks_dim:
            codes = codes.flatten(-2)

        codes = self.project_out(codes)

        if is_img_or_video or self.channel_first:
            codes = codes.permute(0, 3, 1, 2)

        return codes

    def construct(self, z):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension
        c - number of codebook dim
        """

        is_img_or_video = z.ndim >= 4
        need_move_channel_last = is_img_or_video or self.channel_first

        # standardize image or video into (batch, seq, dimension)

        if need_move_channel_last:
            z = z.permute(0, 2, 3, 1)
            z, ps = pack_one(z, "b * d")

        assert z.shape[-1] == self.dim, f"expected dimension of {self.dim} but found dimension of {z.shape[-1]}"

        z = self.project_in(z)

        b, n, cd = z.shape
        c = self.num_codebooks
        d = cd // c
        z = z.reshape(b, n, c, d)

        # whether to force quantization step to be full precision or not

        orig_dtype = z.dtype

        z = z.float()

        codes = self.quantize(z)

        # returning indices could be optional

        indices = None

        if self.return_indices:
            indices = self.codes_to_indices(codes)

        codes = codes.flatten(start_dim=2)

        codes = codes.type(orig_dtype)

        # project out

        out = self.project_out(codes)

        # reconstitute image or video dimensions

        if need_move_channel_last:
            out = unpack_one(out, ps, "b * d")
            out = out.permute(0, 3, 1, 2)

            indices = maybe(unpack_one)(indices, ps, "b * c")

        if not self.keep_num_codebooks_dim and self.return_indices:
            indices = maybe(mint.squeeze)(indices, -1)

        # return quantized output and indices

        return out, indices
