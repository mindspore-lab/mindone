"""
Finite Scalar Quantization: VQ-VAE Made Simple - https://arxiv.org/abs/2309.15505
Code adapted from Jax version in Appendix A.1
"""

from __future__ import annotations
from typing import List, Tuple

import mindspore as ms
from mindspore import nn, ops


# helper functions


def exists(v):
    return v is not None


def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None


# tensor helpers


def round_ste(z: ms.Tensor) -> ms.Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + ops.stop_gradient(zhat - z)


# main class


class FSQ(nn.Cell):
    def __init__(
        self,
        levels: List[int],
        dim: int | None = None,
        num_codebooks=1,
        keep_num_codebooks_dim: bool | None = None,
        scale: float | None = None,
        channel_first: bool = False,
        projection_has_bias: bool = True,
        dtype=ms.float32,
    ):
        super().__init__()
        _levels = ms.Tensor(levels, dtype=ms.int32)
        self._levels = _levels
        # self.register_buffer("_levels", _levels, persistent = False)

        _basis = ops.cumprod(ms.Tensor([1] + levels[:-1]), dim=0, dtype=ms.int32)
        self._basis = _basis
        # self.register_buffer("_basis", _basis, persistent = False)

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
            nn.Dense(self.dim, effective_codebook_dim, has_bias=projection_has_bias)
            if has_projections
            else nn.Identity()
        )
        self.project_out = (
            nn.Dense(effective_codebook_dim, self.dim, has_bias=projection_has_bias)
            if has_projections
            else nn.Identity()
        )

        self.has_projections = has_projections

        self.codebook_size = self._levels.prod()

        implicit_codebook = self._indices_to_codes(ops.arange(self.codebook_size))
        self.implicit_codebook = implicit_codebook
        # self.register_buffer("implicit_codebook", implicit_codebook, persistent = False)

        self.dtype = dtype

    def bound(self, z, eps: float = 1e-3):
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1).float().to(self.dtype) * (1.0 + eps) / 2.0
        offset = ops.where(self._levels % 2 == 0, ms.Tensor(0.5, self.dtype), ms.Tensor(0.0, self.dtype))
        shift = ops.atanh(offset / half_l)
        return ops.tanh(z + shift) * half_l - offset

    def quantize(self, z):
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z, eps=0.01))
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
        return ops.sum(zhat * self._basis, dim=-1).to(ms.int32)

    def indices_to_level_indices(self, indices):
        """C indices at each level, perhaps needed for a transformer with factorized embeddings"""
        # indices = rearrange(indices, '... -> ... 1')
        indices = indices.unsqueeze(-1)
        codes_non_centered = (indices // self._basis) % self._levels
        return codes_non_centered

    def indices_to_codes(self, indices):
        """Inverse of `codes_to_indices`."""

        # is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))
        is_img_or_video = True

        codes = self._indices_to_codes(indices)

        if self.keep_num_codebooks_dim:
            # codes = rearrange(codes, '... c d -> ... (c d)')
            codes = codes.reshape(tuple(codes.shape[:-2]) + (-1))

        codes = self.project_out(codes)

        if is_img_or_video or self.channel_first:
            # codes = rearrange(codes, 'b ... d -> b d ...')
            codes = codes.permute(0, 4, 1, 2, 3)

        return codes

    def construct(self, z):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension
        c - number of codebook dim
        """

        # standardize image or video into (batch, seq, dimension)

        # z = rearrange(z, 'b d ... -> b ... d')
        # z, ps = pack_one(z, 'b * d')

        z = z.permute(0, 2, 3, 4, 1)
        z_shape = z.shape
        b = z.shape[0]
        d = z.shape[-1]
        z = z.reshape(b, -1, d)

        assert (
            z.shape[-1] == self.dim
        ), f"expected dimension of {self.dim} but found dimension of {z.shape[-1]}"

        z = self.project_in(z)

        # z = rearrange(z, 'b n (c d) -> b n c d', c = self.num_codebooks)
        b, n, _ = z.shape
        z = z.reshape(b, n, self.num_codebooks, -1)

        # make sure allowed dtype before quantizing

        # if z.dtype not in self.allowed_dtypes:
        #     z = z.float()

        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)

        # codes = rearrange(codes, 'b n c d -> b n (c d)')
        b, n, c, d = codes.shape
        codes = codes.reshape(b, n, -1)

        # cast codes back to original dtype

        # if codes.dtype != orig_dtype:
        #     codes = codes.type(orig_dtype)

        # project out

        out = self.project_out(codes)

        # reconstitute image or video dimensions

        # out = unpack_one(out, ps, 'b * d')
        # out = rearrange(out, 'b ... d -> b d ...')
        out = out.reshape(*z_shape)
        out = out.permute(0, 4, 1, 2, 3)

        indices = indices.reshape(tuple(z_shape[:-1]) + (-1,))
        # indices = unpack_one(indices, ps, 'b * c')

        if not self.keep_num_codebooks_dim:
            # indices = rearrange(indices, '... 1 -> ...')
            indices = indices.squeeze(-1)

        # return quantized output and indices

        return out, indices, ms.Tensor(0.0, dtype=self.dtype)
