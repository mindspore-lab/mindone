from typing import Optional

import numpy as np

__all__ = ["get_1d_sincos_pos_embed", "get_2d_sincos_pos_embed", "precompute_freqs_cis_2d"]


def get_2d_sincos_pos_embed(embed_dim: int, nh: int, nw: Optional[int] = None) -> np.ndarray:
    """Generate 2D sinusoidal positional embedding based on the given height and width
    referred from https://github.com/facebookresearch/mae

    Args:
        embed_dim: embedding dimension.
        nh: image height
        nw: image width. If it is not given, then `nw` is equal to `nh`. Default: None
    """
    nw = nh if nw is None else nw
    grid_h = np.arange(nh, dtype=np.float32)
    grid_w = np.arange(nw, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, nh, nw])
    pos_embed = _get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_1d_sincos_pos_embed(embed_dim: int, length: int) -> np.ndarray:
    """
    Generate sinusoidal/cosinusoidal positional embeddings for 1D data.

    Args:
        embed_dim (int): The dimensionality of the embeddings.
        length (int): The length of the 1D data.

    Returns:
        numpy.ndarray: The positional embeddings of shape (length, embed_dim).
    """
    pos = np.arange(0, length).reshape((-1, 1))
    return _get_1d_sincos_pos_embed_from_grid(embed_dim, pos)


def precompute_freqs_cis_2d(
    dim: int, nh: int, nw: Optional[int] = None, theta: float = 10000.0, max_length: Optional[int] = None
) -> np.ndarray:
    """Precompute the frequency tensor for complex exponentials (cis) with given dimensions, for 2D RoPE
    referered from 1D RoPE https://github.com/meta-llama/llama and paper `FiT` https://arxiv.org/abs/2402.12376

    If max_length is not None, then a length extrapolation algo. `VisionNTK` from `FiT` will be used for tensor calculation.

    Args:
        dim: dimension of the frequency tensor
        nh: image height
        nw: image width. If it is not given, then `nw` is equal to `nh`. Default: None
        theta: Scaling factor for frequency computation. Defaults: 10000.0.
        max_length: If it is None, then the VisionNTK algo. will be applied. Default: None
    """
    nw = nh if nw is None else nw
    grid_h = np.arange(nh, dtype=np.float32)
    grid_w = np.arange(nw, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, nh, nw])
    freqs_cis = _precompute_freqs_cis_2d_from_grid(dim, grid, theta=theta, max_length=max_length)  # (M, D/2, 2)
    freqs_cis = np.reshape(freqs_cis, (freqs_cis.shape[0], -1))
    return freqs_cis


def _get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_w, emb_h], axis=1)  # (H*W, D)
    return emb


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)
    out = np.outer(pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def _precompute_freqs_cis_2d_from_grid(
    dim: int, grid: np.ndarray, theta: float = 10000.0, max_length: Optional[int] = None
) -> np.ndarray:
    freqs_cis_w = _precompute_freqs_cis_1d_from_grid(dim // 2, grid[0], theta=theta, max_length=max_length)
    freqs_cis_h = _precompute_freqs_cis_1d_from_grid(dim // 2, grid[1], theta=theta, max_length=max_length)
    freqs_cis = np.concatenate([freqs_cis_w, freqs_cis_h], axis=1)
    return freqs_cis


def _precompute_freqs_cis_1d_from_grid(
    dim: int, pos: np.ndarray, theta: float = 10000.0, max_length: Optional[int] = None
) -> np.ndarray:
    if max_length is not None:
        # VisionNTK
        s = max(np.max(pos) / np.sqrt(max_length), 1.0)
        theta = theta * np.power(s, dim / (dim - 2))

    freqs = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32)[: (dim // 2)] / dim))
    freqs = np.outer(pos, freqs)
    a = np.cos(freqs)
    b = np.sin(freqs)  # represent for a + ib
    freqs_cis = np.stack([a, b], axis=-1)
    return freqs_cis
