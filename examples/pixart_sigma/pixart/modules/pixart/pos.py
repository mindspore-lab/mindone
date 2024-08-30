from typing import Optional

import mindspore as ms
import mindspore.ops as ops
from mindspore import Tensor


def cal_2d_sincos_pos_embed(
    embed_dim: int,
    nh: int,
    nw: Optional[int] = None,
    scale: float = 1.0,
    base_size: Optional[int] = None,
    omega: Optional[Tensor] = None,
) -> Tensor:
    nw = nh if nw is None else nw
    if base_size is None:
        grid_h = ops.arange(nh, dtype=ms.float32) / scale
        grid_w = ops.arange(nw, dtype=ms.float32) / scale
    else:
        grid_h = ops.arange(nh, dtype=ms.float32) / (nh / base_size) / scale
        grid_w = ops.arange(nw, dtype=ms.float32) / (nw / base_size) / scale
    grid = ops.meshgrid(grid_w, grid_h)  # here w goes first
    grid = ops.stack(grid, axis=0)
    grid = ops.reshape(grid, [2, nh, nw])
    pos_embed = _cal_2d_sincos_pos_embed_from_grid(embed_dim, grid, omega=omega)
    return pos_embed


def _cal_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: Tensor, omega: Optional[Tensor] = None) -> Tensor:
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_w = _cal_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0], omega=omega)  # (H*W, D/2)
    emb_h = _cal_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1], omega=omega)  # (H*W, D/2)

    emb = ops.concat([emb_w, emb_h], axis=1)  # (H*W, D)
    return emb


def _cal_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: Tensor, omega: Optional[Tensor] = None) -> Tensor:
    if omega is None:
        omega = cal_omega(embed_dim)

    out = ops.outer(pos, omega)  # (M, D/2), outer product

    emb_sin = ops.sin(out)  # (M, D/2)
    emb_cos = ops.cos(out)  # (M, D/2)

    emb = ops.concat([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def cal_omega(embed_dim: int) -> Tensor:
    assert embed_dim % 2 == 0
    omega = ops.arange(embed_dim // 2, dtype=ms.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)
    return omega
