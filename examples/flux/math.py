import math
from typing import Tuple

import mindspore as ms
from mindspore import Tensor, ops


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    q, k = apply_rope(q, k, pe)

    scale_factor = float(1 / math.sqrt(q.shape[-1]))
    x = ops.operations.nn_ops.FlashAttentionScore(head_num=q.shape[1], input_layout="BNSD", scale_value=scale_factor)(
        q.to(ms.bfloat16), k.to(ms.bfloat16), v.to(ms.bfloat16), None, None, None, None
    )[3].to(q.dtype)
    # x = rearrange(x, "B H L D -> B L (H D)")
    B, H, L, D = x.shape
    x = x.swapaxes(1, 2).reshape(B, L, -1)

    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = ops.arange(0, dim, 2, dtype=ms.float64) / dim
    omega = 1.0 / (theta**scale)
    out = pos.unsqueeze(-1) * omega
    out = ops.stack([ops.cos(out), -ops.sin(out), ops.sin(out), ops.cos(out)], axis=-1)
    out = out.reshape(*out.shape[:-1], 2, 2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> Tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).to(xq.dtype), xk_out.reshape(*xk.shape).to(xk.dtype)
