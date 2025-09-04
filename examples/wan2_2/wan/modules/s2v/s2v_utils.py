# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from typing import List, Optional, Union

import numpy as np

import mindspore as ms
import mindspore.mint as mint


def rope_precompute(
    x: ms.Tensor,
    grid_sizes: Union[List[ms.Tensor], ms.Tensor],
    freqs: Union[List[ms.Tensor], ms.Tensor],
    start: Optional[List[ms.Tensor]] = None,
) -> ms.Tensor:
    x = x.to(ms.float32)
    b, s, n, c = x.shape[0], x.shape[1], x.shape[2], x.shape[3] // 2

    # split freqs
    if type(freqs) is list:
        trainable_freqs = freqs[1]
        freqs = freqs[0]
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = x.reshape(b, s, n, -1, 2).to(ms.float32)
    seq_bucket = [0]
    if not type(grid_sizes) is list:
        grid_sizes = [grid_sizes]
    for g in grid_sizes:
        if not type(g) is list:
            g = [mint.zeros_like(g), g]
        batch_size = g[0].shape[0]
        for i in range(batch_size):
            if start is None:
                f_o, h_o, w_o = g[0][i]
            else:
                f_o, h_o, w_o = start[i]

            f, h, w = g[1][i]
            t_f, t_h, t_w = g[2][i]
            seq_f, seq_h, seq_w = f - f_o, h - h_o, w - w_o
            seq_f, seq_h, seq_w = seq_f.item(), seq_h.item(), seq_w.item()
            seq_len = int(seq_f * seq_h * seq_w)
            if seq_len > 0:
                if t_f > 0:
                    # Generate a list of seq_f integers starting from f_o and ending at math.ceil(factor_f * seq_f.item() + f_o.item())
                    if f_o >= 0:
                        f_sam = np.linspace(f_o.item(), (t_f + f_o).item() - 1, seq_f).astype(int).tolist()
                    else:
                        f_sam = np.linspace(-f_o.item(), (-t_f - f_o).item() + 1, seq_f).astype(int).tolist()
                    h_sam = np.linspace(h_o.item(), (t_h + h_o).item() - 1, seq_h).astype(int).tolist()
                    w_sam = np.linspace(w_o.item(), (t_w + w_o).item() - 1, seq_w).astype(int).tolist()

                    assert f_o * f >= 0 and h_o * h >= 0 and w_o * w >= 0
                    if f_o >= 0:
                        freqs_0 = freqs[0][f_sam]
                    else:
                        freqs_0 = freqs[0][f_sam]
                        freqs_0[..., 1] = -freqs_0[..., 1]
                    freqs_0 = freqs_0.view(seq_f, 1, 1, -1, 2)

                    freqs_i = mint.cat(
                        [
                            freqs_0.expand((seq_f, seq_h, seq_w, -1, 2)),
                            freqs[1][h_sam].view(1, seq_h, 1, -1, 2).expand((seq_f, seq_h, seq_w, -1, 2)),
                            freqs[2][w_sam].view(1, 1, seq_w, -1, 2).expand((seq_f, seq_h, seq_w, -1, 2)),
                        ],
                        dim=-2,
                    ).reshape(seq_len, 1, -1, 2)
                elif t_f < 0:
                    freqs_i = trainable_freqs.unsqueeze(1)
                # apply rotary embedding
                output[i, seq_bucket[-1] : seq_bucket[-1] + seq_len] = freqs_i
        seq_bucket.append(seq_bucket[-1] + seq_len)
    return output
