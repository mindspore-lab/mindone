from typing import Optional

import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

__all__ = ["DDIM"]


class DDIM(nn.Cell):
    def __init__(self, model: nn.Cell, betas: np.ndarray) -> None:
        super().__init__()
        self.model = model

        alphas = 1 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        # q(x_t | x_{t-1})
        sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)

        self.alphas_cumprod = ms.Tensor(alphas_cumprod, dtype=ms.float32)
        self.sqrt_recip_alphas_cumprod = ms.Tensor(sqrt_recip_alphas_cumprod, dtype=ms.float32)
        self.sqrt_recipm1_alphas_cumprod = ms.Tensor(sqrt_recipm1_alphas_cumprod, dtype=ms.float32)

    def _schedule(self, model_out: Tensor, xt: Tensor, t: Tensor, stride: Tensor, eta: Tensor) -> Tensor:
        x0 = self.sqrt_recip_alphas_cumprod[t] * xt - self.sqrt_recipm1_alphas_cumprod[t] * model_out

        # derive variables
        eps = (self.sqrt_recip_alphas_cumprod[t] * xt - x0) / self.sqrt_recipm1_alphas_cumprod[t]
        alphas = self.alphas_cumprod[t]
        alphas_prev = self.alphas_cumprod[(t - stride).clamp(0)]
        sigmas = eta * ops.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))

        # random sample
        noise = ops.randn_like(xt)
        direction = ops.sqrt(1 - alphas_prev - sigmas**2) * eps
        mask = t.ne(0).to(xt.dtype)
        xt_1 = ops.sqrt(alphas_prev) * x0 + direction + mask * sigmas * noise
        return xt_1

    def construct(
        self,
        xt: Tensor,
        t: Tensor,
        stride: Tensor,
        eta: Tensor,
        guide_scale: Tensor,
        y: Tensor,
        depth: Optional[Tensor] = None,
        image: Optional[Tensor] = None,
        motion: Optional[Tensor] = None,
        local_image: Optional[Tensor] = None,
        single_sketch: Optional[Tensor] = None,
        masked: Optional[Tensor] = None,
        canny: Optional[Tensor] = None,
        sketch: Optional[Tensor] = None,
        fps: Optional[Tensor] = None,
        video_mask: Optional[Tensor] = None,
        focus_present_mask: Optional[Tensor] = None,
        prob_focus_present: float = 0.0,
        mask_last_frame_num: int = 0,
    ) -> Tensor:
        ts = ms.numpy.full((xt.shape[0],), t, dtype=ms.int32)

        # classifier-free guidance
        xt_tiled = ops.tile(xt, (2, 1, 1, 1, 1))
        ts_tiled = ops.tile(ts, (2,))
        out = self.model(
            xt_tiled,
            ts_tiled,
            y,
            depth,
            image,
            motion,
            local_image,
            single_sketch,
            masked,
            canny,
            sketch,
            fps,
            video_mask,
            focus_present_mask,
            prob_focus_present,
            mask_last_frame_num,
        )
        y_out, u_out = ops.chunk(out, 2, axis=0)
        out = u_out + guide_scale * (y_out - u_out)

        xt = self._schedule(out, xt, t, stride, eta)
        return xt
