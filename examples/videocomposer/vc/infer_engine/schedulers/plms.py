from typing import Optional

import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

__all__ = ["PLMS"]


class PLMS(nn.Cell):
    def __init__(self, model: nn.Cell, betas: np.ndarray, eps_shape=(1, 4, 16, 32, 32)) -> None:
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

        self.num_cache = ms.Parameter(ms.Tensor(0, dtype=ms.int32), requires_grad=False)
        self.eps_0 = ms.Parameter(ms.Tensor(np.zeros(eps_shape), dtype=ms.float32), requires_grad=False)
        self.eps_1 = ms.Parameter(ms.Tensor(np.zeros(eps_shape), dtype=ms.float32), requires_grad=False)
        self.eps_2 = ms.Parameter(ms.Tensor(np.zeros(eps_shape), dtype=ms.float32), requires_grad=False)

    def reset_cache(self):
        self.num_cache *= 0
        self.eps_0 *= 0
        self.eps_1 *= 0
        self.eps_2 *= 0

    def _compute_eps(self, model_out: Tensor, xt: Tensor, t: Tensor) -> Tensor:
        x0 = self.sqrt_recip_alphas_cumprod[t] * xt - self.sqrt_recipm1_alphas_cumprod[t] * model_out
        eps = (self.sqrt_recip_alphas_cumprod[t] * xt - x0) / self.sqrt_recipm1_alphas_cumprod[t]
        return eps

    def _compute_x0(self, eps: Tensor, xt: Tensor, t: Tensor, stride: Tensor) -> Tensor:
        x0 = self.sqrt_recip_alphas_cumprod[t] * xt - self.sqrt_recipm1_alphas_cumprod[t] * eps
        alphas_prev = self.alphas_cumprod[(t - stride).clamp(0)]
        direction = ops.sqrt(1 - alphas_prev) * eps
        xt_1 = ops.sqrt(alphas_prev) * x0 + direction
        return xt_1

    def _compute_eps_prime_0(self, model_out: Tensor, xt: Tensor, t: Tensor, stride: Tensor, eps: Tensor) -> Tensor:
        # 2nd order pseudo improved Euler
        xt_1 = self._compute_x0(eps, xt, t, stride)
        eps_next = self._compute_eps(model_out, xt_1, (t - stride).clamp(0))
        eps_prime = (eps + eps_next) / 2.0
        return eps_prime

    def _compute_eps_prime_1(self, eps: Tensor) -> Tensor:
        # 2nd order pseudo linear multistep (Adams-Bashforth)
        eps_prime = (3 * eps - self.eps_2) / 2.0
        return eps_prime

    def _compute_eps_prime_2(self, eps: Tensor) -> Tensor:
        # 3nd order pseudo linear multistep (Adams-Bashforth)
        eps_prime = (23 * eps - 16 * self.eps_2 + 5 * self.eps_1) / 12.0
        return eps_prime

    def _compute_eps_prime_3(self, eps: Tensor) -> Tensor:
        # 4nd order pseudo linear multistep (Adams-Bashforth)
        eps_prime = (55 * eps - 59 * self.eps_2 + 37 * self.eps_1 - 9 * self.eps_0) / 24.0
        return eps_prime

    def _schedule(self, model_out: Tensor, xt: Tensor, t: Tensor, stride: Tensor) -> Tensor:
        eps = self._compute_eps(model_out, xt, t)
        # compute every branch in each single call, avoid memcpy error or other strange error in graph mode
        eps_prime_0 = self._compute_eps_prime_0(model_out, xt, t, stride, eps)
        eps_prime_1 = self._compute_eps_prime_1(eps)
        eps_prime_2 = self._compute_eps_prime_2(eps)
        eps_prime_3 = self._compute_eps_prime_3(eps)

        eps_prime = ops.concat([eps_prime_0, eps_prime_1, eps_prime_2, eps_prime_3])
        eps_prime = eps_prime[self.num_cache]
        xt_1 = self._compute_x0(eps_prime, xt, t, stride)

        self.num_cache = ops.clip(self.num_cache + 1, 0, 3)
        xt_1 = ops.depend(xt_1, self.num_cache)
        self.eps_0 = self.eps_1
        xt_1 = ops.depend(xt_1, self.eps_0)
        self.eps_1 = self.eps_2
        xt_1 = ops.depend(xt_1, self.eps_1)
        self.eps_2 = eps
        xt_1 = ops.depend(xt_1, self.eps_2)
        return xt_1

    def construct(
        self,
        xt: Tensor,
        t: Tensor,
        stride: Tensor,
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

        xt = self._schedule(out, xt, t, stride)
        return xt


class PLMS_PYNATIVE(PLMS):
    """Only work on PYNATIVE mode"""

    def __init__(self, model: nn.Cell, betas: np.ndarray, eps_shape=(1, 4, 16, 32, 32)) -> None:
        super().__init__(model, betas, eps_shape)
        self.eps_cache = []

    def _schedule(self, model_out: Tensor, xt: Tensor, t: Tensor, stride: Tensor) -> Tensor:
        eps = self._compute_eps(model_out, xt, t)
        if len(self.eps_cache) == 0:
            xt_1 = self._compute_x0(eps, xt, t, stride)
            eps_next = self._compute_eps(model_out, xt_1, (t - stride).clamp(0))
            eps_prime = (eps + eps_next) / 2.0
        elif len(self.eps_cache) == 1:
            eps_prime = (3 * eps - self.eps_cache[-1]) / 2.0
        elif len(self.eps_cache) == 2:
            eps_prime = (23 * eps - 16 * self.eps_cache[-1] + 5 * self.eps_cache[-2]) / 12.0
        else:
            eps_prime = (55 * eps - 59 * self.eps_cache[-1] + 37 * self.eps_cache[-2] - 9 * self.eps_cache[-3]) / 24.0

        xt_1 = self._compute_x0(eps_prime, xt, t, stride)

        self.eps_cache.append(eps)
        if len(self.eps_cache) > 3:
            self.eps_cache.pop(0)

        return xt_1


class _Identity(nn.Cell):
    def construct(self, *args, **kwargs):
        return args[0]


if __name__ == "__main__":
    # test to see if two implementation's result are the same under two mode
    from .sampler import beta_schedule

    np.random.seed(0)
    betas = beta_schedule("linear_sd", 1000, init_beta=0.00085, last_beta=0.0120)
    shape = (1, 4, 16, 32, 32)
    inputs = ms.Tensor(np.random.randn(*shape), dtype=ms.float32)
    dummy = ms.Tensor(1, dtype=ms.float32)
    stride = ms.Tensor(1, dtype=ms.int32)

    ms.set_context(mode=ms.GRAPH_MODE)
    net_graph = PLMS(_Identity(), betas)
    inputs_graph = inputs
    for i in range(10):
        output_graph = net_graph(inputs_graph, ms.Tensor(i, dtype=ms.int32), stride, dummy, dummy)
        inputs_graph = output_graph

    ms.set_context(mode=ms.PYNATIVE_MODE)
    net_pynative = PLMS_PYNATIVE(_Identity(), betas)
    inputs_pynative = inputs
    for i in range(10):
        output_pynative = net_pynative(inputs_pynative, ms.Tensor(i, dtype=ms.int32), stride, dummy, dummy)
        inputs_pynative = output_pynative

    output_graph = output_graph.asnumpy()
    output_pynative = output_pynative.asnumpy()
    assert not np.isnan(output_graph).any()
    np.testing.assert_almost_equal(output_graph, output_pynative, decimal=6)
    print("Test passed.")
