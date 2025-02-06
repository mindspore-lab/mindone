# reference to https://github.com/Stability-AI/generative-models
from __future__ import annotations

from functools import partial
from typing import List, Optional, Tuple, Union

from sgm.util import append_dims, default, instantiate_from_config

from mindspore import Tensor, nn, ops

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # FIXME: python 3.7


class LinearPredictionGuider(nn.Cell):
    def __init__(
        self,
        num_frames: int,
        min_scale: float = 1.0,
        max_scale: float = 2.5,
        additional_cond_keys: Optional[Union[List[str], str]] = None,
    ):
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.num_frames = num_frames

        additional_cond_keys = additional_cond_keys or []
        if isinstance(additional_cond_keys, str):
            additional_cond_keys = [additional_cond_keys]
        self.additional_cond_keys = additional_cond_keys

    def construct(self, x: Tensor, sigma: Tensor) -> Tensor:
        x_u, x_c = x.chunk(2)

        # (b t) ... -> b t ...
        x_u = x_u.reshape(-1, self.num_frames, *x_u.shape[1:])
        x_c = x_c.reshape(-1, self.num_frames, *x_c.shape[1:])

        scale = ops.linspace(self.min_scale, self.max_scale, self.num_frames)[None, :]
        scale = scale.repeat(x_u.shape[0], axis=0)  # 1 t -> b t
        scale = append_dims(scale, x_u.ndim)

        out = x_u + scale * (x_c - x_u)
        out = out.reshape(-1, *out.shape[2:])  # b t ... -> (b t) ...
        return out

    def prepare_inputs(self, x: Tensor, s: Tensor, c: dict, uc: dict) -> Tuple[Tensor, Tensor, dict]:
        c_out = dict()

        # for k in c:
        #     if k in ["vector", "crossattn", "concat"] + self.additional_cond_keys:
        #         c_out[k] = ops.cat((uc[k], c[k]))
        #     else:
        #         assert c[k] == uc[k]
        #         c_out[k] = c[k]

        c_out["vector"] = ops.cat((uc["vector"], c["vector"]))
        c_out["crossattn"] = ops.cat((uc["crossattn"], c["crossattn"]))
        c_out["concat"] = ops.cat((uc["concat"], c["concat"]))
        return ops.concat((x, x)), ops.concat((s, s)), c_out

    # def prepare_inputs(
    #     self,
    #     x: Tensor,
    #     s: Tensor,
    #     vector_c: Tensor,
    #     crossattn_c: Tensor,
    #     concat_c: Tensor,
    #     vector_uc: Tensor,
    #     crossattn_uc: Tensor,
    #     concat_uc: Tensor,
    # ) -> Tuple[Tensor, Tensor, dict]:
    #     c_out = dict()

    #     # for k in ["vector", "crossattn", "concat"] :
    #     c_out["vector"] = ops.cat((vector_uc, vector_c))
    #     c_out["crossattn"] = ops.cat((crossattn_uc, crossattn_c))
    #     c_out["concat"] = ops.cat((concat_uc, concat_c))
    #     return ops.concat((x, x)), ops.concat((s, s)), c_out


class VanillaCFG:
    """
    implements parallelized CFG (classifier-free guidance)
    """

    def __init__(self, scale, dyn_thresh_config=None):
        scale_schedule = lambda scale, sigma: scale  # independent of step
        self.scale_schedule = partial(scale_schedule, scale)
        self.dyn_thresh = instantiate_from_config(
            default(
                dyn_thresh_config,
                {"target": "sgm.modules.diffusionmodules.sampling_utils.NoDynamicThresholding"},
            )
        )

    def __call__(self, x, sigma):
        x = self.dyn_thresh(x)
        x_uncond, x_cond = x.chunk(2)
        scale_value = self.scale_schedule(sigma)
        x_pred = x_uncond + scale_value * (x_cond - x_uncond)
        return x_pred

    def prepare_inputs(self, x, s, c, uc):
        c_out = dict()

        for k in c:
            if k in ["vector", "crossattn", "concat"]:
                c_out[k] = ops.concat((uc[k], c[k]), 0)
            else:
                assert c[k] == uc[k]
                c_out[k] = c[k]

        return ops.concat((x, x)), ops.concat((s, s)), c_out


class IdentityGuider:
    def __call__(self, x, sigma):
        return x

    def prepare_inputs(self, x, s, c, uc):
        c_out = dict()

        for k in c:
            c_out[k] = c[k]

        return x, s, c_out


class TrianglePredictionGuider(LinearPredictionGuider):
    def __init__(
        self,
        max_scale: float,
        num_frames: int,
        min_scale: float = 1.0,
        period: float | List[float] = 1.0,
        period_fusing: Literal["mean", "multiply", "max"] = "max",
        additional_cond_keys: Optional[Union[List[str], str]] = None,
    ):
        super().__init__(num_frames, min_scale, max_scale, additional_cond_keys)
        values = ops.linspace(0, 1, num_frames)
        # Constructs a triangle wave
        if isinstance(period, float):
            period = [period]

        scales = []
        for p in period:
            scales.append(self.triangle_wave(values, p))

        if period_fusing == "mean":
            scale = sum(scales) / len(period)
        elif period_fusing == "multiply":
            scale = ops.prod(ops.stack(scales), axis=0)
        elif period_fusing == "max":
            scale = ops.max(ops.stack(scales), axis=0)[0]
        self.scale = (scale * (max_scale - min_scale) + min_scale).unsqueeze(0)

    def triangle_wave(self, values: Tensor, period: float) -> Tensor:
        return 2 * (values / period - ops.floor(values / period + 0.5)).abs()
