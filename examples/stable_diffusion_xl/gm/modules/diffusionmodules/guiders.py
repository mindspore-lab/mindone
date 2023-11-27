# reference to https://github.com/Stability-AI/generative-models
from functools import partial
from typing import List, Optional, Tuple, Union

import numpy as np
from gm.util import append_dims, default, instantiate_from_config

from mindspore import Tensor, nn, ops


class VanillaCFG:
    """
    implements parallelized CFG
    """

    def __init__(self, scale, dyn_thresh_config=None):
        scale_schedule = lambda scale, sigma: scale  # independent of step
        self.scale_schedule = partial(scale_schedule, scale)
        self.dyn_thresh = instantiate_from_config(
            default(
                dyn_thresh_config,
                {"target": "gm.modules.diffusionmodules.sampling_utils.NoDynamicThresholding"},
            )
        )

    def __call__(self, x, sigma):
        _id = x.shape[0] // 2
        x_u, x_c = x[:_id], x[_id:]
        scale_value = self.scale_schedule(sigma)
        x_pred = self.dyn_thresh(x_u, x_c, scale_value)
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


class LinearPredictionGuider(nn.Cell):
    def __init__(
        self,
        max_scale: float,
        num_frames: int,
        min_scale: float = 1.0,
        additional_cond_keys: Optional[Union[List[str], str]] = None,
    ):
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.num_frames = num_frames
        self.scale = Tensor(np.expand_dims(np.linspace(min_scale, max_scale, num_frames, dtype=np.float32), 0))

        additional_cond_keys = additional_cond_keys or []
        if isinstance(additional_cond_keys, str):
            additional_cond_keys = [additional_cond_keys]
        self.additional_cond_keys = additional_cond_keys

    def construct(self, x: Tensor, sigma: Tensor) -> Tensor:
        x_u, x_c = x.chunk(2)

        # (b t) ... -> b t ...
        x_u = x_u.reshape(-1, self.num_frames, *x_u.shape[1:])
        x_c = x_c.reshape(-1, self.num_frames, *x_c.shape[1:])

        scale = self.scale.repeat(x_u.shape[0], axis=0)  # 1 t -> b t
        scale = append_dims(scale, x_u.ndim)

        out = x_u + scale * (x_c - x_u)
        out = out.reshape(-1, *out.shape[2:])  # b t ... -> (b t) ...
        return out

    def prepare_inputs(self, x: Tensor, s: Tensor, c: dict, uc: dict) -> Tuple[Tensor, Tensor, dict]:
        c_out = dict()

        for k in c:
            if k in ["vector", "crossattn", "concat"] + self.additional_cond_keys:
                c_out[k] = ops.cat((uc[k], c[k]))
            else:
                assert c[k] == uc[k]
                c_out[k] = c[k]
        return ops.concat((x, x)), ops.concat((s, s)), c_out
