from typing import List, Optional, Tuple, Union

from gm.util import append_dims

from mindspore import Tensor, nn, ops


class LinearPredictionGuider(nn.Cell):
    def __init__(
        self,
        min_scale: float = 1.0,
        max_scale: float = 2.5,
        additional_cond_keys: Optional[Union[List[str], str]] = None,
    ):
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale

        additional_cond_keys = additional_cond_keys or []
        if isinstance(additional_cond_keys, str):
            additional_cond_keys = [additional_cond_keys]
        self.additional_cond_keys = additional_cond_keys

    def construct(self, x: Tensor, sigma: Tensor, num_frames: int) -> Tensor:
        x_u, x_c = x.chunk(2)

        # (b t) ... -> b t ...
        x_u = x_u.reshape(-1, num_frames, *x_u.shape[1:])
        x_c = x_c.reshape(-1, num_frames, *x_c.shape[1:])

        scale = ops.linspace(self.min_scale, self.max_scale, num_frames)[None, :]
        scale = scale.repeat(x_u.shape[0], axis=0)  # 1 t -> b t
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
