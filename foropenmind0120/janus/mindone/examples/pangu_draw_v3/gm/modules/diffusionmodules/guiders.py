# reference to https://github.com/Stability-AI/generative-models
from functools import partial

from gm.util import default, instantiate_from_config

from mindspore import ops


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


class PanGuVanillaCFG:
    """
    implements parallelized CFG
    """

    def __init__(self, scale, dyn_thresh_config=None, other_scale=None):
        scale_schedule = lambda scale, sigma: scale  # independent of step
        self.scale_schedule = partial(scale_schedule, scale)
        self.dyn_thresh = instantiate_from_config(
            default(
                dyn_thresh_config,
                {"target": "gm.modules.diffusionmodules.sampling_utils.PanGuNoDynamicThresholding"},
            )
        )
        self.other_scale = other_scale if other_scale is not None else []
        self.other_scale_num = len(self.other_scale)

    def __call__(self, x, sigma):
        x_list = x.chunk(2 + self.other_scale_num)
        scale_value = self.scale_schedule(sigma)
        x_pred = self.dyn_thresh(x_list, scale_value, self.other_scale)
        return x_pred

    def prepare_inputs(self, x, s, c, uc, other_c):
        c_out = dict()
        x_list = [x] * (self.other_scale_num + 2)
        s_list = [s] * (self.other_scale_num + 2)

        for k in c:
            if k in ["vector", "crossattn", "concat"]:
                cat_list = [uc[k], c[k]]
                for _c in other_c:
                    cat_list.append(_c[k])
                c_out[k] = ops.concat(cat_list, 0)
            else:
                assert c[k] == uc[k]
                c_out[k] = c[k]

        return ops.concat(x_list), ops.concat(s_list), c_out


class IdentityGuider:
    def __call__(self, x, sigma):
        return x

    def prepare_inputs(self, x, s, c, uc):
        c_out = dict()

        for k in c:
            c_out[k] = c[k]

        return x, s, c_out
