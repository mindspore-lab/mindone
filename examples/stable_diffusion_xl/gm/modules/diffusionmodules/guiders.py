# reference to https://github.com/Stability-AI/generative-models
from functools import partial

from gm.util import default, instantiate_from_config

from mindspore import ops


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
                {"target": "gm.modules.diffusionmodules.sampling_utils.NoDynamicThresholding"},
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
