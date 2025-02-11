import sys

sys.path.append("../stable_diffusion_v2")
import logging
import os

import numpy as np
from ldm.models.autoencoder import AutoencoderKL as AutoencoderKL_SD
from utils.download import download_checkpoint

import mindspore as ms
from mindspore import nn, ops

__all__ = ["AutoencoderKL", "get_first_stage_encoding"]

_CKPT_URL = {"2.1": "https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v2-1_base-7c8d09ce.ckpt"}


_logger = logging.getLogger(__name__)


class AutoencoderKL(AutoencoderKL_SD):
    def __init__(self, *args, **kwargs):
        if "version" in kwargs:
            assert kwargs["version"] in _CKPT_URL, f"{kwargs['version']} is not supported yet"
            self.version = kwargs["version"]
            del kwargs["version"]
        else:
            self.version = "2.1"
        super().__init__(*args, **kwargs)

    def init_from_ckpt(self, path, ignore_keys=list()):
        if not os.path.exists(path):
            download_checkpoint(_CKPT_URL[self.version], "model_weights/")
        if not os.path.exists(path):
            raise ValueError(
                f"Maybe download failed. Please download it manually from {_CKPT_URL[self.version]} and place it under `model_weights/`"
            )

        def prune_weights(sd):
            # find keys start with prefix "first_stage_model" and remove the prefix.
            sd_new = {}
            for k in sd.keys():
                if k.find("first_stage_model") >= 0:
                    k_new = k.split("first_stage_model.")[-1]
                    sd_new[k_new] = sd[k]
            return sd_new

        sd = ms.load_checkpoint(path)
        sd = prune_weights(sd)
        param_not_load, ckpt_not_load = ms.load_param_into_net(self, sd, strict_load=True)
        if param_not_load or ckpt_not_load:
            _logger.warning(
                f"{param_not_load} in network is not loaded or {ckpt_not_load} in checkpoint is not loaded!"
            )


class DiagonalGaussianDistribution(nn.Cell):
    def __init__(self, parameters, deterministic=False):
        super().__init__()
        self.parameters = parameters
        self.mean, self.logvar = ops.chunk(parameters, 2, axis=1)
        self.logvar = ops.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = ops.exp(0.5 * self.logvar)
        self.var = ops.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = ops.zeros_like(self.mean)

    def sample(self):
        x = self.mean + self.std * ops.randn(self.mean.shape)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return ms.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * ops.sum(ops.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1, 2, 3])
            else:
                return 0.5 * ops.sum(
                    ops.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return ms.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * ops.sum(logtwopi + self.logvar + ops.pow(sample - self.mean, 2) / self.var, dim=dims)

    def mode(self):
        return self.mean


# @torch.no_grad()
def get_first_stage_encoding(encoder_posterior):
    scale_factor = 0.18215
    if isinstance(encoder_posterior, DiagonalGaussianDistribution):
        z = encoder_posterior.sample()
    elif isinstance(encoder_posterior, ms.Tensor):
        z = encoder_posterior
    else:
        raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
    return scale_factor * z
