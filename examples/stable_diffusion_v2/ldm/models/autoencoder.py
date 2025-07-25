# This code is adapted from https://github.com/Stability-AI/stablediffusion
# with modifications to run on MindSpore.

from ldm.modules.diffusionmodules.model import Decoder, Encoder

import mindspore as ms
import mindspore.nn as nn
from mindspore import mint, ops


class AutoencoderKL(nn.Cell):
    def __init__(
        self,
        ddconfig,
        embed_dim,
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
        colorize_nlabels=None,
        monitor=None,
        use_fp16=False,
        upcast_sigmoid=False,
    ):
        super().__init__()
        self.dtype = ms.float16 if use_fp16 else ms.float32
        self.image_key = image_key
        self.encoder = Encoder(dtype=self.dtype, upcast_sigmoid=upcast_sigmoid, **ddconfig)
        self.decoder = Decoder(dtype=self.dtype, upcast_sigmoid=upcast_sigmoid, **ddconfig)
        assert ddconfig["double_z"]
        self.quant_conv = mint.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1, bias=True).to_float(self.dtype)
        self.post_quant_conv = mint.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1, bias=True).to_float(self.dtype)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", ms.ops.standard_normal(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.stdnormal = ops.StandardNormal()

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = ms.load_checkpoint(path)["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        ms.load_param_into_net(self, sd, strict_load=False)
        print(f"Restored from {path}")

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        mean, logvar = mint.split(moments, moments.shape[1] // 2, dim=1)
        logvar = mint.clamp(logvar, -30.0, 20.0)
        std = mint.exp(0.5 * logvar)
        x = mean + std * self.stdnormal(mean.shape)
        return x

    def encode_with_moments_output(self, x):
        """For latent caching usage"""
        h = self.encoder(x)
        moments = self.quant_conv(h)
        mean, logvar = mint.split(moments, moments.shape[1] // 2, dim=1)
        logvar = mint.clamp(logvar, -30.0, 20.0)
        std = mint.exp(0.5 * logvar)
        return mint.concat([mean, std], dim=1)
