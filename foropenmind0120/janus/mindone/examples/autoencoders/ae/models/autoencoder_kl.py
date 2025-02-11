import mindspore as ms
from mindspore import nn, ops

from .modules import Decoder, Encoder

__all__ = ["AutoencoderKL"]


class AutoencoderKL(nn.Cell):
    def __init__(
        self,
        ddconfig,
        embed_dim,
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
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
        self.quant_conv = nn.Conv2d(
            2 * ddconfig["z_channels"], 2 * embed_dim, 1, pad_mode="valid", has_bias=True
        ).to_float(self.dtype)
        self.post_quant_conv = nn.Conv2d(
            embed_dim, ddconfig["z_channels"], 1, pad_mode="valid", has_bias=True
        ).to_float(self.dtype)
        self.embed_dim = embed_dim

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.split = ops.Split(axis=1, output_num=2)
        self.exp = ops.Exp()
        self.stdnormal = ops.StandardNormal()

    def init_from_ckpt(self, path, ignore_keys=list(), remove_prefix=["first_stage_model.", "autoencoder."]):
        # TODO: support auto download pretrained checkpoints
        sd = ms.load_checkpoint(path)
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        vae_prefix = ["encoder.", "decoder.", "quant_conv.", "post_quant_conv."]
        for pname in keys:
            is_vae_param = False
            for pf in remove_prefix:
                if pname.startswith(pf):
                    sd[pname.replace(pf, "")] = sd.pop(pname)
                    is_vae_param = True
            for pf in vae_prefix:
                if pname.startswith(pf):
                    is_vae_param = True
            if not is_vae_param:
                sd.pop(pname)
        ms.load_param_into_net(self, sd, strict_load=False)
        print(f"Restored from {path}")

    def _encode(self, x):
        # return latent distribution, N(mean, logvar)
        h = self.encoder(x)
        moments = self.quant_conv(h)
        mean, logvar = self.split(moments)

        return mean, logvar

    def sample(self, mean, logvar):
        # sample z from latent distribution
        logvar = ops.clip_by_value(logvar, -30.0, 20.0)
        std = self.exp(0.5 * logvar)
        z = mean + std * self.stdnormal(mean.shape)

        return z

    def encode(self, x):
        # embedding, get latent representation z
        posterior_mean, posterior_logvar = self._encode(x)
        z = self.sample(posterior_mean, posterior_logvar)

        return z

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def construct(self, input):
        # overall pass, mostly for training
        posterior_mean, posterior_logvar = self._encode(input)
        z = self.sample(posterior_mean, posterior_logvar)

        recons = self.decode(z)

        return recons, posterior_mean, posterior_logvar
