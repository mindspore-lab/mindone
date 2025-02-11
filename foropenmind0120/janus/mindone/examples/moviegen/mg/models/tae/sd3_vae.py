import mindspore as ms
from mindspore import nn, ops

from .modules_2d import Decoder, Encoder

# TODO: set z_channels to 16
SD3d5_CONFIG = {
    "double_z": True,
    "z_channels": 16,
    "resolution": 256,
    "in_channels": 3,
    "out_ch": 3,
    "ch": 128,
    "ch_mult": [1, 2, 4, 4],
    "num_res_blocks": 2,
    "attn_resolutions": [],
    "dropout": 0.0,
    "scaling_factor": 1.5305,
    "shift_factor": 0.0609,
    "use_post_quant_conv": False,
    "use_quant_conv": False,
}


class SD3d5_VAE(nn.Cell):
    def __init__(
        self,
        config: dict = SD3d5_CONFIG,
        pretrained: str = None,
        use_recompute: bool = False,
        sample_deterministic: bool = False,
    ):
        super().__init__()

        # encoder
        self.encoder = Encoder(**config)

        # quant and post quant
        embed_dim = config["z_channels"]
        if config["use_quant_conv"]:
            self.quant_conv = nn.Conv2d(2 * embed_dim, 2 * embed_dim, 1, pad_mode="valid", has_bias=True)
        if config["use_post_quant_conv"]:
            self.post_quant_conv = nn.Conv2d(embed_dim, embed_dim, 1, pad_mode="valid", has_bias=True)

        self.use_quant_conv = config["use_quant_conv"]
        self.use_post_quant_conv = config["use_post_quant_conv"]

        # decoder
        self.decoder = Decoder(**config)

        self.exp = ops.Exp()
        self.stdnormal = ops.StandardNormal()
        self.split = ms.ops.split

        self.sample_deterministic = sample_deterministic

        if use_recompute:
            # self.recompute(self.encoder)
            # self.recompute(self.quant_conv)
            # self.recompute(self.post_quant_conv)
            self.recompute(self.decoder)

    def recompute(self, b):
        if not b._has_config_recompute:
            b.recompute()
        if isinstance(b, nn.CellList):
            self.recompute(b[-1])
        else:
            b.add_flags(output_no_recompute=True)

    def _encode(self, x):
        # return latent distribution, N(mean, logvar)
        h = self.encoder(x)
        if self.use_quant_conv:
            moments = self.quant_conv(h)
        else:
            moments = h
        mean, logvar = self.split(moments, moments.shape[1] // 2, 1)

        return mean, logvar

    def sample(self, mean, logvar):
        # sample z from latent distribution
        logvar = ops.clip_by_value(logvar, -30.0, 20.0)
        std = self.exp(0.5 * logvar)
        z = mean + std * self.stdnormal(mean.shape)

        return z

    def encode(self, x: ms.Tensor) -> ms.Tensor:
        # embedding, get latent representation z
        posterior_mean, posterior_logvar = self._encode(x)
        if self.sample_deterministic:
            return posterior_mean
        z = self.sample(posterior_mean, posterior_logvar)

        return z

    def decode(self, z: ms.Tensor) -> ms.Tensor:
        if self.use_post_quant_conv:
            z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        """
        video reconstruction

        x: (b c h w)
        """

        posterior_mean, posterior_logvar = self._encode(x)
        z = self.sample(posterior_mean, posterior_logvar)
        recons = self.decode(z)

        return recons, z, posterior_mean, posterior_logvar

    def load_pretrained(self, ckpt_path: str):
        if ckpt_path.endswith("safetensors"):
            # load vae parameters from safetensors into my mindspore model
            import safetensors

            ckpt = safetensors.safe_open(ckpt_path, framework="pt")
            state_dict = {}
            for key in ckpt.keys():
                state_dict[key] = ckpt.get_tensor(key)
            raise NotImplementedError
        else:
            param_dict = ms.load_checkpoint(ckpt_path)
            param_not_load, ckpt_not_load = ms.load_param_into_net(self, param_dict, strict_load=True)
            if param_not_load or ckpt_not_load:
                print(f"{param_not_load} in network is not loaded or {ckpt_not_load} in checkpoint is not loaded!")
        print("vae checkpoint loaded")
