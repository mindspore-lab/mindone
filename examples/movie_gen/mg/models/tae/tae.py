import mindspore as ms
from mindspore import nn, ops
from .modules import Conv2_5d, Encoder, Decoder

# TODO: set z_channels to 16
SDXL_CONFIG = {
    "double_z": True,
    "z_channels": 4,
    "resolution": 256,
    "in_channels": 3,
    "out_ch": 3,
    "ch": 128,
    "ch_mult": [1, 2, 4, 4],
    "num_res_blocks": 2,
    "attn_resolutions": [],
    "dropout": 0.0,
}


class TemporalAutoencoder(nn.Cell):
    r"""
    TAE

    Parameters:
        config (`dict`): config dict
        pretrained (`str`): checkpoint path
    """

    def __init__(
        self,
        config: dict = SDXL_CONFIG,
        pretrained: str = None,
        use_recompute: bool=False,
    ):
        super().__init__()

        # encoder
        self.encoder = Encoder(**config)

        # quant and post quant
        embed_dim = config['z_channels']
        self.quant_conv = Conv2_5d(2 * config["z_channels"], 2 * embed_dim, 1, pad_mode="valid", has_bias=True)
        self.post_quant_conv = Conv2_5d(embed_dim, config["z_channels"], 1, pad_mode="valid", has_bias=True)

        # decoder
        self.decoder = Decoder(**config)

        self.exp = ops.Exp()
        self.stdnormal = ops.StandardNormal()
        self.split = ms.ops.split

        self.sample_deterministic = False
        self.discard_spurious_frames = True

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
        moments = self.quant_conv(h)
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
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        """
        video reconstruction

        x: (b c t h w)
        """

        posterior_mean, posterior_logvar = self._encode(x)
        z = self.sample(posterior_mean, posterior_logvar)
        recons = self.decode(z)

        if self.discard_spurious_frames and (recons.shape[-3] != x.shape[-3]):
            recons = recons[:, :, :x.shape[-3], :, :]

        return recons, z, posterior_mean, posterior_logvar

