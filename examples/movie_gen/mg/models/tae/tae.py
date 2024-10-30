import mindspore as ms
from mindspore import nn, ops

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


class VideoAutoencoder(nn.Cell):
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
    ):
        super().__init__()

        # encoder
        self.encoder = Encoder(**config)

        # quant and post quant
        self.quant_conv = Conv2_5d(2 * config["z_channels"], 2 * embed_dim, 1, pad_mode="valid", has_bias=True)
        self.post_quant_conv = Conv2_5d(embed_dim, config["z_channels"], 1, pad_mode="valid", has_bias=True)

        # decoder
        self.decoder = Decoder(**config)

    def encode(self, x: ms.Tensor) -> ms.Tensor:
        return x

    def decode(self, x: ms.Tensor) -> ms.Tensor:
        return x

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        """
        video reconstruction

        x: (b c t h w)
        """

        return x
