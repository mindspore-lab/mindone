from videogvt.models.quantization import LFQ, LFQ2d
from videogvt.models.vqvae import enc_dec_2dcnn, enc_dec_3dcnn

import mindspore as ms
from mindspore import nn

from .model_utils import CausalConv3d, pad_at_dim


class VQVAE_3D(nn.Cell):
    def __init__(
        self,
        config,
        is_training=True,
        dtype=ms.float32,
    ):
        super().__init__()

        self.config = config.vqvae
        self.dtype = dtype
        self.encoder = enc_dec_3dcnn.Encoder(config=self.config, dtype=self.dtype)
        self.decoder = enc_dec_3dcnn.Decoder(config=self.config, dtype=self.dtype)
        self.quant_conv = CausalConv3d(self.config.embedding_dim, self.config.embedding_dim, 1, dtype=dtype)
        self.post_quant_conv = CausalConv3d(self.config.embedding_dim, self.config.embedding_dim, 1, dtype=dtype)
        self.quantizer = LFQ(config=config.lfq, is_training=is_training, dtype=self.dtype)

        self.time_downsample_factor = 2 ** sum(self.config.temporal_downsample)
        self.patch_size = (self.time_downsample_factor, 1, 1)
        self.out_channels = self.config.channels
        self.num_frames = self.config.num_frames

        self.dtype = dtype

    def encode(self, x):
        time_padding = (
            0
            if (x.shape[2] % self.time_downsample_factor == 0)
            else self.time_downsample_factor - x.shape[2] % self.time_downsample_factor
        )
        x = pad_at_dim(x, (time_padding, 0), dim=2)
        encoded_feature = self.encoder(x)
        z_e = self.quant_conv(encoded_feature).to(x.dtype)
        return z_e

    def decode(self, z):
        time_padding = (
            0
            if (self.num_frames % self.time_downsample_factor == 0)
            else self.time_downsample_factor - self.num_frames % self.time_downsample_factor
        )
        z = self.post_quant_conv(z)
        x = self.decoder(z)
        x = x[:, :, time_padding:]
        return x

    def _forward(self, x):
        # encode
        z_e = self.encode(x)

        # quantization
        embed_dtype = z_e.dtype
        z_e = z_e.astype(self.dtype)
        z_q, indices, aux_loss = self.quantizer(z_e)

        # decode
        z_q = z_q.astype(embed_dtype)
        recon_video = self.decode(z_q)
        return recon_video

    def construct(self, x):
        # encode
        z_e = self.encode(x)

        # quantization
        embed_dtype = z_e.dtype
        z_e = z_e.astype(self.dtype)
        z_q, indices, aux_loss = self.quantizer(z_e)

        # decode
        z_q = z_q.astype(embed_dtype)
        recon_video = self.decode(z_q)

        return z_e, z_q, recon_video, aux_loss


class VQVAE_2D(nn.Cell):
    def __init__(
        self,
        config,
        is_training=True,
        dtype=ms.float32,
    ):
        super().__init__()

        self.config = config.vqvae

        self.space_downsample_factor = 2 ** sum(self.config.spatial_downsample)
        self.patch_size = (self.space_downsample_factor, 1, 1)
        self.out_channels = self.config.channels

        # NOTE: following MAGVIT, conv in bias=False in encoder first conv
        self.encoder = enc_dec_2dcnn.Encoder(self.config, dtype=dtype)
        self.decoder = enc_dec_2dcnn.Decoder(self.config, dtype=dtype)
        self.quant_conv = nn.Conv2d(self.config.embedding_dim, self.config.embedding_dim, 1, dtype=dtype)
        self.post_quant_conv = nn.Conv2d(self.config.embedding_dim, self.config.embedding_dim, 1, dtype=dtype)

        self.quantizer = LFQ2d(
            config=config.lfq,
            is_training=is_training,
            dtype=dtype,
        )

    def encode(self, x):
        encoded_feature = self.encoder(x)
        z_e = self.quant_conv(encoded_feature).to(x.dtype)
        return z_e

    def decode(self, z):
        z = self.post_quant_conv(z)
        x = self.decoder(z)
        return x

    def _forward(self, x):
        # encode
        z_e = self.encode(x)

        # quantization
        z_q, _, _ = self.quantizer(z_e)

        # decode
        recon_video = self.decode(z_q)

        return recon_video

    def construct(self, x):
        # encode
        z_e = self.encode(x)

        # quantization
        z_q, indices, aux_loss = self.quantizer(z_e)

        # decode
        recon_video = self.decode(z_q)

        return z_e, z_q, recon_video, aux_loss
