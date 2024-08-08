import mindspore as ms
from mindspore import nn, ops
import numpy as np

from videogvt.models.vqvae.model_utils import (
    ResnetBlock3D,
    CausalConv3d,
    GroupNormExtend,
    SpatialDownsample2x,
    TimeDownsample2x,
    nonlinearity,
    _get_selected_flags,
)


class Encoder3D(nn.Cell):
    """
    This is the q_theta (z|x) network. Given a data sample x q_theta
    maps to the latent space x -> z.

    For a VQ VAE, q_theta outputs parameters of a categorical distribution.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, config, dtype=ms.float32):
        super(Encoder3D, self).__init__()

        self.config = config

        self.in_channels = self.config.vqvae.channels  # 3
        self.out_channels = self.config.vqvae.middle_channels  # 18
        self.init_dim = self.config.vqvae.filters  # 128
        self.input_conv_kernel_size = (3, 3, 3)  # (7, 7, 7)
        self.output_conv_kernel_size = (1, 1, 1)

        self.filters = self.config.vqvae.filters
        self.num_res_blocks = self.config.vqvae.num_enc_res_blocks
        self.channel_multipliers = self.config.vqvae.channel_multipliers
        self.temporal_downsample = self.config.vqvae.temporal_downsample
        if isinstance(self.temporal_downsample, int):
            self.temporal_downsample = _get_selected_flags(
                len(self.channel_multipliers) - 1, self.temporal_downsample, False
            )
        self.embedding_dim = self.config.vqvae.embedding_dim
        self.downsample = self.config.vqvae.get("downsample", "time+spatial")
        self.custom_conv_padding = self.config.vqvae.get("custom_conv_padding")
        self.norm_type = self.config.vqvae.norm_type
        self.num_remat_block = self.config.vqvae.get("num_enc_remat_blocks", 0)

        dim_gp = self.filters * self.channel_multipliers[-1]

        self.conv_out = CausalConv3d(
            dim_gp,
            self.out_channels,
            self.output_conv_kernel_size,
            padding=0,
            dtype=dtype,
        )
        self.residual_stack = nn.SequentialCell()
        self.norm = GroupNormExtend(dim_gp, dim_gp, dtype=dtype)

        num_blocks = len(self.channel_multipliers)
        for i in range(num_blocks):
            filters = self.filters * self.channel_multipliers[i]

            if i == 0:
                dim_in = self.filters
                t_stride = (1, 2, 2)
            else:
                dim_in = self.filters * self.channel_multipliers[i - 1]
                t_stride = (2, 2, 2)

            self.residual_stack.append(ResnetBlock3D(dim_in, filters, dtype=dtype))

            for _ in range(self.num_res_blocks - 1):
                self.residual_stack.append(ResnetBlock3D(filters, filters, dtype=dtype))

            if self.temporal_downsample[i]:
                if self.downsample == "conv":
                    self.residual_stack.append(
                        CausalConv3d(
                            filters,
                            filters,
                            kernel_size=(3, 3, 3),
                            stride=t_stride,
                            padding=1,
                            dtype=dtype,
                        )
                    )
                elif self.downsample == "time+spatial":
                    if t_stride[0] > 1:
                        self.residual_stack.append(
                            TimeDownsample2x(filters, filters, dtype=dtype)
                        )
                    self.residual_stack.append(
                        SpatialDownsample2x(filters, filters, dtype=dtype)
                    )
                else:
                    raise NotImplementedError(f"Unknown downsampler: {self.downsample}")

    def construct(self, x):
        # x = self.conv_in(x)
        x = self.residual_stack(x)
        x = self.norm(x)
        x = nonlinearity(x)
        x = self.conv_out(x)
        return x
