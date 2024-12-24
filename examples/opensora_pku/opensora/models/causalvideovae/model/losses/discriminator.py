import functools
import math

from opensora.npu_config import npu_config

import mindspore as ms
from mindspore import nn
from mindspore.common.initializer import HeUniform, Uniform


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class NLayerDiscriminator3D(nn.Cell):
    """Defines a 3D PatchGAN discriminator as in Pix2Pix but for 3D inputs."""

    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False, dtype=ms.float32):
        """
        Construct a 3D PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input volumes
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            use_actnorm (bool) -- flag to use actnorm instead of batchnorm
        """
        super(NLayerDiscriminator3D, self).__init__()
        self.dtype = dtype
        if not use_actnorm:
            norm_layer = nn.BatchNorm3d
        else:
            raise NotImplementedError("Not implemented.")
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != nn.BatchNorm3d
        else:
            use_bias = norm_layer != nn.BatchNorm3d
        conv_dtype = npu_config.conv_dtype if npu_config is not None else ms.bfloat16

        kw = 3
        padw = 1
        sequence = [
            nn.Conv3d(
                input_nc,
                ndf,
                kernel_size=kw,
                stride=2,
                pad_mode="pad",
                padding=padw,
                has_bias=True,
                weight_init=HeUniform(negative_slope=math.sqrt(5)),
                bias_init=Uniform(scale=1 / math.sqrt(input_nc * kw * kw * kw)),
                dtype=conv_dtype,
            ),
            nn.LeakyReLU(0.2).to_float(self.dtype),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv3d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=(kw, kw, kw),
                    stride=(2 if n == 1 else 1, 2, 2),
                    padding=padw,
                    pad_mode="pad",
                    has_bias=use_bias,
                    weight_init=HeUniform(negative_slope=math.sqrt(5)),
                    bias_init=Uniform(scale=1 / math.sqrt(ndf * nf_mult_prev * kw * kw * kw)),
                    dtype=conv_dtype,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2).to_float(self.dtype),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv3d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=(kw, kw, kw),
                stride=1,
                padding=padw,
                pad_mode="pad",
                has_bias=use_bias,
                weight_init=HeUniform(negative_slope=math.sqrt(5)),
                bias_init=Uniform(scale=1 / math.sqrt(ndf * nf_mult_prev * kw * kw * kw)),
                dtype=conv_dtype,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2).to_float(self.dtype),
        ]

        sequence += [
            nn.Conv3d(
                ndf * nf_mult,
                1,
                kernel_size=kw,
                stride=1,
                padding=padw,
                pad_mode="pad",
                has_bias=True,
                weight_init=HeUniform(negative_slope=math.sqrt(5)),
                bias_init=Uniform(scale=1 / math.sqrt(ndf * nf_mult * kw * kw * kw)),
                dtype=conv_dtype,
            )
        ]  # output 1 channel prediction map
        self.main = nn.CellList(sequence)

    def construct(self, input):
        """Standard forward."""
        x = input
        for layer in self.main:
            if isinstance(layer, nn.Conv3d):
                x = npu_config.run_conv3d(layer, x, x.dtype)
            elif isinstance(layer, nn.BatchNorm3d):
                x = npu_config.run_batch_norm(layer, x)
            else:
                x = layer(x)
        return x
