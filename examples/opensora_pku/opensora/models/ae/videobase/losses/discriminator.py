import functools

import mindspore as ms
from mindspore import nn, ops


class NLayerDiscriminator(nn.Cell):
    """Defines a PatchGAN discriminator as in Pix2Pix
    --> refer to: https://github.com/junyanz/pyms-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False, dtype=ms.float32):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """

        # TODO: check forward consistency!!!
        super().__init__()
        self.dtype = dtype
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            # norm_layer = ActNorm
            raise NotImplementedError
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        # Fixed
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, pad_mode="pad", padding=padw, has_bias=True).to_float(
                self.dtype
            ),
            nn.LeakyReLU(0.2),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    pad_mode="pad",
                    padding=padw,
                    has_bias=use_bias,
                ).to_float(self.dtype),
                norm_layer(ndf * nf_mult, momentum=0.1),
                nn.LeakyReLU(0.2),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                pad_mode="pad",
                padding=padw,
                has_bias=use_bias,
            ).to_float(self.dtype),
            norm_layer(ndf * nf_mult, momentum=0.1),
            nn.LeakyReLU(0.2),
        ]

        # output 1 channel prediction map
        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, pad_mode="pad", padding=padw, has_bias=True).to_float(
                self.dtype
            )
        ]
        self.main = nn.SequentialCell(sequence)
        self.cast = ops.Cast()

    def construct(self, x):
        y = self.main(x)
        return y


class NLayerDiscriminator3D(nn.Cell):
    """Defines a 3D PatchGAN discriminator as in Pix2Pix but for 3D inputs."""

    def __init__(self, input_nc=1, ndf=64, n_layers=3, use_actnorm=False, dtype=ms.float32):
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

        kw = 3
        padw = 1
        sequence = [
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, pad_mode="pad", padding=padw, has_bias=True),
            nn.LeakyReLU(0.2, True),
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
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
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
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw, pad_mode="pad")
        ]  # output 1 channel prediction map
        self.main = nn.SequentialCell(*sequence)

    def construct(self, input):
        """Standard forward."""
        return self.main(input)
