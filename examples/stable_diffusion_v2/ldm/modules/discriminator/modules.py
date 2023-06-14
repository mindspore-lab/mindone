import functools
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


class ActNorm(nn.Cell):
    def __init__(self, num_features, logdet=False, affine=True,
                 allow_reverse_init=False):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(ms.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(ms.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init

        self.register_buffer('initialized', ms.Tensor(0, dtype=ms.uint8))

    def initialize(self, input):
        flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
        mean = (
            flatten.mean(1)
            .unsqueeze(1)
            .unsqueeze(2)
            .unsqueeze(3)
            .permute(1, 0, 2, 3)
        )
        std = (
            flatten.std(1)
            .unsqueeze(1)
            .unsqueeze(2)
            .unsqueeze(3)
            .permute(1, 0, 2, 3)
        )

        self.loc.assign_value(ms.ops.stop_gradient(-mean))
        self.scale.assign_value(ms.ops.stop_gradient(1 / (std + 1e-6)))

    def construct(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = ms.log(ms.abs(self.scale))
            logdet = height*width*ms.sum(log_abs)
            logdet = logdet * ms.ones(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output):
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm in reverse direction is "
                    "disabled by default. Use allow_reverse_init=True to enable."
                )
            else:
                self.initialize(output)
                self.initialized.fill_(1)

        if len(output.shape) == 2:
            output = output[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h


def hinge_d_loss(logits_real, logits_fake):
    relu = ops.ReLU()
    loss_real = (relu(1. - logits_real)).mean()
    loss_fake = (relu(1. + logits_fake)).mean()
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (ms.mean(ms.nn.functional.softplus(-logits_real)) +
                    ms.mean(ms.nn.functional.softplus(logits_fake)))
    return d_loss


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


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
        super().__init__()
        self.dtype = dtype
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, pad_mode='pad', padding=padw).to_float(self.dtype),
                    nn.LeakyReLU(0.2)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, pad_mode='pad', padding=padw, has_bias=use_bias).to_float(self.dtype),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, pad_mode='pad', padding=padw, has_bias=use_bias).to_float(self.dtype),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, pad_mode='pad', padding=padw).to_float(self.dtype)]  # output 1 channel prediction map
        self.main = nn.SequentialCell(sequence)
        self.cast = ops.Cast()

    def construct(self, x):
        y = self.main(x)
        return y
