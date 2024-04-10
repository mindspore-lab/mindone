import mindspore as ms
from mindspore import nn, ops

from .lpips import LPIPS


def _rearrange_in(x):
    b, c, t, h, w = x.shape
    x = x.permute(0, 2, 1, 3, 4)
    x = ops.reshape(x, (b * t, c, h, w))

    return x


class GeneratorWithLoss(nn.Cell):
    def __init__(
        self,
        autoencoder,
        disc_start=50001,
        kl_weight=1.0e-06,
        disc_weight=0.5,
        disc_factor=1.0,
        perceptual_weight=1.0,
        logvar_init=0.0,
        discriminator=None,
        dtype=ms.float32,
    ):
        super().__init__()

        # build perceptual models for loss compute
        self.autoencoder = autoencoder
        # TODO: set dtype for LPIPS ?
        self.perceptual_loss = LPIPS()  # freeze params inside

        self.l1 = nn.L1Loss(reduction="none")
        # TODO: is self.logvar trainable?
        self.logvar = ms.Parameter(ms.Tensor([logvar_init], dtype=dtype))

        self.disc_start = disc_start
        self.kl_weight = kl_weight
        self.disc_weight = disc_weight
        self.disc_factor = disc_factor
        self.perceptual_weight = perceptual_weight

        self.discriminator = discriminator
        # assert discriminator is None, "Discriminator is not supported yet"

    def kl(self, mean, logvar):
        # cast to fp32 to avoid overflow in exp and sum ops.
        mean = mean.astype(ms.float32)
        logvar = logvar.astype(ms.float32)

        var = ops.exp(logvar)
        kl_loss = 0.5 * ops.sum(
            ops.pow(mean, 2) + var - 1.0 - logvar,
            dim=[1, 2, 3],
        )
        return kl_loss

    def loss_function(self, x, recons, mean, logvar, global_step: ms.Tensor = -1, weights: ms.Tensor = None, cond=None):
        bs = x.shape[0]

        # 2.1 reconstruction loss in pixels
        rec_loss = self.l1(x, recons)

        # 2.2 perceptual loss
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(x, recons)
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / ops.exp(self.logvar) + self.logvar
        if weights is not None:
            weighted_nll_loss = weights * nll_loss
            mean_weighted_nll_loss = weighted_nll_loss.sum() / bs
            # mean_nll_loss = nll_loss.sum() / bs
        else:
            mean_weighted_nll_loss = nll_loss.sum() / bs
            # mean_nll_loss = mean_weighted_nll_loss

        # 2.3 kl loss
        kl_loss = self.kl(mean, logvar)
        kl_loss = kl_loss.sum() / bs

        loss = mean_weighted_nll_loss + self.kl_weight * kl_loss

        # 2.4 discriminator loss if enabled
        # g_loss = ms.Tensor(0., dtype=ms.float32)
        # TODO: how to get global_step?
        if global_step >= self.disc_start:
            if (self.discriminator is not None) and (self.disc_factor > 0.0):
                # calc gan loss
                if cond is None:
                    logits_fake = self.discriminator(recons)
                else:
                    logits_fake = self.discriminator(ops.concat((recons, cond), dim=1))
                g_loss = -ops.reduce_mean(logits_fake)
                # TODO: do adaptive weighting based on grad
                # d_weight = self.calculate_adaptive_weight(mean_nll_loss, g_loss, last_layer=last_layer)
                d_weight = self.disc_weight
                loss += d_weight * self.disc_factor * g_loss
        # print(f"nll_loss: {mean_weighted_nll_loss.asnumpy():.4f}, kl_loss: {kl_loss.asnumpy():.4f}")

        """
        split = "train"
        log = {"{}/total_loss".format(split): loss.asnumpy().mean(),
           "{}/logvar".format(split): self.logvar.value().asnumpy(),
           "{}/kl_loss".format(split): kl_loss.asnumpy().mean(),
           "{}/nll_loss".format(split): nll_loss.asnumpy().mean(),
           "{}/rec_loss".format(split): rec_loss.asnumpy().mean(),
           # "{}/d_weight".format(split): d_weight.detach(),
           # "{}/disc_factor".format(split): torch.tensor(disc_factor),
           # "{}/g_loss".format(split): g_loss.detach().mean(),
           }
        for k in log:
            print(k.split("/")[1], log[k])
        """
        # TODO: return more losses

        return loss

    # in graph mode, construct code will run in graph. TODO: in pynative mode, need to add ms.jit decorator
    def construct(self, x: ms.Tensor, global_step: ms.Tensor = -1, weights: ms.Tensor = None, cond=None):
        """
        x: input images or videos, images: (b c h w), videos: (b c t h w)
        weights: sample weights
        global_step: global training step
        """

        # 1. AE forward, get posterior (mean, logvar) and recons
        recons, mean, logvar = self.autoencoder(x)

        # For videos, treat them as independent frame images
        # TODO: regularize on temporal consistency
        if x.ndim >= 5:
            # x: b c t h w -> (b*t c h w), shape for image perceptual loss
            x = _rearrange_in(x)
            recons = _rearrange_in(recons)
            # mean and var kl loss

        # 2. compuate loss
        loss = self.loss_function(x, recons, mean, logvar, global_step, weights, cond)

        return loss


class DiscriminatorWithLoss(nn.Cell):
    """
    Training logic:
        For training step i, input data x:
            1. AE generator takes input x, feedforward to get posterior/latent and reconstructed data, and compute ae loss
            2. AE optimizer updates AE trainable params
            3. D takes the same input x, feed x to AE again **again** to get
                the new posterior and reconstructions (since AE params has updated), feed x and recons to D, and compute D loss
            4. D optimizer updates D trainable params
            --> Go to next training step
        Ref: sd-vae training
    """

    def __init__(
        self,
        autoencoder,
        discriminator,
        disc_start=50001,
        disc_factor=1.0,
        disc_loss="hinge",
    ):
        super().__init__()
        self.autoencoder = autoencoder
        self.discriminator = discriminator
        self.disc_start = disc_start
        self.disc_factor = disc_factor

        assert disc_loss in ["hinge", "vanilla"]
        if disc_loss == "hinge":
            self.disc_loss = self.hinge_loss
        else:
            self.softplus = ops.Softplus()
            self.disc_loss = self.vanilla_d_loss

    def hinge_loss(self, logits_real, logits_fake):
        loss_real = ops.mean(ops.relu(1.0 - logits_real))
        loss_fake = ops.mean(ops.relu(1.0 + logits_fake))
        d_loss = 0.5 * (loss_real + loss_fake)
        return d_loss

    def vanilla_d_loss(self, logits_real, logits_fake):
        d_loss = 0.5 * (ops.mean(self.softplus(-logits_real)) + ops.mean(self.softplus(logits_fake)))
        return d_loss

    def construct(self, x: ms.Tensor, global_step=-1, cond=None):
        """
        Second pass
        Args:
            x: input image/video, (bs c h w)
            weights: sample weights
        """

        # 1. AE forward, get posterior (mean, logvar) and recons
        recons, mean, logvar = ops.stop_gradient(self.autoencoder(x))

        if x.ndim >= 5:
            # TODO: use 3D discriminator
            # x: b c t h w -> (b*t c h w), shape for image perceptual loss
            x = _rearrange_in(x)
            recons = _rearrange_in(recons)

        # 2. Disc forward to get class prediction on real input and reconstrucions
        if cond is None:
            logits_real = self.discriminator(x)
            logits_fake = self.discriminator(recons)
        else:
            logits_real = self.discriminator(ops.concat((x, cond), dim=1))
            logits_fake = self.discriminator(ops.concat((recons, cond), dim=1))

        if global_step >= self.disc_start:
            disc_factor = self.disc_factor
        else:
            disc_factor = 0.0

        d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

        # log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
        #        "{}/logits_real".format(split): logits_real.detach().mean(),
        #       "{}/logits_fake".format(split): logits_fake.detach().mean()
        #       }

        return d_loss
