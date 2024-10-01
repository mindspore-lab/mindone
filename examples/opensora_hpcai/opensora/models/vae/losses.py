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
        kl_weight=1.0e-06,
        perceptual_weight=1.0,
        logvar_init=0.0,
        use_real_rec_loss=False,
        use_z_rec_loss=False,
        use_image_identity_loss=False,
        dtype=ms.float32,
    ):
        super().__init__()

        # build perceptual models for loss compute
        self.autoencoder = autoencoder
        # TODO: set dtype for LPIPS ?
        self.perceptual_loss = LPIPS()  # freeze params inside

        # self.l1 = nn.L1Loss(reduction="none")
        # TODO: is self.logvar trainable?
        self.logvar = ms.Parameter(ms.Tensor([logvar_init], dtype=ms.float32))

        self.kl_weight = kl_weight
        self.perceptual_weight = perceptual_weight

        self.use_real_rec_loss = use_real_rec_loss
        self.use_z_rec_loss = use_z_rec_loss
        self.use_image_identity_loss = use_image_identity_loss

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

    def vae_loss_fn(
        self, x, recons, mean, logvar, nll_weights=None, no_perceptual=False, no_kl=False, pixelwise_mean=False
    ):
        """
        return:
            nll_loss: weighted sum of pixel reconstruction loss and perceptual loss
            weighted_nll_loss:  weighted mean of nll_loss
            weighted_kl_loss: KL divergence on posterior
        """
        bs = x.shape[0]
        # (b c t h w) -> (b*t c h w)
        x = _rearrange_in(x)
        recons = _rearrange_in(recons)

        # reconstruction loss in pixels
        # FIXME: debugging: use pixelwise mean to reduce loss scale
        if pixelwise_mean:
            rec_loss = ((x - recons) ** 2).mean()
        else:
            rec_loss = ops.abs(x - recons)

        # perceptual loss
        if (self.perceptual_weight > 0) and (not no_perceptual):
            p_loss = self.perceptual_loss(x, recons)
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / ops.exp(self.logvar) + self.logvar
        if nll_weights is not None:
            weighted_nll_loss = nll_weights * nll_loss
            weighted_nll_loss = weighted_nll_loss.sum() / bs
        else:
            weighted_nll_loss = nll_loss.sum() / bs

        # kl loss
        # TODO: FIXME: it may not fit for graph mode training
        if (self.kl_weight > 0) and (not no_kl):
            kl_loss = self.kl(mean, logvar)
            kl_loss = kl_loss.sum() / bs
            weighted_kl_loss = self.kl_weight * kl_loss
        else:
            weighted_kl_loss = 0

        return nll_loss, weighted_nll_loss, weighted_kl_loss

    def construct(self, x: ms.Tensor, global_step: ms.Tensor = -1, weights: ms.Tensor = None, cond=None):
        """
        x: input images or videos, images: (b c 1 h w), videos: (b c t h w)
        weights: sample weights
        global_step: global training step
        """
        print("D--: x shape: ", x.shape)
        # 3d vae forward, get posterior (mean, logvar) and recons
        # x -> VAE2d-Enc -> x_z -> TemporalVAE-Enc -> z ~ posterior -> TempVAE-Dec -> x_z_rec -> VAE2d-Dec -> x_rec
        x_rec, x_z_rec, z, posterior_mean, posterior_logvar, x_z = self.autoencoder(x)
        # FIXME: debugging
        x_rec, x_z_rec, z, posterior_mean, posterior_logvar, x_z = (
            x_rec.to(ms.float32),
            x_z_rec.to(ms.float32),
            z.to(ms.float32),
            posterior_mean.to(ms.float32),
            posterior_logvar.to(ms.float32),
            x_z.to(ms.float32),
        )

        frames = x.shape[2]

        # Loss compute
        # 1. VAE 2d, video frames x reconstruction loss
        # TODO: loss dtype setting
        if self.use_real_rec_loss:
            # x: (b 3 t h w)
            _, weighted_nll_loss, weighted_kl_loss = self.vae_loss_fn(
                x, x_rec, posterior_mean, posterior_logvar, no_perceptual=False
            )
            loss = weighted_nll_loss + weighted_kl_loss
        else:
            loss = 0

        # 2. temporal vae, spatial latent x_z reconstruction loss
        if self.use_z_rec_loss:
            # x_z: (b 4 t h//8 w//8)
            # NOTE: since KL loss on posterior is the same as that in part 1. We can skip it.
            _, weighted_nll_loss_z, _ = self.vae_loss_fn(
                x_z, x_z_rec, posterior_mean, posterior_logvar, no_perceptual=True, no_kl=True
            )
            loss += weighted_nll_loss_z

        # 3. identity regularization loss for pure image input
        if self.use_image_identity_loss and frames == 1:
            _, image_identity_loss, _ = self.vae_loss_fn(
                x_z, z, posterior_mean, posterior_logvar, no_perceptual=True, no_kl=True
            )
            loss += image_identity_loss

        return loss


# Discriminator is not used in opensora v1.2
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
