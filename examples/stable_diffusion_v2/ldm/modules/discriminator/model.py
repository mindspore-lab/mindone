import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from ldm.modules.discriminator.lpips import LPIPS
from ldm.modules.discriminator.modules import (
    hinge_d_loss,
    vanilla_d_loss,
    weights_init,
    adopt_weight,
    NLayerDiscriminator,
)


def kl_diagonal_normal(mean, var, logvar, other_mean=0., other_var=1., other_logvar=0.):
    return 0.5 * ops.reduce_sum(
        ops.pow(mean - other_mean, 2) / other_var
        + var / other_var - 1.0 - logvar + other_logvar,
        [1, 2, 3])


class LPIPSWithDiscriminator(nn.Cell):
    def __init__(
        self,
        disc_start,
        logvar_init=0.0,
        kl_weight=1e-6,
        pixelloss_weight=1.0,
        disc_num_layers=3,
        disc_in_channels=3,
        disc_factor=1.0,
        disc_weight=1.0,
        perceptual_weight=1.0,
        use_actnorm=False,
        disc_conditional=False,
        disc_loss="hinge",
        use_fp16=False,
    ):
        super().__init__()
        self.dtype = ms.float16 if use_fp16 else ms.float32
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS(dtype=self.dtype).to_float(self.dtype)
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = ms.Parameter(ms.Tensor([logvar_init], dtype=self.dtype))

        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels,
            n_layers=disc_num_layers,
            use_actnorm=use_actnorm,
            dtype=self.dtype
        ).to_float(self.dtype)#.apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.l1 = nn.L1Loss(reduction='none')
        self.grad_fn = ops.GradOperation()
        self.cast = ops.Cast()

    def construct(
        self,
        inputs,
        reconstructions,
        z, mean,
        var,
        logvar,
        global_step, 
        d_weight=1., cond=None, weights=None, is_generator=False
    ):
        d_weight = self.cast(d_weight, self.dtype)
        if not is_generator:
            if cond is None:
                logits_real = self.discriminator(ops.stop_gradient(inputs))
                logits_fake = self.discriminator(ops.stop_gradient(reconstructions))
            else:
                logits_real = self.discriminator(ops.concat((ops.stop_gradient(inputs), cond), axis=1))
                logits_fake = self.discriminator(ops.concat((ops.stop_gradient(reconstructions), cond), axis=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            return d_loss

        rec_loss = self.l1(inputs, reconstructions)
        p_loss = ms.Tensor(0, dtype=self.dtype)
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs, reconstructions)
            rec_loss = rec_loss + self.perceptual_weight * p_loss
            p_loss = p_loss.mean()
        recons = rec_loss.mean()

        nll_loss = rec_loss / ops.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights * nll_loss
        weighted_nll_loss = ops.reduce_sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = ops.reduce_sum(nll_loss) / nll_loss.shape[0]

        kl_loss = kl_diagonal_normal(mean, var, logvar, other_mean=0., other_var=1., other_logvar=0.)
        kl_loss = ops.reduce_sum(kl_loss) / kl_loss.shape[0]

        # generator update
        if cond is None:
            assert not self.disc_conditional
            logits_fake = self.discriminator(reconstructions)
        else:
            assert self.disc_conditional
            logits_fake = self.discriminator(ops.concat((reconstructions, cond), dim=1))
        g_loss = -ops.reduce_mean(logits_fake)

        disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
        loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * self.discriminator_weight * disc_factor * g_loss
        return loss, recons, p_loss, self.kl_weight * kl_loss, nll_loss, g_loss
