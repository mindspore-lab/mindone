import mindspore as ms
import mindspore.nn as nn


class AEKLLoss(nn.Cell):
    def __init__(
        self,
        first_stage_model,
        discriminator,
    ):
        super().__init__()

        self.first_stage_model = first_stage_model
        self.discriminator = discriminator
        # save loss for each step
        self.loss = ms.Parameter(ms.Tensor(0, dtype=first_stage_model.dtype), requires_grad=False)
        self.recons = ms.Parameter(ms.Tensor(0, dtype=first_stage_model.dtype), requires_grad=False)
        self.kl = ms.Parameter(ms.Tensor(0, dtype=first_stage_model.dtype), requires_grad=False)
        self.perceptual = ms.Parameter(ms.Tensor(0, dtype=first_stage_model.dtype), requires_grad=False)
        self.nll_loss = ms.Parameter(ms.Tensor(0, dtype=first_stage_model.dtype), requires_grad=False)
        self.adv_d_real = ms.Parameter(ms.Tensor(0, dtype=first_stage_model.dtype), requires_grad=False)
        self.adv_d_fake = ms.Parameter(ms.Tensor(0, dtype=first_stage_model.dtype), requires_grad=False)

    def construct(self, 
        x,
        global_step, 
        d_weight,
        is_generator,
        return_loss,
        return_nll,
        return_g,
    ):
        z, mean, var, logvar = self.first_stage_model.encode(x)
        yhat = self.first_stage_model.decode(z)

        loss = recons = perceptual = kl = nll_loss = adv_d_real = adv_d_fake = 0.
        if is_generator:
            loss, recons, perceptual, kl, nll_loss, adv_d_real = self.discriminator(
                x,
                yhat,
                z, mean, var, logvar,
                global_step,
                d_weight=d_weight,
                is_generator=is_generator
            )
            self.loss = (loss)
            self.recons = (recons)
            self.kl = (kl)
            self.perceptual = (perceptual)
            self.nll_loss = (nll_loss)
            self.adv_d_real = (adv_d_real)
        else:
            adv_d_fake = self.discriminator(
                x,
                yhat,
                z, mean, var, logvar,
                global_step,
                d_weight=d_weight,
                is_generator=is_generator
            )
            self.adv_d_fake = (adv_d_fake)

        # below adapts to mindspore grad operation design
        if return_loss:
            return loss
        elif return_nll:
            return nll_loss
        else:
            return adv_d_real
