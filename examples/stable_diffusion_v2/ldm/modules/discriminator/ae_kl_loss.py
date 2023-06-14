import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np

# __all__ = [
#     'Loss_G',
# ]


class Loss_G(nn.Cell):
    def __init__(
        self,
        first_stage_model,
        D,
    ):
        super().__init__()
        self.first_stage_model = first_stage_model
        self.D = D
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
        d_weight=1.,
        G=True, 
        return_loss=True, 
        return_nll=False,
        return_g=False
    ):
        z, mean, var, logvar = self.first_stage_model.encode(x)
        yhat = self.first_stage_model.decode(z)
        if G:
            loss, recons, perceptual, kl, nll_loss, adv_d_real = self.D(
                x,
                yhat,
                z, mean, var, logvar,
                global_step,
                d_weight=d_weight,
                G=G
            )
            self.loss = (loss)
            self.recons = (recons)
            self.kl = (kl)
            self.perceptual = (perceptual)
            self.nll_loss = (nll_loss)
            self.adv_d_real = (adv_d_real)
        else:
            adv_d_fake = self.D(
                x,
                yhat,
                z, mean, var, logvar,
                global_step,
                d_weight=d_weight,
                G=G
            )
            self.adv_d_fake = (adv_d_fake)
        # below adapts to mindspore grad operation design
        if return_loss:
            return loss
        elif return_nll:
            return nll_loss
        else:
            return adv_d_real
