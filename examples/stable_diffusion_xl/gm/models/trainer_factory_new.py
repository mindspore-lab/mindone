# This file only applies to static graph mode

from gm.util import append_dims

import mindspore as ms
from mindspore import nn, ops


class LatentDiffusionWithLoss(nn.Cell):
    def __init__(self, model):
        super(LatentDiffusionWithLoss, self).__init__()
        self.model = model.model
        self.denoiser = model.denoiser
        self.loss_fn = model.loss_fn

        # first stage model
        self.scale_factor = model.scale_factor
        self.disable_first_stage_amp = model.disable_first_stage_amp
        self.first_stage_model = model.first_stage_model
        if self.disable_first_stage_amp:
            self.first_stage_model.to_float(ms.float32)

        self.sigma_sampler = model.sigma_sampler
        self.conditioner = model.conditioner
        self.loss_fn = model.loss_fn
        self.denoiser = model.denoiser

    def get_first_stage_encoding(self, x):
        if self.disable_first_stage_amp:
            x = x.to(ms.float32)
        else:
            x = x.to(ms.float16)
        z = ops.stop_gradient(self.first_stage_model.encode(x))
        z = self.scale_factor * z
        return z

    def construct(self, x, *tokens):
        # get latent target
        x = self.get_first_stage_encoding(x)

        # get condition
        vector, crossattn, concat = self.conditioner(*tokens)
        context, y = crossattn, vector

        # get noise and sigma
        sigmas = self.sigma_sampler(x.shape[0])
        noise = ops.randn_like(x)
        noised_input = self.loss_fn.get_noise_input(x, noise, sigmas)
        w = append_dims(self.denoiser.w(sigmas), x.ndim)

        c_skip, c_out, c_in, c_noise = self.denoiser(sigmas, noised_input.ndim)
        model_output = self.model(
            ops.cast(noised_input * c_in, ms.float32),
            ops.cast(c_noise, ms.int32),
            concat=concat,
            context=context,
            y=y,
        )
        model_output = model_output * c_out + noised_input * c_skip
        loss = self.loss_fn(model_output, x, w)
        loss = loss.mean()
        return loss


class LatentDiffusionWithLossDreamBooth(LatentDiffusionWithLoss):
    def __init__(self, model, prior_loss_weight=1.0):
        super().__init__(model)
        self.prior_loss_weight = prior_loss_weight

    def construct(self, x, reg_x, *all_tokens):
        assert len(all_tokens) % 2 == 0
        position = len(all_tokens) // 2
        tokens, reg_tokens = all_tokens[:position], all_tokens[position:]
        loss_train = super().construct(x, *tokens)
        loss_reg = super().construct(reg_x, *reg_tokens)
        loss = loss_train + self.prior_loss_weight * loss_reg
        return loss
