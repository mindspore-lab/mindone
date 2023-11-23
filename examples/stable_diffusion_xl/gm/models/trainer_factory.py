# This file only applies to static graph mode

from gm.util import append_dims

import mindspore as ms
from mindspore import nn, ops


class LatentDiffusionWithLoss(nn.Cell):
    def __init__(self, model, scaler):
        super(LatentDiffusionWithLoss, self).__init__()
        self.model = model.model
        self.denoiser = model.denoiser
        self.loss_fn = model.loss_fn
        self.scaler = scaler

    def construct(self, x, noised_input, sigmas, w, concat, context, y):
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
        return self.scaler.scale(loss)


class LatentDiffusionWithLossGrad(nn.Cell):
    def __init__(self, network, optimizer, scaler, reducer, overflow_still_update=True):
        super(LatentDiffusionWithLossGrad, self).__init__()
        self.grad_fn = ops.value_and_grad(network, grad_position=None, weights=optimizer.parameters)
        self.optimizer = optimizer
        self.scaler = scaler
        self.reducer = reducer
        self.overflow_still_update = overflow_still_update

    def construct(self, *inputs):
        loss, grads = self.grad_fn(*inputs)
        grads = self.reducer(grads)
        unscaled_grads = self.scaler.unscale(grads)
        grads_finite = ms.amp.all_finite(unscaled_grads)

        if self.overflow_still_update:
            loss = ops.depend(loss, self.optimizer(unscaled_grads))
        else:
            if grads_finite:
                loss = ops.depend(loss, self.optimizer(unscaled_grads))

        overflow_tag = not grads_finite
        return self.scaler.unscale(loss), unscaled_grads, overflow_tag


class TrainOneStepCell(nn.Cell):
    def __init__(self, model, optimizer, reducer, scaler, overflow_still_update=True):
        super(TrainOneStepCell, self).__init__()

        # train net
        ldm_with_loss = LatentDiffusionWithLoss(model, scaler)
        self.ldm_with_loss_grad = LatentDiffusionWithLossGrad(
            ldm_with_loss, optimizer, scaler, reducer, overflow_still_update
        )
        # scaling_sens = Tensor([1024], dtype=ms.float32)
        # self.ldm_with_loss_grad = nn.TrainOneStepWithLossScaleCell(ldm_with_loss, optimizer, scale_sense=scaling_sens)

        # first stage model
        self.scale_factor = model.scale_factor
        disable_first_stage_amp = model.disable_first_stage_amp
        self.first_stage_model = model.first_stage_model
        if disable_first_stage_amp:
            self.first_stage_model.to_float(ms.float32)

        #
        self.sigma_sampler = model.sigma_sampler
        self.conditioner = model.conditioner
        self.loss_fn = model.loss_fn
        self.denoiser = model.denoiser

    def construct(self, x, *tokens):
        # get latent target
        x = self.first_stage_model.encode(x)
        x = self.scale_factor * x

        # get condition
        vector, crossattn, concat = self.conditioner(*tokens)
        context, y = crossattn, vector

        # get noise and sigma
        sigmas = self.sigma_sampler(x.shape[0])
        noise = ops.randn_like(x)
        noised_input = self.loss_fn.get_noise_input(x, noise, sigmas)
        w = append_dims(self.denoiser.w(sigmas), x.ndim)

        # compute loss
        loss, _, overflow = self.ldm_with_loss_grad(x, noised_input, sigmas, w, concat, context, y)

        return loss, overflow
