# This file only applies to static graph mode

from gm.util import append_dims

import mindspore as ms
from mindspore import nn, ops
from mindspore.boost.grad_accumulation import gradient_accumulation_op as _grad_accum_op
from mindspore.boost.grad_accumulation import gradient_clear_op as _grad_clear_op
from mindspore.ops import functional as F


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


class LatentDiffusionWithLossDreamBooth(LatentDiffusionWithLoss):
    def __init__(self, model, scaler, prior_loss_weight=1.0):
        super().__init__(model, scaler)
        self.prior_loss_weight = prior_loss_weight

    def construct(
        self,
        x,
        noised_input,
        sigmas,
        w,
        concat,
        context,
        y,
        reg_x,
        reg_noised_input,
        reg_sigmas,
        reg_w,
        reg_concat,
        reg_context,
        reg_y,
    ):
        loss_train = super(LatentDiffusionWithLossDreamBooth, self).construct(
            x, noised_input, sigmas, w, concat, context, y
        )
        loss_reg = super(LatentDiffusionWithLossDreamBooth, self).construct(
            reg_x, reg_noised_input, reg_sigmas, reg_w, reg_concat, reg_context, reg_y
        )
        loss = loss_train + self.prior_loss_weight * loss_reg
        return self.scaler.scale(loss)


class LatentDiffusionWithLossGrad(nn.Cell):
    def __init__(
        self,
        network,
        optimizer,
        scaler,
        reducer,
        overflow_still_update=True,
        grad_accum_steps=1,
        clip_grad=False,
        clip_norm=1.0,
    ):
        super(LatentDiffusionWithLossGrad, self).__init__()
        self.grad_fn = ops.value_and_grad(network, grad_position=None, weights=optimizer.parameters)
        self.optimizer = optimizer
        self.scaler = scaler
        self.reducer = reducer
        self.overflow_still_update = overflow_still_update

        self.accum_steps = grad_accum_steps
        self.accum_step = ms.Parameter(ms.Tensor(0, dtype=ms.int32), name="accum_step")
        self.accumulated_grads = optimizer.parameters.clone(prefix="accum_grad", init="zeros")
        self.hyper_map = ops.HyperMap()
        self.clip_grad = clip_grad
        self.clip_norm = clip_norm

    def do_optim(self, loss, grads):
        self.accum_step += 1
        loss = F.depend(
            loss, self.hyper_map(F.partial(_grad_accum_op, self.accum_steps), self.accumulated_grads, grads)
        )
        if self.accum_step % self.accum_steps == 0:
            if self.clip_grad:
                grads = ops.clip_by_global_norm(self.accumulated_grads, self.clip_norm)
                loss = F.depend(loss, self.optimizer(grads))
            else:
                loss = F.depend(loss, self.optimizer(self.accumulated_grads))
            loss = F.depend(loss, self.hyper_map(F.partial(_grad_clear_op), self.accumulated_grads))
        else:
            # update the learning rate, do not update the parameter
            loss = F.depend(loss, self.optimizer.get_lr())
        return loss

    def construct(self, *inputs):
        loss, grads = self.grad_fn(*inputs)
        grads = self.reducer(grads)
        unscaled_grads = self.scaler.unscale(grads)
        grads_finite = ms.amp.all_finite(unscaled_grads)

        if self.overflow_still_update:
            loss = self.do_optim(loss, unscaled_grads)
        else:
            if grads_finite:
                loss = self.do_optim(loss, unscaled_grads)

        overflow_tag = not grads_finite
        return self.scaler.unscale(loss), unscaled_grads, overflow_tag


class TrainOneStepCell(nn.Cell):
    def __init__(
        self,
        model,
        optimizer,
        reducer,
        scaler,
        overflow_still_update=True,
        gradient_accumulation_steps=1,
        clip_grad=False,
        clip_norm=1.0,
    ):
        super(TrainOneStepCell, self).__init__()

        # train net
        ldm_with_loss = LatentDiffusionWithLoss(model, scaler)
        self.ldm_with_loss_grad = LatentDiffusionWithLossGrad(
            ldm_with_loss,
            optimizer,
            scaler,
            reducer,
            overflow_still_update,
            gradient_accumulation_steps,
            clip_grad,
            clip_norm,
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


class TrainOneStepCellDreamBooth(nn.Cell):
    def __init__(
        self,
        model,
        optimizer,
        reducer,
        scaler,
        overflow_still_update=True,
        gradient_accumulation_steps=1,
        prior_loss_weight=1.0,
    ):
        super(TrainOneStepCellDreamBooth, self).__init__()

        # train net
        ldm_with_loss = LatentDiffusionWithLossDreamBooth(model, scaler, prior_loss_weight)
        self.ldm_with_loss_grad = LatentDiffusionWithLossGrad(
            ldm_with_loss, optimizer, scaler, reducer, overflow_still_update, gradient_accumulation_steps
        )

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

    def _get_inputs(self, x, *tokens):
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

        return x, noised_input, sigmas, w, concat, context, y

    def construct(self, x, reg_x, *all_tokens):
        assert len(all_tokens) % 2 == 0
        position = len(all_tokens) // 2
        tokens, reg_tokens = all_tokens[:position], all_tokens[position:]
        x, noised_input, sigmas, w, concat, context, y = self._get_inputs(x, *tokens)
        reg_x, reg_noised_input, reg_sigmas, reg_w, reg_concat, reg_context, reg_y = self._get_inputs(
            reg_x, *reg_tokens
        )

        # compute loss
        loss, _, overflow = self.ldm_with_loss_grad(
            x,
            noised_input,
            sigmas,
            w,
            concat,
            context,
            y,
            reg_x,
            reg_noised_input,
            reg_sigmas,
            reg_w,
            reg_concat,
            reg_context,
            reg_y,
        )

        return loss, overflow
