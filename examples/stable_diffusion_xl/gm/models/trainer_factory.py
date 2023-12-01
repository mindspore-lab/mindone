# This file only applies to static graph mode

from gm.util import append_dims

import mindspore as ms
from mindspore import Parameter, Tensor, nn, ops


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
    def __init__(self, network, optimizer, scaler, reducer, overflow_still_update=True, gradient_accumulation_steps=1):
        super(LatentDiffusionWithLossGrad, self).__init__()
        self.grad_fn = ops.value_and_grad(network, grad_position=None, weights=optimizer.parameters)
        self.optimizer = optimizer
        self.scaler = scaler
        self.reducer = reducer
        self.overflow_still_update = overflow_still_update

        assert gradient_accumulation_steps >= 1
        self.grad_accu_steps = gradient_accumulation_steps
        if gradient_accumulation_steps > 1:
            # additionally caches network trainable parameters. overhead caused.
            # TODO: try to store it in CPU memory instead of GPU/NPU memory.
            self.accumulated_grads = optimizer.parameters.clone(prefix="grad_accumulated_", init="zeros")
            self.zeros = optimizer.parameters.clone(prefix="zeros_", init="zeros")
            self.cur_accu_step = Parameter(Tensor(0, ms.int32), "grad_accumulate_step_", requires_grad=False)
            self.zero = Tensor(0, ms.int32)
            for p in self.accumulated_grads:
                p.requires_grad = False
            for z in self.zeros:
                z.requires_grad = False
        self.map = ops.Map()
        self.partial = ops.Partial()

    def construct(self, *inputs):
        loss, grads = self.grad_fn(*inputs)
        grads = self.reducer(grads)
        unscaled_grads = self.scaler.unscale(grads)
        grads_finite = ms.amp.all_finite(unscaled_grads)

        if self.overflow_still_update or grads_finite:
            if self.grad_accu_steps > 1:
                # self.accumulated_grads += unscaled_grads
                loss = ops.depend(loss, self.map(self.partial(ops.assign_add), self.accumulated_grads, unscaled_grads))
                # self.cur_accu_step += 1
                loss = ops.depend(loss, ops.assign_add(self.cur_accu_step, Tensor(1, ms.int32)))

                if self.cur_accu_step % self.grad_accu_steps == 0:
                    loss = ops.depend(loss, self.optimizer(self.accumulated_grads))

                    # clear gradient accumulation states
                    loss = ops.depend(
                        loss, self.map(self.partial(ops.assign), self.accumulated_grads, self.zeros)
                    )  # self.accumulated_grads = 0
                    loss = ops.depend(loss, ops.assign(self.cur_accu_step, self.zero))  # self.cur_accu_step = 0
                else:
                    # update LR in each gradient step but not optimize net parameter to ensure the LR curve is
                    # consistent
                    loss = ops.depend(loss, self.optimizer.get_lr())  # .get_lr() will make lr step increased by 1
            else:
                loss = ops.depend(loss, self.optimizer(unscaled_grads))

        overflow_tag = not grads_finite
        return self.scaler.unscale(loss), unscaled_grads, overflow_tag


class TrainOneStepCell(nn.Cell):
    def __init__(self, model, optimizer, reducer, scaler, overflow_still_update=True, gradient_accumulation_steps=1):
        super(TrainOneStepCell, self).__init__()

        # train net
        ldm_with_loss = LatentDiffusionWithLoss(model, scaler)
        self.ldm_with_loss_grad = LatentDiffusionWithLossGrad(
            ldm_with_loss, optimizer, scaler, reducer, overflow_still_update, gradient_accumulation_steps
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
