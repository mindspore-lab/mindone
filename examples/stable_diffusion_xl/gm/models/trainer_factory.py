# This file only applies to static graph mode

from gm.util import append_dims

import mindspore as ms
from mindspore import nn, ops
from mindspore.boost.grad_accumulation import gradient_accumulation_op as _grad_accum_op
from mindspore.boost.grad_accumulation import gradient_clear_op as _grad_clear_op
from mindspore.ops import functional as F


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

        # get conditioner trainable status
        trainable_conditioner = False
        for embedder in model.conditioner.embedders:
            if embedder.is_trainable:
                trainable_conditioner = True
                print(f"Build Trainer: conditioner {type(embedder).__name__} is trainable.")

        # train net
        if not trainable_conditioner:
            ldm_with_loss = LatentDiffusionWithLoss(model, scaler)
            self.conditioner = model.conditioner
        else:
            ldm_with_loss = LatentDiffusionWithConditionerAndLoss(model, scaler)
            self.conditioner = None
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

        # first stage model
        self.scale_factor = model.scale_factor
        self.first_stage_model = model.first_stage_model

        #
        self.sigma_sampler = model.sigma_sampler
        self.loss_fn = model.loss_fn
        self.denoiser = model.denoiser

    def construct(self, x, *tokens):
        # get latent target
        x = self.first_stage_model.encode(x)
        x = self.scale_factor * x

        # get noise and sigma
        sigmas = self.sigma_sampler(x.shape[0])
        noise = ops.randn_like(x)
        noised_input = self.loss_fn.get_noise_input(x, noise, sigmas)
        w = append_dims(self.denoiser.w(sigmas), x.ndim)

        # compute loss
        if self.conditioner:
            # get condition
            vector, crossattn, concat = self.conditioner(*tokens)
            context, y = crossattn, vector
            loss, _, overflow = self.ldm_with_loss_grad(x, noised_input, sigmas, w, concat, context, y)
        else:
            loss, _, overflow = self.ldm_with_loss_grad(x, noised_input, sigmas, w, *tokens)

        return loss, overflow


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


class LatentDiffusionWithConditionerAndLoss(nn.Cell):
    def __init__(self, model, scaler):
        super(LatentDiffusionWithConditionerAndLoss, self).__init__()
        self.model = model.model
        self.conditioner = model.conditioner
        self.denoiser = model.denoiser
        self.loss_fn = model.loss_fn
        self.scaler = scaler

    def construct(self, x, noised_input, sigmas, w, *tokens):
        # get condition
        vector, crossattn, concat = self.conditioner(*tokens)
        context, y = crossattn, vector

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

        self.clip_grad = clip_grad
        self.clip_norm = clip_norm

        self.enable_accum_grad = False
        if grad_accum_steps > 1:
            self.enable_accum_grad = True
            self.accum_steps = grad_accum_steps
            self.accum_step = ms.Parameter(ms.Tensor(0, dtype=ms.int32), name="accum_step")
            self.accumulated_grads = optimizer.parameters.clone(prefix="accum_grad", init="zeros")
            self.hyper_map = ops.HyperMap()

    def do_optim(self, loss, grads):
        if not self.enable_accum_grad:
            if self.clip_grad:
                grads = ops.clip_by_global_norm(grads, self.clip_norm)
            loss = F.depend(loss, self.optimizer(grads))
        else:
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


# TODO: Below is experimental
class TrainerMultiGraphTwoStage:
    def __init__(self, engine, optimizers, reducers, scaler, overflow_still_update=True, amp_level="O2"):
        # get conditioner trainable status
        self.trainable_conditioner = False
        for embedder in engine.conditioner.embedders:
            if embedder.is_trainable:
                self.trainable_conditioner = True
                print(f"Build Trainer: conditioner {type(embedder).__name__} is trainable.")

        if self.trainable_conditioner:
            self.pre_process = PreProcessModelWithoutConditioner(engine)
            self.stage1_fp = LatentDiffusionStage1WithConditioner(engine)
        else:
            self.pre_process = PreProcessModel(engine)
            self.stage1_fp = LatentDiffusionStage1(engine)

        # set module train status
        self.pre_process.set_train(False)
        self.pre_process.set_grad(False)
        self.stage1_fp.set_train(False)
        self.stage1_fp.set_grad(False)

        # diffusion multi-stage model
        optimizer1, optimizer2 = optimizers
        reducer1, reducer2 = reducers
        self.stage1_with_grad = LatentDiffusionStage1Grad(
            self.stage1_fp, optimizer1, scaler, reducer1, overflow_still_update=overflow_still_update
        )
        stage2_fp = LatentDiffusionStage2WithLoss(engine, scaler)
        self.stage2_with_grad = LatentDiffusionStage2Grad(
            stage2_fp, optimizer2, scaler, reducer2, overflow_still_update=overflow_still_update
        )
        self.stage1_with_grad.set_train(True)
        self.stage2_with_grad.set_train(True)

        if amp_level:
            from gm.util import auto_mixed_precision

            self.pre_process = auto_mixed_precision(self.pre_process, amp_level)
            self.stage1_fp = auto_mixed_precision(self.stage1_fp, amp_level)
            self.stage1_with_grad = auto_mixed_precision(self.stage1_with_grad, amp_level)
            self.stage2_with_grad = auto_mixed_precision(self.stage2_with_grad, amp_level)

            disable_first_stage_amp = engine.disable_first_stage_amp
            if disable_first_stage_amp:
                self.pre_process.first_stage_model.to_float(ms.float32)

    def __call__(self, x, *tokens):
        if self.trainable_conditioner:
            return self.train_unet_conditioner(x, *tokens)
        else:
            return self.train_unet(x, *tokens)

    def train_unet_conditioner(self, x, *tokens):
        diffusion_inputs = self.pre_process(x)

        self.stage1_fp.set_train(False)
        self.stage1_fp.set_grad(False)
        out_stage1 = self.stage1_fp(*diffusion_inputs, *tokens)

        loss, grads_i, unscaled_grads2, overflow_tag2 = self.stage2_with_grad(*out_stage1)

        self.stage1_with_grad.network.set_train(True)
        self.stage1_with_grad.network.set_grad(True)
        _, unscaled_grads1, overflow_tag1 = self.stage1_with_grad(*diffusion_inputs, *tokens, *grads_i)

        overflow_tag = overflow_tag1 or overflow_tag2
        # unscaled_grads = unscaled_grads1 + unscaled_grads2  # TODO: enable it if need grad

        return loss, overflow_tag

    def train_unet(self, x, *tokens):
        diffusion_inputs = self.pre_process(x, *tokens)

        self.stage1_fp.set_train(False)
        self.stage1_fp.set_grad(False)
        out_stage1 = self.stage1_fp(*diffusion_inputs)

        loss, grads_i, unscaled_grads2, overflow_tag2 = self.stage2_with_grad(*out_stage1)

        self.stage1_with_grad.network.set_train(True)
        self.stage1_with_grad.network.set_grad(True)
        _, unscaled_grads1, overflow_tag1 = self.stage1_with_grad(*diffusion_inputs, *grads_i)

        overflow_tag = overflow_tag1 or overflow_tag2
        # unscaled_grads = unscaled_grads1 + unscaled_grads2  # TODO: enable it if need grad

        return loss, overflow_tag


class PreProcessModel(nn.Cell):
    def __init__(self, engine):
        super(PreProcessModel, self).__init__()

        # first stage model
        self.scale_factor = engine.scale_factor
        self.first_stage_model = engine.first_stage_model

        # others
        self.sigma_sampler = engine.sigma_sampler
        self.conditioner = engine.conditioner
        self.loss_fn = engine.loss_fn
        self.denoiser = engine.denoiser

    @ms.jit
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

        return x, noised_input, sigmas, w, context, y


class PreProcessModelWithoutConditioner(nn.Cell):
    def __init__(self, engine):
        super(PreProcessModelWithoutConditioner, self).__init__()

        # first stage model
        self.scale_factor = engine.scale_factor
        self.first_stage_model = engine.first_stage_model

        # others
        self.sigma_sampler = engine.sigma_sampler
        self.loss_fn = engine.loss_fn
        self.denoiser = engine.denoiser

    @ms.jit
    def construct(self, x):
        # get latent target
        x = self.first_stage_model.encode(x)
        x = self.scale_factor * x

        # get noise and sigma
        sigmas = self.sigma_sampler(x.shape[0])
        noise = ops.randn_like(x)
        noised_input = self.loss_fn.get_noise_input(x, noise, sigmas)
        w = append_dims(self.denoiser.w(sigmas), x.ndim)

        return x, noised_input, sigmas, w


class LatentDiffusionStage1(nn.Cell):
    def __init__(self, engine):
        super(LatentDiffusionStage1, self).__init__()
        self.stage1 = engine.stage1
        self.denoiser = engine.denoiser

    @ms.jit
    def construct(self, x, noised_input, sigmas, w, context, y):
        c_skip, c_out, c_in, c_noise = self.denoiser(sigmas, noised_input.ndim)
        stage1_outputs = self.stage1(
            ops.cast(noised_input * c_in, ms.float32),
            ops.cast(c_noise, ms.int32),
            context=context,
            y=y,
        )

        x, noised_input, w, c_out, c_skip = (
            ops.stop_gradient(x),
            ops.stop_gradient(noised_input),
            ops.stop_gradient(w),
            ops.stop_gradient(c_out),
            ops.stop_gradient(c_skip),
        )

        outs = (x, noised_input, w, c_out, c_skip) + stage1_outputs

        return outs


class LatentDiffusionStage1WithConditioner(nn.Cell):
    def __init__(self, engine):
        super(LatentDiffusionStage1WithConditioner, self).__init__()
        self.conditioner = engine.conditioner
        self.stage1 = engine.stage1
        self.denoiser = engine.denoiser

    @ms.jit
    def construct(self, x, noised_input, sigmas, w, *tokens):
        # get condition
        vector, crossattn, concat = self.conditioner(*tokens)
        context, y = crossattn, vector

        c_skip, c_out, c_in, c_noise = self.denoiser(sigmas, noised_input.ndim)
        stage1_outputs = self.stage1(
            ops.cast(noised_input * c_in, ms.float32),
            ops.cast(c_noise, ms.int32),
            context=context,
            y=y,
        )

        x, noised_input, w, c_out, c_skip = (
            ops.stop_gradient(x),
            ops.stop_gradient(noised_input),
            ops.stop_gradient(w),
            ops.stop_gradient(c_out),
            ops.stop_gradient(c_skip),
        )

        outs = (x, noised_input, w, c_out, c_skip) + stage1_outputs

        return outs


class LatentDiffusionStage2WithLoss(nn.Cell):
    def __init__(self, engine, scaler):
        super(LatentDiffusionStage2WithLoss, self).__init__()
        self.stage2 = engine.stage2
        self.loss_fn = engine.loss_fn
        self.scaler = scaler

    @ms.jit
    def construct(self, x, noised_input, w, c_out, c_skip, *stage2_inputs):
        model_output = self.stage2(*stage2_inputs)
        model_output = model_output * c_out + noised_input * c_skip
        loss = self.loss_fn(model_output, x, w)
        loss = loss.mean()
        return self.scaler.scale(loss)


class LatentDiffusionStage1Grad(nn.Cell):
    def __init__(self, network, optimizer, scaler, reducer, overflow_still_update=True):
        super(LatentDiffusionStage1Grad, self).__init__()
        self.optimizer = optimizer
        self.scaler = scaler
        self.reducer = reducer
        self.overflow_still_update = overflow_still_update

        self.network = network
        self.network.set_grad()
        self.weights = self.optimizer.parameters
        self.grad_fn = ops.GradOperation(get_all=False, get_by_list=True, sens_param=True)

    @ms.jit
    def construct(self, *args):
        inputs = args[:-17]
        sens = args[-17:]
        outs = self.network(*inputs)
        grads = self.grad_fn(self.network, self.weights)(*inputs, sens)
        grads = self.reducer(grads)
        unscaled_grads = self.scaler.unscale(grads)
        grads_finite = ms.amp.all_finite(unscaled_grads)

        if self.overflow_still_update:
            outs = ops.depend(outs, self.optimizer(unscaled_grads))
        else:
            if grads_finite:
                outs = ops.depend(outs, self.optimizer(unscaled_grads))

        overflow_tag = not grads_finite
        return outs, unscaled_grads, overflow_tag


class LatentDiffusionStage2Grad(nn.Cell):
    def __init__(self, network, optimizer, scaler, reducer, overflow_still_update=True):
        super(LatentDiffusionStage2Grad, self).__init__()
        self.optimizer = optimizer
        self.scaler = scaler
        self.reducer = reducer
        self.overflow_still_update = overflow_still_update

        self.network = network
        self.network.set_grad()
        self.weights = self.optimizer.parameters
        self.grad_fn = ops.GradOperation(get_all=True, get_by_list=True, sens_param=False)

    @ms.jit
    def construct(self, *args):
        inputs = args[:]
        loss = self.network(*inputs)
        grads_i, grad_w = self.grad_fn(self.network, self.weights)(*inputs)
        grads = self.reducer(grad_w)
        unscaled_grads = self.scaler.unscale(grads)
        grads_finite = ms.amp.all_finite(unscaled_grads)

        if self.overflow_still_update:
            loss = ops.depend(loss, self.optimizer(unscaled_grads))
        else:
            if grads_finite:
                loss = ops.depend(loss, self.optimizer(unscaled_grads))

        overflow_tag = not grads_finite
        return self.scaler.unscale(loss), grads_i, unscaled_grads, overflow_tag
