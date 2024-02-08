from gm.models.trainer_factory import LatentDiffusionWithLossGrad
from gm.util import append_dims

from mindspore import dtype as ms_dtype
from mindspore import nn, ops


class LatentDiffusionWithLoss(nn.Cell):
    def __init__(self, model, scaler):
        super(LatentDiffusionWithLoss, self).__init__()
        self.model = model.model
        self.denoiser = model.denoiser
        self.loss_fn = model.loss_fn
        self.scaler = scaler

    def construct(self, x, noised_input, sigmas, w, concat, context, y, num_frames):
        c_skip, c_out, c_in, c_noise = self.denoiser(sigmas, noised_input.ndim)
        model_output = self.model(
            ops.cast(noised_input * c_in, ms_dtype.float32),
            ops.cast(c_noise, ms_dtype.int32),
            concat=concat,
            context=context,
            y=y,
            num_frames=num_frames,
        )
        model_output = model_output * c_out + noised_input * c_skip
        loss = self.loss_fn(model_output, x, w)
        loss = loss.mean()
        return self.scaler.scale(loss)


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
        ema=None,
    ):
        super(TrainOneStepCell, self).__init__()
        ldm_with_loss = LatentDiffusionWithLoss(model, scaler)
        self.conditioner = model.conditioner

        self.ldm_with_loss_grad = LatentDiffusionWithLossGrad(
            ldm_with_loss,
            optimizer,
            scaler,
            reducer,
            overflow_still_update,
            gradient_accumulation_steps,
            clip_grad,
            clip_norm,
            ema,
        )

        # first stage model
        self.scale_factor = model.scale_factor
        self.first_stage_model = model.first_stage_model

        self.sigma_sampler = model.sigma_sampler
        self.loss_fn = model.loss_fn
        self.denoiser = model.denoiser

    def construct(self, x, *tokens):
        num_frames = x.shape[1]
        cond_frames_without_noise, fps_id, motion_bucket_id, cond_frames, cond_aug = tokens
        # merge batch dimension with num_frames
        x = x.reshape(-1, *x.shape[2:])
        fps_id = fps_id.reshape(-1, *fps_id.shape[2:])
        motion_bucket_id = motion_bucket_id.reshape(-1, *motion_bucket_id.shape[2:])
        cond_aug = cond_aug.reshape(-1, *cond_aug.shape[2:])

        tokens = (cond_frames_without_noise, fps_id, motion_bucket_id, cond_frames, cond_aug)

        # get latent target
        x = self.first_stage_model.encode(x)
        x = self.scale_factor * x

        # get noise and sigma
        sigmas = self.sigma_sampler(x.shape[0])
        noise = ops.randn_like(x)
        noised_input = self.loss_fn.get_noise_input(x, noise, sigmas)
        w = append_dims(self.denoiser.w(sigmas), x.ndim)

        # compute loss
        vector, crossattn, concat = self.conditioner(*tokens)
        context, y = crossattn, vector
        loss, _, overflow = self.ldm_with_loss_grad(x, noised_input, sigmas, w, concat, context, y, num_frames)

        return loss, overflow
