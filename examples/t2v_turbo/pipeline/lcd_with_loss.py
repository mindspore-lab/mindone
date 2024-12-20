import logging

from utils.common_utils import (
    append_dims,
    get_predicted_noise,
    get_predicted_original_sample,
    guidance_scale_embedding,
    huber_loss,
    scalings_for_boundary_conditions,
)

import mindspore as ms
from mindspore import Tensor, _no_grad, jit_class, mint, nn, ops

__all__ = ["LCDWithLoss"]

logger = logging.getLogger(__name__)


@jit_class
class no_grad(_no_grad):
    """
    A context manager that suppresses gradient memory allocation in PyNative mode.
    """

    def __init__(self):
        super().__init__()
        self._pynative = ms.get_context("mode") == ms.PYNATIVE_MODE

    def __enter__(self):
        if self._pynative:
            super().__enter__()

    def __exit__(self, *args):
        if self._pynative:
            super().__exit__(*args)


class LCDWithLoss(nn.Cell):
    def __init__(
        self,
        vae: nn.Cell,
        text_encoder: nn.Cell,
        teacher_unet: nn.Cell,
        unet: nn.Cell,
        noise_scheduler,
        alpha_schedule,
        sigma_schedule,
        weight_dtype,
        vae_scale_factor,
        time_cond_proj_dim,
        args,
        solver,
        uncond_prompt_embeds,
        reward_fn,
        video_rm_fn,
    ):
        super().__init__()

        self.weight_dtype = weight_dtype

        self.unet = unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.teacher_unet = teacher_unet

        self.noise_scheduler = noise_scheduler
        self.alpha_schedule = alpha_schedule
        self.sigma_schedule = sigma_schedule
        self.vae_scale_factor = vae_scale_factor
        self.time_cond_proj_dim = time_cond_proj_dim
        self.args = args
        self.solver = solver
        self.uncond_prompt_embeds = uncond_prompt_embeds

        self.reward_fn = reward_fn
        self.video_rm_fn = video_rm_fn

    def compute_embeddings(self, text_tokens):
        prompt_embeds = self.text_encoder(text_tokens)
        return prompt_embeds

    def get_latents(self, pixel_values):
        """
        pixel_values: (b, t, c, h, w)
        """
        b, t = pixel_values.shape[:2]
        pixel_values_flatten = pixel_values.view(b * t, *pixel_values.shape[2:])
        # encode pixel values with batch size of at most args.vae_encode_batch_size
        latents = []
        for i in range(0, self.args.n_frames, self.args.vae_encode_batch_size):
            latents.append(self.vae.encode(pixel_values_flatten[i : i + self.args.vae_encode_batch_size]).sample())
        latents = mint.cat(latents, dim=0)
        latents = latents.reshape(b, t, latents.shape[-3], latents.shape[-2], latents.shape[-1])
        # Convert latents from (b, t, c, h, w) to (b, c, t, h, w)
        latents = latents.permute(0, 2, 1, 3, 4)
        latents = latents * self.vae_scale_factor

        # assert not pretrained_t2v.scale_by_std
        latents = latents.to(self.weight_dtype)

        return latents

    def _compute_image_text_rewards(self, model_pred, text, t):
        # sample args.reward_batch_size frames
        assert self.args.train_batch_size == 1
        idx = ops.randint(0, t, (self.args.reward_batch_size,))

        selected_latents = model_pred[:, :, idx].to(self.weight_dtype) / self.vae_scale_factor
        num_images = self.args.train_batch_size * self.args.reward_batch_size
        selected_latents = selected_latents.permute(0, 2, 1, 3, 4)
        selected_latents = selected_latents.reshape(num_images, *selected_latents.shape[2:])
        decoded_imgs = self.vae.decode(selected_latents)
        decoded_imgs = (decoded_imgs / 2 + 0.5).clamp(0, 1)
        expert_rewards = self.reward_fn(decoded_imgs, text)
        reward_loss = -expert_rewards.mean() * self.args.reward_scale

        return reward_loss

    def _compute_video_text_rewards(self, model_pred, text, t):
        assert self.args.train_batch_size == 1
        assert t >= self.args.video_rm_batch_size

        skip_frames = t // self.args.video_rm_batch_size
        start_id = ops.randint(0, skip_frames, (1,))[0].item()
        idx = mint.arange(start_id, t, skip_frames)[: self.args.video_rm_batch_size]
        assert len(idx) == self.args.video_rm_batch_size

        selected_latents = model_pred[:, :, idx].to(self.weight_dtype) / self.vae_scale_factor
        num_images = self.args.train_batch_size * self.args.video_rm_batch_size
        selected_latents = selected_latents.permute(0, 2, 1, 3, 4)
        selected_latents = selected_latents.reshape(num_images, *selected_latents.shape[2:])
        decoded_imgs = self.vae.decode(selected_latents)
        decoded_imgs = (decoded_imgs / 2 + 0.5).clamp(0, 1)
        decoded_imgs = decoded_imgs.reshape(
            self.args.train_batch_size,
            self.args.video_rm_batch_size,
            *decoded_imgs.shape[1:],
        )
        video_rewards = self.video_rm_fn(decoded_imgs, text)
        video_rm_loss = -video_rewards.mean() * self.args.video_reward_scale

        return video_rm_loss

    def construct(self, pixel_values: Tensor, captions: Tensor, text_tokens: Tensor):
        b, t = pixel_values.shape[:2]

        # 1. Load and process the image and text conditioning
        pixel_values = pixel_values.to(dtype=self.weight_dtype)
        latents = self.get_latents(pixel_values)

        with no_grad():
            encoded_text = self.compute_embeddings(text_tokens)

        # 2. Sample a random timestep for each image t_n from the ODE solver timesteps without bias.
        # For the DDIM solver, the timestep schedule is [T - 1, T - k - 1, T - 2 * k - 1, ...]
        bsz = latents.shape[0]
        index = ops.randint(0, self.args.num_ddim_timesteps, (bsz,))
        start_timesteps = self.solver.ddim_timesteps[index]
        timesteps = start_timesteps - self.args.topk
        timesteps = mint.where(timesteps < 0, mint.zeros_like(timesteps), timesteps)

        # 3. Get boundary scalings for start_timesteps and (end) timesteps.
        c_skip_start, c_out_start = scalings_for_boundary_conditions(
            start_timesteps, timestep_scaling=self.args.timestep_scaling_factor
        )
        c_skip_start, c_out_start = [append_dims(x, latents.ndim) for x in [c_skip_start, c_out_start]]
        c_skip, c_out = scalings_for_boundary_conditions(timesteps, timestep_scaling=self.args.timestep_scaling_factor)
        c_skip, c_out = [append_dims(x, latents.ndim) for x in [c_skip, c_out]]

        # 4. Sample noise from the prior and add it to the latents according to the noise magnitude at each
        # timestep (this is the forward diffusion process) [z_{t_{n + k}} in Algorithm 1]
        noise = ops.randn_like(latents)
        noisy_model_input = self.noise_scheduler.add_noise(latents, noise, start_timesteps)

        # 5. Sample a random guidance scale w from U[w_min, w_max] and embed it
        w = (self.args.w_max - self.args.w_min) * ops.rand((bsz,)) + self.args.w_min
        w_embedding = guidance_scale_embedding(w, embedding_dim=self.time_cond_proj_dim)
        w = w.reshape(bsz, 1, 1, 1, 1)
        # Move to U-Net device and dtype
        w = w.to(dtype=latents.dtype)
        w_embedding = w_embedding.to(dtype=latents.dtype)

        # 6. Prepare prompt embeds and unet_added_conditions
        prompt_embeds = encoded_text

        # 7. Get online LCM prediction on z_{t_{n + k}} (noisy_model_input), w, c, t_{n + k} (start_timesteps)
        context = ops.cat([prompt_embeds], 1)
        fps = 16
        noise_pred = self.unet(
            noisy_model_input,
            start_timesteps,
            context=context,
            fps=fps,
            timestep_cond=w_embedding,
        )
        pred_x_0 = get_predicted_original_sample(
            noise_pred,
            start_timesteps,
            noisy_model_input,
            "epsilon",
            self.alpha_schedule,
            self.sigma_schedule,
        )

        model_pred = c_skip_start * noisy_model_input + c_out_start * pred_x_0

        distill_loss = mint.zeros_like(model_pred).mean()
        reward_loss = mint.zeros_like(model_pred).mean()
        video_rm_loss = mint.zeros_like(model_pred).mean()

        if self.args.reward_scale > 0:
            text = captions.numpy().astype(str)
            reward_loss = self._compute_image_text_rewards(model_pred, text, t)
            logger.info(f"image reward score: {reward_loss.numpy():.4f}")

        if self.args.video_reward_scale > 0:
            text = captions.numpy().astype(str)
            video_rm_loss = self._compute_video_text_rewards(model_pred, text, t)
            logger.info(f"video reward score: {video_rm_loss.numpy():.4f}")

        # 8. Compute the conditional and unconditional teacher model predictions to get CFG estimates of the
        # predicted noise eps_0 and predicted original sample x_0, then run the ODE solver using these
        # estimates to predict the data point in the augmented PF-ODE trajectory corresponding to the next ODE
        # solver timestep.
        with no_grad():
            # 8.1. Get teacher model prediction on noisy_model_input z_{t_{n + k}} and conditional embedding c
            cond_teacher_output = self.teacher_unet(
                noisy_model_input,
                start_timesteps,
                context=context,
                fps=fps,
            )
            cond_pred_x0 = get_predicted_original_sample(
                cond_teacher_output,
                start_timesteps,
                noisy_model_input,
                "epsilon",
                self.alpha_schedule,
                self.sigma_schedule,
            )
            cond_pred_noise = get_predicted_noise(
                cond_teacher_output,
                start_timesteps,
                noisy_model_input,
                "epsilon",
                self.alpha_schedule,
                self.sigma_schedule,
            )

            # 8.2. Get teacher model prediction on noisy_model_input z_{t_{n + k}} and unconditional embedding 0
            uncond_teacher_output = self.teacher_unet(
                noisy_model_input,
                start_timesteps,
                context=self.uncond_prompt_embeds,
            )
            uncond_pred_x0 = get_predicted_original_sample(
                uncond_teacher_output,
                start_timesteps,
                noisy_model_input,
                "epsilon",
                self.alpha_schedule,
                self.sigma_schedule,
            )
            uncond_pred_noise = get_predicted_noise(
                uncond_teacher_output,
                start_timesteps,
                noisy_model_input,
                "epsilon",
                self.alpha_schedule,
                self.sigma_schedule,
            )

            # 8.3. Calculate the CFG estimate of x_0 (pred_x0) and eps_0 (pred_noise)
            # Note that this uses the LCM paper's CFG formulation rather than the Imagen CFG formulation
            pred_x0 = cond_pred_x0 + w * (cond_pred_x0 - uncond_pred_x0)
            pred_noise = cond_pred_noise + w * (cond_pred_noise - uncond_pred_noise)
            # 8.4. Run one step of the ODE solver to estimate the next point x_prev on the
            # augmented PF-ODE trajectory (solving backward in time)
            # Note that the DDIM step depends on both the predicted x_0 and source noise eps_0.
            x_prev = self.solver.ddim_step(pred_x0, pred_noise, index)

        # 9. Get target LCM prediction on x_prev, w, c, t_n (timesteps)
        with no_grad():
            target_noise_pred = self.unet(
                x_prev.float(),
                timesteps,
                context=context,
                fps=fps,
                timestep_cond=w_embedding,
            )
            pred_x_0 = get_predicted_original_sample(
                target_noise_pred,
                timesteps,
                x_prev,
                "epsilon",
                self.alpha_schedule,
                self.sigma_schedule,
            )
            target = c_skip * x_prev + c_out * pred_x_0

        # 10. Calculate loss
        if self.args.loss_type == "l2":
            distill_loss = ops.mse_loss(model_pred.float(), target.float(), reduction="mean")
        elif self.args.loss_type == "huber":
            distill_loss = huber_loss(model_pred.float(), target.float(), self.args.huber_c)

        logger.info(f"distill_loss: {distill_loss.numpy():.4f}")

        # accelerator.backward(distill_loss + reward_loss + video_rm_loss)
        loss = distill_loss + reward_loss + video_rm_loss
        return loss


class LCDWithStageLoss(nn.Cell):
    def __init__(
        self,
        vae: nn.Cell,
        text_encoder: nn.Cell,
        teacher_unet: nn.Cell,
        unet: nn.Cell,
        noise_scheduler,
        alpha_schedule,
        sigma_schedule,
        weight_dtype,
        vae_scale_factor,
        time_cond_proj_dim,
        args,
        solver,
        uncond_prompt_embeds,
        reward_fn,
        video_rm_fn,
    ):
        super().__init__()

        self.weight_dtype = weight_dtype

        self.unet = unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.teacher_unet = teacher_unet

        self.noise_scheduler = noise_scheduler
        self.alpha_schedule = alpha_schedule
        self.sigma_schedule = sigma_schedule
        self.vae_scale_factor = vae_scale_factor
        self.time_cond_proj_dim = time_cond_proj_dim
        self.args = args
        self.solver = solver
        self.uncond_prompt_embeds = uncond_prompt_embeds

        self.reward_fn = reward_fn
        self.video_rm_fn = video_rm_fn

    def compute_embeddings(self, text_tokens):
        prompt_embeds = self.text_encoder(text_tokens)
        return prompt_embeds

    def get_latents(self, pixel_values):
        """
        pixel_values: (b, t, c, h, w)
        """
        b, t = pixel_values.shape[:2]
        pixel_values_flatten = pixel_values.view(b * t, *pixel_values.shape[2:])
        # encode pixel values with batch size of at most args.vae_encode_batch_size
        latents = []
        for i in range(0, self.args.n_frames, self.args.vae_encode_batch_size):
            latents.append(self.vae.encode(pixel_values_flatten[i : i + self.args.vae_encode_batch_size]).sample())
        latents = mint.cat(latents, dim=0)
        latents = latents.reshape(b, t, latents.shape[-3], latents.shape[-2], latents.shape[-1])
        # Convert latents from (b, t, c, h, w) to (b, c, t, h, w)
        latents = latents.permute(0, 2, 1, 3, 4)
        latents = latents * self.vae_scale_factor

        # assert not pretrained_t2v.scale_by_std
        latents = latents.to(self.weight_dtype)

        return latents

    def _compute_image_text_rewards(self, model_pred, text, t):
        # sample args.reward_batch_size frames
        assert self.args.train_batch_size == 1
        idx = ops.randint(0, t, (self.args.reward_batch_size,))

        selected_latents = model_pred[:, :, idx].to(self.weight_dtype) / self.vae_scale_factor
        num_images = self.args.train_batch_size * self.args.reward_batch_size
        selected_latents = selected_latents.permute(0, 2, 1, 3, 4)
        selected_latents = selected_latents.reshape(num_images, *selected_latents.shape[2:])
        decoded_imgs = self.vae.decode(selected_latents)
        decoded_imgs = (decoded_imgs / 2 + 0.5).clamp(0, 1)
        expert_rewards = self.reward_fn(decoded_imgs, text)
        reward_loss = -expert_rewards.mean() * self.args.reward_scale

        return reward_loss

    def _compute_video_text_rewards(self, model_pred, text, t):
        assert self.args.train_batch_size == 1
        assert t >= self.args.video_rm_batch_size

        skip_frames = t // self.args.video_rm_batch_size
        start_id = ops.randint(0, skip_frames, (1,))[0].item()
        idx = mint.arange(start_id, t, skip_frames)[: self.args.video_rm_batch_size]
        assert len(idx) == self.args.video_rm_batch_size

        selected_latents = model_pred[:, :, idx].to(self.weight_dtype) / self.vae_scale_factor
        num_images = self.args.train_batch_size * self.args.video_rm_batch_size
        selected_latents = selected_latents.permute(0, 2, 1, 3, 4)
        selected_latents = selected_latents.reshape(num_images, *selected_latents.shape[2:])
        decoded_imgs = self.vae.decode(selected_latents)
        decoded_imgs = (decoded_imgs / 2 + 0.5).clamp(0, 1)
        decoded_imgs = decoded_imgs.reshape(
            self.args.train_batch_size,
            self.args.video_rm_batch_size,
            *decoded_imgs.shape[1:],
        )
        video_rewards = self.video_rm_fn(decoded_imgs, text)
        video_rm_loss = -video_rewards.mean() * self.args.video_reward_scale

        return video_rm_loss

    def construct(self, pixel_values: Tensor, captions: Tensor, text_tokens: Tensor, stage_id: int = 0):
        assert stage_id in [
            0,
            1,
        ], f"Unsupported stage_id: {stage_id}! Only support [0]: distillation + image; [1]: video loss"
        b, t = pixel_values.shape[:2]

        # 1. Load and process the image and text conditioning
        pixel_values = pixel_values.to(dtype=self.weight_dtype)
        latents = self.get_latents(pixel_values)

        with no_grad():
            encoded_text = self.compute_embeddings(text_tokens)

        # 2. Sample a random timestep for each image t_n from the ODE solver timesteps without bias.
        # For the DDIM solver, the timestep schedule is [T - 1, T - k - 1, T - 2 * k - 1, ...]
        bsz = latents.shape[0]
        index = ops.randint(0, self.args.num_ddim_timesteps, (bsz,))
        start_timesteps = self.solver.ddim_timesteps[index]
        timesteps = start_timesteps - self.args.topk
        timesteps = mint.where(timesteps < 0, mint.zeros_like(timesteps), timesteps)

        # 3. Get boundary scalings for start_timesteps and (end) timesteps.
        c_skip_start, c_out_start = scalings_for_boundary_conditions(
            start_timesteps, timestep_scaling=self.args.timestep_scaling_factor
        )
        c_skip_start, c_out_start = [append_dims(x, latents.ndim) for x in [c_skip_start, c_out_start]]
        c_skip, c_out = scalings_for_boundary_conditions(timesteps, timestep_scaling=self.args.timestep_scaling_factor)
        c_skip, c_out = [append_dims(x, latents.ndim) for x in [c_skip, c_out]]

        # 4. Sample noise from the prior and add it to the latents according to the noise magnitude at each
        # timestep (this is the forward diffusion process) [z_{t_{n + k}} in Algorithm 1]
        noise = ops.randn_like(latents)
        noisy_model_input = self.noise_scheduler.add_noise(latents, noise, start_timesteps)

        # 5. Sample a random guidance scale w from U[w_min, w_max] and embed it
        w = (self.args.w_max - self.args.w_min) * ops.rand((bsz,)) + self.args.w_min
        w_embedding = guidance_scale_embedding(w, embedding_dim=self.time_cond_proj_dim)
        w = w.reshape(bsz, 1, 1, 1, 1)
        # Move to U-Net device and dtype
        w = w.to(dtype=latents.dtype)
        w_embedding = w_embedding.to(dtype=latents.dtype)

        # 6. Prepare prompt embeds and unet_added_conditions
        prompt_embeds = encoded_text

        # 7. Get online LCM prediction on z_{t_{n + k}} (noisy_model_input), w, c, t_{n + k} (start_timesteps)
        context = ops.cat([prompt_embeds], 1)
        fps = 16
        noise_pred = self.unet(
            noisy_model_input,
            start_timesteps,
            context=context,
            fps=fps,
            timestep_cond=w_embedding,
        )
        pred_x_0 = get_predicted_original_sample(
            noise_pred,
            start_timesteps,
            noisy_model_input,
            "epsilon",
            self.alpha_schedule,
            self.sigma_schedule,
        )

        model_pred = c_skip_start * noisy_model_input + c_out_start * pred_x_0

        distill_loss = mint.zeros_like(model_pred).mean()
        reward_loss = mint.zeros_like(model_pred).mean()
        video_rm_loss = mint.zeros_like(model_pred).mean()

        if self.args.reward_scale > 0 and stage_id == 0:
            text = captions.numpy().astype(str)
            reward_loss = self._compute_image_text_rewards(model_pred, text, t)
            logger.info(f"image reward score: {reward_loss.numpy():.4f}")

        if self.args.video_reward_scale > 0 and stage_id == 1:
            text = captions.numpy().astype(str)
            video_rm_loss = self._compute_video_text_rewards(model_pred, text, t)
            logger.info(f"video reward score: {video_rm_loss.numpy():.4f}")

        if stage_id == 0:
            # 8. Compute the conditional and unconditional teacher model predictions to get CFG estimates of the
            # predicted noise eps_0 and predicted original sample x_0, then run the ODE solver using these
            # estimates to predict the data point in the augmented PF-ODE trajectory corresponding to the next ODE
            # solver timestep.
            with no_grad():
                # 8.1. Get teacher model prediction on noisy_model_input z_{t_{n + k}} and conditional embedding c
                cond_teacher_output = self.teacher_unet(
                    noisy_model_input,
                    start_timesteps,
                    context=context,
                    fps=fps,
                )
                cond_pred_x0 = get_predicted_original_sample(
                    cond_teacher_output,
                    start_timesteps,
                    noisy_model_input,
                    "epsilon",
                    self.alpha_schedule,
                    self.sigma_schedule,
                )
                cond_pred_noise = get_predicted_noise(
                    cond_teacher_output,
                    start_timesteps,
                    noisy_model_input,
                    "epsilon",
                    self.alpha_schedule,
                    self.sigma_schedule,
                )

                # 8.2. Get teacher model prediction on noisy_model_input z_{t_{n + k}} and unconditional embedding 0
                uncond_teacher_output = self.teacher_unet(
                    noisy_model_input,
                    start_timesteps,
                    context=self.uncond_prompt_embeds,
                )
                uncond_pred_x0 = get_predicted_original_sample(
                    uncond_teacher_output,
                    start_timesteps,
                    noisy_model_input,
                    "epsilon",
                    self.alpha_schedule,
                    self.sigma_schedule,
                )
                uncond_pred_noise = get_predicted_noise(
                    uncond_teacher_output,
                    start_timesteps,
                    noisy_model_input,
                    "epsilon",
                    self.alpha_schedule,
                    self.sigma_schedule,
                )

                # 8.3. Calculate the CFG estimate of x_0 (pred_x0) and eps_0 (pred_noise)
                # Note that this uses the LCM paper's CFG formulation rather than the Imagen CFG formulation
                pred_x0 = cond_pred_x0 + w * (cond_pred_x0 - uncond_pred_x0)
                pred_noise = cond_pred_noise + w * (cond_pred_noise - uncond_pred_noise)
                # 8.4. Run one step of the ODE solver to estimate the next point x_prev on the
                # augmented PF-ODE trajectory (solving backward in time)
                # Note that the DDIM step depends on both the predicted x_0 and source noise eps_0.
                x_prev = self.solver.ddim_step(pred_x0, pred_noise, index)

            # 9. Get target LCM prediction on x_prev, w, c, t_n (timesteps)
            with no_grad():
                target_noise_pred = self.unet(
                    x_prev.float(),
                    timesteps,
                    context=context,
                    fps=fps,
                    timestep_cond=w_embedding,
                )
                pred_x_0 = get_predicted_original_sample(
                    target_noise_pred,
                    timesteps,
                    x_prev,
                    "epsilon",
                    self.alpha_schedule,
                    self.sigma_schedule,
                )
                target = c_skip * x_prev + c_out * pred_x_0

            # 10. Calculate loss
            if self.args.loss_type == "l2":
                distill_loss = ops.mse_loss(model_pred.float(), target.float(), reduction="mean")
            elif self.args.loss_type == "huber":
                distill_loss = huber_loss(model_pred.float(), target.float(), self.args.huber_c)

            logger.info(f"distill_loss: {distill_loss.numpy():.4f}")

        # accelerator.backward(distill_loss + reward_loss + video_rm_loss)
        loss = distill_loss + reward_loss + video_rm_loss
        return loss
