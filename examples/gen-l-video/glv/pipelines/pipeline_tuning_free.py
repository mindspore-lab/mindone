import inspect
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import numpy as np
import PIL
from PIL import Image
from tqdm import tqdm

import mindspore as ms
from mindspore import ops

from ..models.outputs import BaseOutput
from ..util import get_views, save_videos_grid


@dataclass
class TuningFreePipelineOutput(BaseOutput):
    videos: Union[ms.Tensor, np.ndarray]


class TuningFreePipeline:
    def __init__(self, sd, unet, scheduler, depth_estimator):
        super().__init__()

        self.sd = sd
        self.unet = unet
        self.scheduler = scheduler
        self.depth_estimator = depth_estimator

        self.vae_scale_factor = 2 ** (self.sd.first_stage_model.encoder.num_resolutions - 1)

    def _encode_prompt(self, prompt, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_input = self.sd.tokenize(prompt)

        # TODO: sequence length exceeding max sequence length needs to be handled
        # TODO: encoding with attention mask needs to be handled

        text_embeddings = self.sd.get_learned_conditioning(text_input)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.tile((1, num_videos_per_prompt, 1))
        text_embeddings = text_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]

            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            uncond_input = self.sd.tokenize(uncond_tokens)

            # TODO: encoding with attention mask needs to be handled

            uncond_embeddings = self.sd.get_learned_conditioning(uncond_input)

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.tile((1, num_videos_per_prompt, 1))
            uncond_embeddings = uncond_embeddings.view(batch_size * num_videos_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = ops.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def check_inputs(self, prompt, strength, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [1.0, 1.0] but is {strength}")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def decode_latents(self, latents):
        video_length = latents.shape[2]

        latents = latents.permute(0, 2, 1, 3, 4)
        latents = latents.reshape(
            latents.shape[0] * latents.shape[1], latents.shape[2], latents.shape[3], latents.shape[4]
        )

        video = self.sd.decode_first_stage(latents)

        video = video.reshape(
            video.shape[0] // video_length, video_length, video.shape[1], video.shape[2], video.shape[3]
        )
        video = video.permute(0, 2, 1, 3, 4)

        video = (video / 2 + 0.5).clamp(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.float().asnumpy()

        return video

    def get_timesteps(self, num_inference_steps, strength):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start

    @staticmethod
    def numpy_to_pil(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]

        images = (images * 255).round().astype("uint8")

        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def prepare_depth_map(self, image, depth_map, batch_size, do_classifier_free_guidance):
        if isinstance(image, PIL.Image.Image):
            image = [image]
        else:
            image = [img for img in image]

        if depth_map is None:
            if len(image) < 20:
                depth_map = ms.Tensor(self.depth_estimator(image))
            else:
                depth_map = []

                for i in range(0, len(image), 20):
                    depth_map.append(ms.Tensor(self.depth_estimator(image[i : i + 20])))

                depth_map = ops.cat(depth_map)

        depth_map = ops.interpolate(
            depth_map.unsqueeze(1),
            size=(512 // self.vae_scale_factor, 512 // self.vae_scale_factor),
            mode="bicubic",
            align_corners=False,
        )

        depth_min = ops.amin(depth_map, axis=[1, 2, 3], keepdims=True)
        depth_max = ops.amax(depth_map, axis=[1, 2, 3], keepdims=True)
        depth_map = 2.0 * (depth_map - depth_min) / (depth_max - depth_min) - 1.0

        save_videos_grid((depth_map + 1) / 2, "./depth_map.gif")

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if depth_map.shape[0] < batch_size:
            depth_map = depth_map.tile((batch_size, 1, 1, 1))

        depth_map = ops.cat([depth_map] * 2) if do_classifier_free_guidance else depth_map  # classifier free guidance

        return depth_map

    def prepare_extra_step_kwargs(self, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}

        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        return extra_step_kwargs

    def prepare_latents(self, batch_size, num_channels_latents, video_length, height, width, dtype, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )

        if latents is None:
            latents = ops.randn(shape, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        return latents

    def __call__(
        self,
        prompt: Union[str, List[str]],
        image: Union[ms.Tensor, PIL.Image.Image],
        video_length: Optional[int],
        height: Optional[int] = None,
        width: Optional[int] = None,
        depth_map: Optional[ms.Tensor] = None,
        strength: float = 1.0,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        latents: Optional[ms.Tensor] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, ms.Tensor], None]] = None,
        callback_steps: Optional[int] = 1,
        use_l2=False,
        window_size: Optional[int] = 16,
        stride: Optional[int] = 8,
        **kwargs,
    ):
        # 1. Check inputs
        self.check_inputs(prompt, strength, callback_steps)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_embeddings = self._encode_prompt(
            prompt, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # 4. Preprocess image
        depth_mask = self.prepare_depth_map(
            image,
            depth_map,
            batch_size * num_videos_per_prompt,
            do_classifier_free_guidance,
        )

        depth_mask = depth_mask.reshape(
            depth_mask.shape[0] // video_length,
            video_length,
            depth_mask.shape[1],
            depth_mask.shape[2],
            depth_mask.shape[3],
        ).permute(0, 2, 1, 3, 4)

        # 5. set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength)

        # 6. Prepare latent variables
        if hasattr(self.unet, "_backbone"):
            num_channels_latents = self.unet._backbone.conv_in.in_channels - 1
        else:
            num_channels_latents = self.unet.conv_in.in_channels - 1

        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            video_length,
            height,
            width,
            text_embeddings.dtype,
            latents,
        )

        # 7. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(eta)

        # 8. Denoising loop
        views = get_views(video_length, window_size=window_size, stride=stride)
        count = ops.zeros_like(latents)
        value = ops.zeros_like(latents)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        with tqdm(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                count.fill(0)
                value.fill(0)

                for t_start, t_end in views:
                    depth_mask_view = depth_mask[:, :, t_start:t_end]
                    latents_view = ops.stop_gradient(latents[:, :, t_start:t_end])

                    if use_l2:
                        depth_mask_view = ops.stop_gradient(depth_mask_view)
                        depth_mask_view.requires_grad = False

                    latent_model_input = ops.cat([latents_view] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    latent_model_input = ops.cat([latent_model_input, depth_mask_view], axis=1)

                    # predict the noise residual
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    if use_l2 and i < 25:
                        scheduler_outputs = self.scheduler.step(noise_pred, t, latents_view, **extra_step_kwargs)
                        latents_view = scheduler_outputs.prev_sample

                        for j in range(5):
                            latents_view.data[:, :, 1:] = latents_view.data[:, :, 1:] + 0.001 * (
                                latents_view.data[:, :, :-1] - latents_view.data[:, :, 1:]
                            )

                        latents_view_denoised = latents_view
                    else:
                        scheduler_outputs = self.scheduler.step(noise_pred, t, latents_view, **extra_step_kwargs)
                        latents_view_denoised = scheduler_outputs.prev_sample

                    value[:, :, t_start:t_end] += ops.stop_gradient(latents_view_denoised)
                    count[:, :, t_start:t_end] += 1

                latents = ops.where(count > 0, value / count, value)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # 1. Post-processing
        video = self.decode_latents(latents)

        # 11. Convert to PIL
        if output_type == "pil":
            video = self.numpy_to_pil(video)
        elif output_type == "tensor":
            video = ms.Tensor(video)

        if not return_dict:
            return (video,)

        return TuningFreePipelineOutput(videos=video)
