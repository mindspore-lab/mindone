import os
import sys
from typing import Union

import numpy as np

import mindspore as ms
from mindspore import ops

from .pipelines.pipeline_tuning_free_inpaint import prepare_mask_and_masked_image

workspace = os.path.dirname(os.path.abspath(__file__))
sys.path.append(workspace + "/../../stable_diffusion_v2")


def init_prompt(prompt, pipeline):
    uncond_input = pipeline.sd.tokenize([""])
    uncond_embeddings = pipeline.sd.get_learned_conditioning(uncond_input)

    text_input = pipeline.sd.tokenize([prompt])
    text_embeddings = pipeline.sd.get_learned_conditioning(text_input)

    context = ops.cat([uncond_embeddings, text_embeddings])

    return context


def get_views(video_length, window_size=16, stride=4):
    num_blocks_time = (video_length - window_size) // stride + 1
    views = []

    for i in range(num_blocks_time):
        t_start = int(i * stride)
        t_end = t_start + window_size
        views.append((t_start, t_end))

    return views


def get_noise_pred_single(latents, t, context, unet, clip_id, control):
    return unet(latents, t, clip_id=clip_id, encoder_hidden_states=context, control=control)["sample"]


def next_step(
    model_output: Union[ms.Tensor, np.ndarray],
    timestep: int,
    sample: Union[ms.Tensor, np.ndarray],
    ddim_scheduler,
):
    timestep, next_timestep = (
        min(timestep - ddim_scheduler.ddpm_num_timesteps // len(ddim_scheduler.ddim_timesteps), 999),
        timestep,
    )

    # NOTE: ddim_scheduler.alphas_cumprod[0] in MindONE is not equal to ddim_scheduler.final_alpha_cumprod
    # if set_alpha_to_one is True for huggingface.diffusers.DDIMScheduler. Please refer to
    # https://github.com/G-U-N/Gen-L-Video/blob/master/glv/util.py#L55. This should be handled later.
    alpha_prod_t = ddim_scheduler.alphas_cumprod[timestep] if timestep >= 0 else ddim_scheduler.alphas_cumprod[0]
    alpha_prod_t_next = ddim_scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next**0.5 * next_original_sample + next_sample_direction

    return next_sample


def ddim_loop_long(pipeline, ddim_scheduler, latent, num_inv_steps, prompt, window_size, stride, control, depth_map):
    context = init_prompt(prompt, pipeline)
    _, cond_embeddings = context.chunk(2)
    all_latent = [latent]
    latent = latent.copy()
    video_length = latent.shape[2]
    views = get_views(video_length, window_size=window_size, stride=stride)
    count = ops.zeros_like(latent)
    value = ops.zeros_like(latent)

    ddim_scheduler_timesteps = ddim_scheduler.ddim_timesteps[::-1]

    for i in range(num_inv_steps):
        count.fill(0)
        value.fill(0)

        for t_start, t_end in views:
            control_tmp = None if control is None else control[:, :, t_start:t_end]
            latent_view = latent[:, :, t_start:t_end]
            t = ddim_scheduler_timesteps[len(ddim_scheduler_timesteps) - i - 1]

            if depth_map is not None:
                latent_input = ops.cat([latent_view, depth_map[:, :, t_start:t_end]], axis=1)
            else:
                latent_input = latent_view

            noise_pred = get_noise_pred_single(latent_input, t, cond_embeddings, pipeline.unet, t_start, control_tmp)
            latent_view_denoised = next_step(noise_pred, t, latent_view, ddim_scheduler)
            value[:, :, t_start:t_end] += latent_view_denoised
            count[:, :, t_start:t_end] += 1

        latent = ops.where(count > 0, value / count, value)
        all_latent.append(latent)

    return all_latent


def ddim_inversion_long(
    pipeline,
    ddim_scheduler,
    video_latent,
    num_inv_steps,
    prompt="",
    window_size=16,
    stride=8,
    control=None,
    pixel_values=None,
    mask=None,
):
    if mask is not None:
        assert pixel_values is not None
        mask, masked_image = prepare_mask_and_masked_image(pixel_values, mask)
        bz, _, video_length, height, width = video_latent.shape
        mask, masked_image_latents = pipeline.prepare_mask_latents(
            mask,
            masked_image,
            bz,
            height * pipeline.vae_scale_factor,
            width * pipeline.vae_scale_factor,
            video_latent.dtype,
            video_latent.device,
            None,
            False,
        )
        depth_map = ops.cat([mask, masked_image_latents], axis=1)
        depth_map = depth_map.reshape(
            depth_map.shape[0] // video_length,
            video_length,
            depth_map.shape[1],
            depth_map.shape[2],
            depth_map.shape[3],
        ).permute(0, 2, 1, 3, 4)
    elif pixel_values is not None and hasattr(pipeline, "prepare_depth_map"):
        video_length = video_latent.shape[2]
        depth_map = pipeline.prepare_depth_map(
            pixel_values,
            None,
            1,
            False,
            video_latent.dtype,
            video_latent.device,
        )
        depth_map = depth_map.reshape(
            depth_map.shape[0] // video_length,
            video_length,
            depth_map.shape[1],
            depth_map.shape[2],
            depth_map.shape[3],
        ).permute(0, 2, 1, 3, 4)
    else:
        depth_map = None

    ddim_latents = ddim_loop_long(
        pipeline,
        ddim_scheduler,
        video_latent,
        num_inv_steps,
        prompt,
        window_size,
        stride,
        control,
        depth_map,
    )

    return ddim_latents
