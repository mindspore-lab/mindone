import inspect
import json
import sys
import random
import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from einops import repeat
from tqdm import tqdm

from audioldm.audio.stft import TacotronSTFT
from audioldm.variational_autoencoder import AutoencoderKL
from audioldm.latent_diffusion.dpm_solver import DPMSolverSampler

# from transformers import CLIPTokenizer, AutoTokenizer
# from transformers import CLIPTextModel, T5EncoderModel, AutoModel

# import diffusers
# from diffusers.utils import randn_tensor
# from diffusers import DDPMScheduler, UNet2DConditionModel
# from diffusers import AutoencoderKL as DiffuserAutoencoderKL
from flan_t5_large.t5 import get_t5_tokenizer, get_t5_encoder
from audioldm.latent_diffusion.ddpm import DDPM, DiffusionWrapper


class AudioDiffusion(nn.Cell):
    def __init__(
        self,
        text_encoder_name,
        scheduler_name,
        unet_model_name=None,
        unet_model_config_path=None,
        snr_gamma=None,
        freeze_text_encoder=True,
        uncondition=False,
    ):
        super().__init__()

        assert unet_model_name is not None or unet_model_config_path is not None, "Either UNet pretrain model name or a config file path is required"

        self.text_encoder_name = text_encoder_name
        self.scheduler_name = scheduler_name
        self.unet_model_name = unet_model_name
        self.unet_model_config_path = unet_model_config_path
        self.snr_gamma = snr_gamma
        self.freeze_text_encoder = freeze_text_encoder
        self.uncondition = uncondition

        # https://huggingface.co/docs/diffusers/v0.14.0/en/api/schedulers/overview
        # self.noise_scheduler = DDPMScheduler.from_pretrained(self.scheduler_name, subfolder="scheduler")
        # self.inference_scheduler = DDPMScheduler.from_pretrained(self.scheduler_name, subfolder="scheduler")

        if unet_model_config_path:
            # unet_config = UNet2DConditionModel.load_config(unet_model_config_path)
            # self.unet = UNet2DConditionModel.from_config(unet_config, subfolder="unet")
            unet_config = json.load(open(unet_model_config_path))
            self.unet = DiffusionWrapper(unet_config, conditioning_key="crossattn")
            self.set_from = "random"
            print("UNet initialized randomly.")
        else:
            # self.unet = UNet2DConditionModel.from_pretrained(unet_model_name, subfolder="unet")
            self.unet = DiffusionWrapper(unet_config, conditioning_key="crossattn")
            self.set_from = "pre-trained"
            self.group_in = nn.Sequential(nn.Linear(8, 512), nn.Linear(512, 4))
            self.group_out = nn.Sequential(nn.Linear(4, 512), nn.Linear(512, 8))
            print("UNet initialized from stable diffusion checkpoint.")

        if "t5" in self.text_encoder_name:
            self.tokenizer = get_t5_tokenizer()
            self.text_encoder = get_t5_encoder()
        elif "stable-diffusion" in self.text_encoder_name:
            raise NotImplementedError("sd text encoder")
            # self.tokenizer = CLIPTokenizer.from_pretrained(self.text_encoder_name, subfolder="tokenizer")
            # self.text_encoder = CLIPTextModel.from_pretrained(self.text_encoder_name, subfolder="text_encoder")
        else:
            raise NotImplementedError(self.text_encoder_name)
            # self.tokenizer = AutoTokenizer.from_pretrained(self.text_encoder_name)
            # self.text_encoder = AutoModel.from_pretrained(self.text_encoder_name)

    def compute_snr(self, timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = self.noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    def encode_text(self, prompt):
        batch = self.tokenizer(
            prompt, max_length=self.tokenizer.model_max_length, padding=True, truncation=True, return_tensors="pt"
        )
        input_ids, attention_mask = batch.input_ids, batch.attention_mask

        # with torch.no_grad():
        encoder_hidden_states = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )[0]

        boolean_encoder_mask = (attention_mask == 1)
        return encoder_hidden_states, boolean_encoder_mask

    def inference(
        self,
        prompt,
        inference_scheduler,
        num_steps=20,
        guidance_scale=3,
        num_samples_per_prompt=1,
        disable_progress=True,
        ddim_eta=0.0,
    ):
        classifier_free_guidance = guidance_scale > 1.0
        batch_size = len(prompt) * num_samples_per_prompt

        if classifier_free_guidance:
            uc, ucmask, c, cmask = self.encode_text_classifier_free(prompt, num_samples_per_prompt)
        else:
            c, cmask = self.encode_text(prompt)
            c = c.repeat_interleave(num_samples_per_prompt, 0)
            cmask = cmask.repeat_interleave(num_samples_per_prompt, 0)

        inference_scheduler.set_timesteps(num_steps)
        timesteps = inference_scheduler.timesteps

        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(batch_size, inference_scheduler, num_channels_latents, prompt_embeds.dtype)

        sampler = DPMSolverSampler(self.unet, "dpmsolver", prediction_type="noise")

        shape = latents.shape[1:]
        latents, _ = sampler.sample(
            S=num_steps,
            conditioning=c,
            batch_size=1,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=guidance_scale,
            unconditional_conditioning=uc,
            eta=ddim_eta,
            x_T=latents,
        )

        '''
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = ops.cat([latents] * 2) if classifier_free_guidance else latents
            latent_model_input = inference_scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=c,
                encoder_attention_mask=cmask
            ).sample

            # perform guidance
            if classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = inference_scheduler.step(noise_pred, t, latents).prev_sample

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % inference_scheduler.order == 0):
                progress_bar.update(1)
        '''

        if self.set_from == "pre-trained":
            latents = self.group_out(latents.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
        return latents

    def prepare_latents(self, batch_size, inference_scheduler, num_channels_latents, dtype, device):
        shape = (batch_size, num_channels_latents, 256, 16)
        latents = ops.StandardNormal()(shape)
        # latents = randn_tensor(shape, generator=None, device=device, dtype=dtype)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * inference_scheduler.init_noise_sigma
        return latents

    def encode_text_classifier_free(self, prompt, num_samples_per_prompt):
        batch = self.tokenizer(
            prompt, max_length=self.tokenizer.model_max_length, padding=True, truncation=True, return_tensors="pt"
        )
        input_ids, attention_mask = batch.input_ids, batch.attention_mask

        # with torch.no_grad():
        prompt_embeds = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )[0]

        prompt_embeds = prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
        attention_mask = attention_mask.repeat_interleave(num_samples_per_prompt, 0)

        # get unconditional embeddings for classifier free guidance
        uncond_tokens = [""] * len(prompt)

        max_length = prompt_embeds.shape[1]
        uncond_batch = self.tokenizer(
            uncond_tokens, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt",
        )
        uncond_input_ids = uncond_batch.input_ids
        uncond_attention_mask = uncond_batch.attention_mask

        # with torch.no_grad():
        negative_prompt_embeds = self.text_encoder(
            input_ids=uncond_input_ids, attention_mask=uncond_attention_mask
        )[0]

        negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
        uncond_attention_mask = uncond_attention_mask.repeat_interleave(num_samples_per_prompt, 0)

        # For classifier free guidance, we need to do two forward passes.
        # We concatenate the unconditional and text embeddings into a single batch to avoid doing two forward passes
        # prompt_embeds = ops.concat([negative_prompt_embeds, prompt_embeds])
        # prompt_mask = ops.concat([uncond_attention_mask, attention_mask])
        # boolean_prompt_mask = (prompt_mask == 1)

        return (
            negative_prompt_embeds,
            uncond_attention_mask == 1,
            prompt_embeds,
            attention_mask == 1,
        )
