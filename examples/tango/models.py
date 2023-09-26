import json
from functools import partial

import numpy as np
from audioldm.latent_diffusion.ddpm import DDPMScheduler, DiffusionWrapper
from audioldm.latent_diffusion.util import make_beta_schedule
from audioldm.utils import exists
from flan_t5_large.t5 import get_t5_encoder, get_t5_tokenizer

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


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
        timesteps=1000,
        beta_schedule="linear",
        loss_type="l2",
        ckpt_path=None,
        ignore_keys=[],
        load_only_unet=False,
        monitor="val/loss",
        use_ema=True,
        first_stage_key="image",
        latent_t_size=256,
        latent_f_size=16,
        channels=3,
        log_every_t=100,
        clip_denoised=True,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        given_betas=None,
        original_elbo_weight=0.0,
        v_posterior=0.0,
        l_simple_weight=1.0,
        conditioning_key=None,
        parameterization="eps",
        scheduler_config=None,
        use_positional_encodings=False,
        learn_logvar=False,
        logvar_init=0.0,
        use_fp16=False,
    ):
        super().__init__()

        assert (
            unet_model_name is not None or unet_model_config_path is not None
        ), "Either UNet pretrain model name or a config file path is required"

        self.text_encoder_name = text_encoder_name
        self.scheduler_name = scheduler_name
        self.unet_model_name = unet_model_name
        self.unet_model_config_path = unet_model_config_path
        self.snr_gamma = snr_gamma
        self.freeze_text_encoder = freeze_text_encoder
        self.uncondition = uncondition
        self.v_posterior = v_posterior
        self.dtype = ms.float16 if use_fp16 else ms.float32

        if unet_model_config_path:
            self.unet_config = json.load(open(unet_model_config_path))
            self.unet = DiffusionWrapper(self.unet_config, conditioning_key="crossattn")
            self.set_from = "random"
        else:
            raise NotImplementedError("specify unet_model_config_path, path to a json file")

        if "t5" in self.text_encoder_name:
            self.tokenizer = get_t5_tokenizer()
            self.text_encoder = get_t5_encoder()
        elif "stable-diffusion" in self.text_encoder_name:
            raise NotImplementedError("sd text encoder")
        else:
            raise NotImplementedError(self.text_encoder_name)
        self.register_schedule(
            given_betas=given_betas,
            beta_schedule=beta_schedule,
            timesteps=timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )

    def register_schedule(
        self,
        given_betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(
                beta_schedule,
                timesteps,
                linear_start=linear_start,
                linear_end=linear_end,
                cosine_s=cosine_s,
            )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, "alphas have to be defined for each timestep"

        to_mindspore = partial(ms.Tensor, dtype=self.dtype)
        self.betas = to_mindspore(betas)
        self.alphas_cumprod = to_mindspore(alphas_cumprod)
        self.alphas_cumprod_prev = to_mindspore(alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = to_mindspore(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = to_mindspore(np.sqrt(1.0 - alphas_cumprod))
        self.log_one_minus_alphas_cumprod = to_mindspore(np.log(1.0 - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = to_mindspore(np.sqrt(1.0 / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = to_mindspore(np.sqrt(1.0 / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1.0 - alphas_cumprod_prev) / (
            1.0 - alphas_cumprod
        ) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = to_mindspore(posterior_variance)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = to_mindspore(np.log(np.maximum(posterior_variance, 1e-20)))
        self.posterior_mean_coef1 = to_mindspore(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.posterior_mean_coef2 = to_mindspore((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod))

        self.noise_scheduler = DDPMScheduler()
        self.noise_scheduler.alphas_cumprod = alphas_cumprod
        self.noise_scheduler.num_train_timesteps = timesteps

    def encode_text(self, prompt, num_samples_per_prompt, padding, truncation):
        batch = self.tokenizer(prompt, self.tokenizer.model_max_length, padding, truncation)
        input_ids, attention_mask = batch

        encoder_hidden_states = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_samples_per_prompt, 0)
        attention_mask = attention_mask.repeat_interleave(num_samples_per_prompt, 0)

        return encoder_hidden_states, attention_mask == 1

    def apply_model(self, x_noisy, t, cond, return_ids=False, **kwargs):
        """
        args:
            cond: it can be a dictionary or a Tensor. When `cond` is a dictionary,
                it passes through `DiffusionWrapper` as keyword arguments. When it
                is a Tensor, it is the input argument of "c_concat" or `c_crossattn`
                depends on the predefined `conditioning_key`.
        """
        if isinstance(cond, dict):
            # hybrid case, cond is expected to be a dict
            pass
        else:
            key = "c_concat" if self.unet.conditioning_key == "concat" else "c_crossattn"
            cond = {key: cond}

        x_recon = self.unet(x_noisy, t, **cond, **kwargs)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def inference(
        self,
        prompt,
        num_steps=20,
        guidance_scale=3,
        num_samples_per_prompt=1,
        disable_progress=True,
        ddim_eta=0.0,
        inference_scheduler=None,
        sampler=None,
        padding=True,
        truncation=True,
    ):
        classifier_free_guidance = guidance_scale > 1.0
        batch_size = len(prompt) * num_samples_per_prompt

        if classifier_free_guidance:
            uc, ucmask, c, cmask = self.encode_text_classifier_free(prompt, num_samples_per_prompt, padding, truncation)
        else:
            c, cmask = self.encode_text(prompt, num_samples_per_prompt, padding, truncation)

        num_channels_latents = self.unet_config["params"]["in_channels"]
        latents = self.prepare_latents(batch_size, inference_scheduler, num_channels_latents)

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

        # if self.set_from == "pre-trained":
        #     latents = self.group_out(latents.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
        return latents

    def prepare_latents(self, batch_size, inference_scheduler, num_channels_latents):
        shape = (batch_size, num_channels_latents, 256, 16)
        latents = ops.StandardNormal()(shape)
        # scale the initial noise by the standard deviation required by the scheduler
        # latents = latents * inference_scheduler.init_noise_sigma
        return latents

    def encode_text_classifier_free(self, prompt, num_samples_per_prompt, padding, truncation):
        batch = self.tokenizer(prompt, self.tokenizer.model_max_length, padding, truncation)
        input_ids, attention_mask = batch

        prompt_embeds = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)[0]

        prompt_embeds = prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
        attention_mask = attention_mask.repeat_interleave(num_samples_per_prompt, 0)

        # get unconditional embeddings for classifier free guidance
        uncond_tokens = [""] * len(prompt)

        max_length = prompt_embeds.shape[1]
        uncond_batch = self.tokenizer(uncond_tokens, max_length=max_length, padding="max_length", truncation=True)
        uncond_input_ids, uncond_attention_mask = uncond_batch

        negative_prompt_embeds = self.text_encoder(input_ids=uncond_input_ids, attention_mask=uncond_attention_mask)[0]

        negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
        uncond_attention_mask = uncond_attention_mask.repeat_interleave(num_samples_per_prompt, 0)

        return (
            negative_prompt_embeds,
            uncond_attention_mask == 1,
            prompt_embeds,
            attention_mask == 1,
        )
