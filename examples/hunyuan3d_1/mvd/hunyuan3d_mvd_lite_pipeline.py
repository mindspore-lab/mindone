# Open Source Model Licensed under the Apache License Version 2.0
# and Other Licenses of the Third-Party Components therein:
# The below Model in this distribution may have been modified by THL A29 Limited
# ("Tencent Modifications"). All Tencent Modifications are Copyright (C) 2024 THL A29 Limited.

# Copyright (C) 2024 THL A29 Limited, a Tencent company.  All rights reserved.
# The below software and/or models in this distribution may have been
# modified by THL A29 Limited ("Tencent Modifications").
# All Tencent Modifications are Copyright (C) THL A29 Limited.

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import inspect
from typing import List, Optional

from transformers import CLIPImageProcessor, CLIPTokenizer

import mindspore as ms
from mindspore import mint, nn, ops

from mindone.diffusers.image_processor import VaeImageProcessor
from mindone.diffusers.loaders import FromSingleFileMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from mindone.diffusers.models import AutoencoderKL, UNet2DConditionModel
from mindone.diffusers.models.attention_processor import AttnProcessor
from mindone.diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from mindone.diffusers.schedulers import KarrasDiffusionSchedulers
from mindone.diffusers.utils.mindspore_utils import randn_tensor
from mindone.transformers import CLIPTextModel, CLIPVisionModelWithProjection
from mindone.utils.version_control import check_valid_flash_attention

from .utils import recenter_img, to_rgb_image

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import mindspore as ms
        >>> from here import Hunyuan3d_MVD_Lite_Pipeline

        >>> pipe = Hunyuan3d_MVD_Lite_Pipeline.from_pretrained(
        ...     "weights/mvd_lite", mindspore_dtype=ms.float16
        ... )

        >>> img = Image.open("demo.png")
        >>> res_img = pipe(img).images[0]
"""


def unscale_latents(latents):
    return latents / 0.75 + 0.22


def unscale_image(image):
    return image / 0.50 * 0.80


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


class ReferenceOnlyAttnProc(nn.Cell):
    # reference attention
    def __init__(self, chained_proc, enabled=False, name=None):
        super().__init__()
        self.enabled = enabled
        self.chained_proc = chained_proc
        self.name = name

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, mode="w", ref_dict=None):
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        if self.enabled:
            if mode == "w":
                ref_dict[self.name] = encoder_hidden_states
            elif mode == "r":
                encoder_hidden_states = mint.cat([encoder_hidden_states, ref_dict.pop(self.name)], dim=1)
        res = self.chained_proc(attn, hidden_states, encoder_hidden_states, attention_mask)
        return res


class RefOnlyNoisedUNet(nn.Cell):
    def __init__(self, unet, train_sched, val_sched):
        super().__init__()
        self.unet = unet
        self.train_sched = train_sched
        self.val_sched = val_sched

        unet_lora_attn_procs = dict()
        default_attn_proc = AttnProcessor()
        for name, _ in unet.attn_processors.items():
            unet_lora_attn_procs[name] = ReferenceOnlyAttnProc(
                default_attn_proc,
                enabled=name.endswith("attn1.processor"),
                name=name,
            )
        unet.set_attn_processor(unet_lora_attn_procs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.unet, name)

    # @ms.jit FAILED
    def construct(self, sample, timestep, encoder_hidden_states, cross_attention_kwargs, return_dict: bool):
        cond_lat = cross_attention_kwargs["cond_lat"]
        noise = ops.randn_like(cond_lat)
        if self.training and (self.train_sched is not None):
            noisy_cond_lat = self.train_sched.add_noise(cond_lat, noise, timestep)
            noisy_cond_lat = self.train_sched.scale_model_input(noisy_cond_lat, timestep)
        else:
            noisy_cond_lat = self.val_sched.add_noise(cond_lat, noise, timestep.reshape(-1))
            noisy_cond_lat = self.val_sched.scale_model_input(noisy_cond_lat, timestep.reshape(-1))

        ref_dict = {}
        self.unet(
            noisy_cond_lat,
            timestep,
            encoder_hidden_states,
            cross_attention_kwargs=dict(mode="w", ref_dict=ref_dict),
            return_dict=return_dict,
        )
        return self.unet(
            sample,
            timestep,
            encoder_hidden_states,
            cross_attention_kwargs=dict(mode="r", ref_dict=ref_dict),
            return_dict=return_dict,
        )


class Hunyuan3d_MVD_Lite_Pipeline(DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin, FromSingleFileMixin):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        vision_encoder: CLIPVisionModelWithProjection,
        feature_extractor_clip: CLIPImageProcessor,
        feature_extractor_vae: CLIPImageProcessor,
        ramping_coefficients: Optional[list] = None,
        safety_checker=None,
    ):
        DiffusionPipeline.__init__(self)
        self.register_modules(
            vae=vae,
            unet=unet,
            tokenizer=tokenizer,
            scheduler=scheduler,
            text_encoder=text_encoder,
            vision_encoder=vision_encoder,
            feature_extractor_vae=feature_extractor_vae,
            feature_extractor_clip=feature_extractor_clip,
        )
        # rewrite the stable diffusion pipeline
        # vae: vae
        # unet: unet
        # tokenizer: tokenizer
        # scheduler: scheduler
        # text_encoder: text_encoder
        # vision_encoder: vision_encoder
        # feature_extractor_vae: feature_extractor_vae
        # feature_extractor_clip: feature_extractor_clip
        self.register_to_config(ramping_coefficients=ramping_coefficients)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        if check_valid_flash_attention():
            self.set_use_memory_efficient_attention_xformers(True)

    def prepare_extra_step_kwargs(self, generator, eta):
        extra_step_kwargs = {}
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        latents = randn_tensor(shape, generator=generator, dtype=dtype)
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def _encode_prompt(
        self,
        prompt,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[ms.Tensor] = None,
        negative_prompt_embeds: Optional[ms.Tensor] = None,
        lora_scale: Optional[float] = None,
    ):
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="np",
            )
            text_input_ids = ms.Tensor(text_inputs.input_ids)

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = ms.Tensor(text_inputs.attention_mask)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(text_input_ids, attention_mask=attention_mask)[0]

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype)
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.tile((1, num_images_per_prompt, 1))
        prompt_embeds = prompt_embeds.view((bs_embed * num_images_per_prompt, seq_len, -1))

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError()
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError()
            else:
                uncond_tokens = negative_prompt
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens, padding="max_length", max_length=max_length, truncation=True, return_tensors="np"
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = ms.Tensor(uncond_input.attention_mask)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(ms.Tensor(uncond_input.input_ids), attention_mask=attention_mask)
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype)
            negative_prompt_embeds = negative_prompt_embeds.tile((1, num_images_per_prompt, 1))
            negative_prompt_embeds = negative_prompt_embeds.view((batch_size * num_images_per_prompt, seq_len, -1))
            prompt_embeds = mint.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def encode_condition_image(self, image: ms.Tensor):
        image_latents = self.vae.encode(image)[0]
        image_latents = self.vae.diag_gauss_dist.sample(image_latents)
        return image_latents

    def __call__(
        self,
        image=None,
        width=640,
        height=960,
        num_inference_steps=75,
        return_dict=True,
        generator=None,
        guidance_scale=2.0,  # no use
    ):
        batch_size = 1
        num_images_per_prompt = 1
        # output_type = "pil"
        do_classifier_free_guidance = True
        guidance_rescale = 0.0
        if isinstance(self.unet, UNet2DConditionModel):
            self.unet = RefOnlyNoisedUNet(self.unet, None, self.scheduler).set_train(False)
            self.unet.to_float(self.dtype)

        cond_image = recenter_img(image)
        cond_image = to_rgb_image(image)
        image = cond_image
        image_1 = self.feature_extractor_vae(images=image, return_tensors="np").pixel_values
        image_2 = self.feature_extractor_clip(images=image, return_tensors="np").pixel_values
        image_1 = ms.Tensor(image_1).to(dtype=self.vae.dtype)
        image_2 = ms.Tensor(image_2).to(dtype=self.vae.dtype)

        cond_lat = self.encode_condition_image(image_1)
        negative_lat = self.encode_condition_image(mint.zeros_like(image_1))
        cond_lat = mint.cat([negative_lat, cond_lat])
        cross_attention_kwargs = dict(cond_lat=cond_lat)

        global_embeds = self.vision_encoder(image_2, output_hidden_states=False)[0].unsqueeze(-2)  # image_embeds
        encoder_hidden_states = self._encode_prompt("", num_images_per_prompt, False)

        ramp = ms.Tensor(self.config.ramping_coefficients, dtype=global_embeds.dtype).unsqueeze(-1)
        prompt_embeds = mint.cat([encoder_hidden_states, encoder_hidden_states + global_embeds * ramp])

        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            generator,
            None,
        )
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, 0.0)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        # set adaptive cfg
        # the image order is:
        #    [0, 60,
        #     120, 180,
        #     240, 300]
        # the cfg is set as 3, 2.5, 2, 1.5

        tmp_guidance_scale = mint.ones_like(latents)
        tmp_guidance_scale[:, :, :40, :40] = 3
        tmp_guidance_scale[:, :, :40, 40:] = 2.5
        tmp_guidance_scale[:, :, 40:80, :40] = 2
        tmp_guidance_scale[:, :, 40:80, 40:] = 1.5
        tmp_guidance_scale[:, :, 80:120, :40] = 2
        tmp_guidance_scale[:, :, 80:120, 40:] = 2.5

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = mint.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                noise_pred = ops.stop_gradient(
                    self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        return_dict=False,
                    )
                )[0]

                adaptive_guidance_scale = (2 + 16 * (t / 1000) ** 5) / 3
                noise_pred_text = None
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + tmp_guidance_scale * adaptive_guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        latents = unscale_latents(latents).to(self.vae.dtype)
        image = unscale_image(self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0])
        image = self.image_processor.postprocess(image, output_type="pil")[0]
        image = [image, cond_image]
        return ImagePipelineOutput(images=image) if return_dict else (image,)
