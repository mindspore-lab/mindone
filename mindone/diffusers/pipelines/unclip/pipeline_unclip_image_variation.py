# Copyright 2024 Kakao Brain and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import List, Optional, Union

import numpy as np
import PIL.Image
from transformers import CLIPImageProcessor, CLIPTokenizer

import mindspore as ms
from mindspore import ops

from ....transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection
from ...models import UNet2DConditionModel, UNet2DModel
from ...schedulers import UnCLIPScheduler
from ...utils import logging
from ...utils.mindspore_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from .text_proj import UnCLIPTextProjModel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class UnCLIPImageVariationPipeline(DiffusionPipeline):
    """
    Pipeline to generate image variations from an input image using UnCLIP.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        text_encoder ([`~transformers.CLIPTextModelWithProjection`]):
            Frozen text-encoder.
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            Model that extracts features from generated images to be used as inputs for the `image_encoder`.
        image_encoder ([`~transformers.CLIPVisionModelWithProjection`]):
            Frozen CLIP image-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        text_proj ([`UnCLIPTextProjModel`]):
            Utility class to prepare and combine the embeddings before they are passed to the decoder.
        decoder ([`UNet2DConditionModel`]):
            The decoder to invert the image embedding into an image.
        super_res_first ([`UNet2DModel`]):
            Super resolution UNet. Used in all but the last step of the super resolution diffusion process.
        super_res_last ([`UNet2DModel`]):
            Super resolution UNet. Used in the last step of the super resolution diffusion process.
        decoder_scheduler ([`UnCLIPScheduler`]):
            Scheduler used in the decoder denoising process (a modified [`DDPMScheduler`]).
        super_res_scheduler ([`UnCLIPScheduler`]):
            Scheduler used in the super resolution denoising process (a modified [`DDPMScheduler`]).
    """

    decoder: UNet2DConditionModel
    text_proj: UnCLIPTextProjModel
    text_encoder: CLIPTextModelWithProjection
    tokenizer: CLIPTokenizer
    feature_extractor: CLIPImageProcessor
    image_encoder: CLIPVisionModelWithProjection
    super_res_first: UNet2DModel
    super_res_last: UNet2DModel

    decoder_scheduler: UnCLIPScheduler
    super_res_scheduler: UnCLIPScheduler
    model_cpu_offload_seq = "text_encoder->image_encoder->text_proj->decoder->super_res_first->super_res_last"

    def __init__(
        self,
        decoder: UNet2DConditionModel,
        text_encoder: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        text_proj: UnCLIPTextProjModel,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection,
        super_res_first: UNet2DModel,
        super_res_last: UNet2DModel,
        decoder_scheduler: UnCLIPScheduler,
        super_res_scheduler: UnCLIPScheduler,
    ):
        super().__init__()

        self.register_modules(
            decoder=decoder,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_proj=text_proj,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            super_res_first=super_res_first,
            super_res_last=super_res_last,
            decoder_scheduler=decoder_scheduler,
            super_res_scheduler=super_res_scheduler,
        )

    # Copied from diffusers.pipelines.unclip.pipeline_unclip.UnCLIPPipeline.prepare_latents
    def prepare_latents(self, shape, dtype, generator, latents, scheduler):
        if latents is None:
            latents = randn_tensor(shape, generator=generator, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")

        latents = latents * scheduler.init_noise_sigma
        return latents

    def _encode_prompt(self, prompt, num_images_per_prompt, do_classifier_free_guidance):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        # get prompt text embeddings
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="np",
        )
        text_input_ids = ms.Tensor.from_numpy(text_inputs.input_ids)
        text_mask = ms.Tensor.from_numpy(text_inputs.attention_mask)  # MindSpore mask does not require bool()
        text_encoder_output = self.text_encoder(text_input_ids)

        prompt_embeds = text_encoder_output[0]
        text_encoder_hidden_states = text_encoder_output[1]

        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        text_encoder_hidden_states = text_encoder_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
        text_mask = text_mask.repeat_interleave(num_images_per_prompt, dim=0)

        if do_classifier_free_guidance:
            uncond_tokens = [""] * batch_size

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="np",
            )
            uncond_text_mask = ms.Tensor.from_numpy(
                uncond_input.attention_mask
            )  # MindSpore mask does not require bool()
            negative_prompt_embeds_text_encoder_output = self.text_encoder(ms.Tensor.from_numpy(uncond_input.input_ids))

            negative_prompt_embeds = negative_prompt_embeds_text_encoder_output[0]
            uncond_text_encoder_hidden_states = negative_prompt_embeds_text_encoder_output[1]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method

            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.tile((1, num_images_per_prompt))
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len)

            seq_len = uncond_text_encoder_hidden_states.shape[1]
            uncond_text_encoder_hidden_states = uncond_text_encoder_hidden_states.tile((1, num_images_per_prompt, 1))
            uncond_text_encoder_hidden_states = uncond_text_encoder_hidden_states.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )
            uncond_text_mask = uncond_text_mask.repeat_interleave(num_images_per_prompt, dim=0)

            # done duplicates

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = ops.cat([negative_prompt_embeds, prompt_embeds])
            text_encoder_hidden_states = ops.cat([uncond_text_encoder_hidden_states, text_encoder_hidden_states])

            text_mask = ops.cat([uncond_text_mask, text_mask])

        return prompt_embeds, text_encoder_hidden_states, text_mask

    def _encode_image(self, image, num_images_per_prompt, image_embeddings: Optional[ms.Tensor] = None):
        dtype = next(self.image_encoder.get_parameters()).dtype

        if image_embeddings is None:
            if not isinstance(image, ms.Tensor):
                image = self.feature_extractor(images=image, return_tensors="np").pixel_values
                image = ms.Tensor(image)

            image = image.to(dtype=dtype)
            image_embeddings = self.image_encoder(image)[0]

        image_embeddings = image_embeddings.repeat_interleave(num_images_per_prompt, dim=0)

        return image_embeddings

    def __call__(
        self,
        image: Optional[Union[PIL.Image.Image, List[PIL.Image.Image], ms.Tensor]] = None,
        num_images_per_prompt: int = 1,
        decoder_num_inference_steps: int = 25,
        super_res_num_inference_steps: int = 7,
        generator: Optional[np.random.Generator] = None,
        decoder_latents: Optional[ms.Tensor] = None,
        super_res_latents: Optional[ms.Tensor] = None,
        image_embeddings: Optional[ms.Tensor] = None,
        decoder_guidance_scale: float = 8.0,
        output_type: Optional[str] = "pil",
        return_dict: bool = False,
    ):
        """
        The call function to the pipeline for generation.

        Args:
            image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `ms.Tensor`):
                `Image` or tensor representing an image batch to be used as the starting point. If you provide a
                tensor, it needs to be compatible with the [`CLIPImageProcessor`]
                [configuration](https://huggingface.co/fusing/karlo-image-variations-diffusers/blob/main/feature_extractor/preprocessor_config.json).
                Can be left as `None` only when `image_embeddings` are passed.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            decoder_num_inference_steps (`int`, *optional*, defaults to 25):
                The number of denoising steps for the decoder. More denoising steps usually lead to a higher quality
                image at the expense of slower inference.
            super_res_num_inference_steps (`int`, *optional*, defaults to 7):
                The number of denoising steps for super resolution. More denoising steps usually lead to a higher
                quality image at the expense of slower inference.
            generator (`np.random.Generator`, *optional*):
                A [`np.random.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            decoder_latents (`ms.Tensor` of shape (batch size, channels, height, width), *optional*):
                Pre-generated noisy latents to be used as inputs for the decoder.
            super_res_latents (`ms.Tensor` of shape (batch size, channels, super res height, super res width), *optional*):
                Pre-generated noisy latents to be used as inputs for the decoder.
            decoder_guidance_scale (`float`, *optional*, defaults to 4.0):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            image_embeddings (`ms.Tensor`, *optional*):
                Pre-defined image embeddings that can be derived from the image encoder. Pre-defined image embeddings
                can be passed for tasks like image interpolations. `image` can be left as `None`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images.
        """
        if image is not None:
            if isinstance(image, PIL.Image.Image):
                batch_size = 1
            elif isinstance(image, list):
                batch_size = len(image)
            else:
                batch_size = image.shape[0]
        else:
            batch_size = image_embeddings.shape[0]

        prompt = [""] * batch_size

        batch_size = batch_size * num_images_per_prompt

        do_classifier_free_guidance = decoder_guidance_scale > 1.0

        prompt_embeds, text_encoder_hidden_states, text_mask = self._encode_prompt(
            prompt, num_images_per_prompt, do_classifier_free_guidance
        )

        image_embeddings = self._encode_image(image, num_images_per_prompt, image_embeddings)

        # decoder
        text_encoder_hidden_states, additive_clip_time_embeddings = self.text_proj(
            image_embeddings=image_embeddings,
            prompt_embeds=prompt_embeds,
            text_encoder_hidden_states=text_encoder_hidden_states,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )

        decoder_text_mask = ops.pad(text_mask, (self.text_proj.clip_extra_context_tokens, 0), value=True)

        self.decoder_scheduler.set_timesteps(decoder_num_inference_steps)
        decoder_timesteps_tensor = self.decoder_scheduler.timesteps

        num_channels_latents = self.decoder.config.in_channels
        height = self.decoder.config.sample_size
        width = self.decoder.config.sample_size

        if decoder_latents is None:
            decoder_latents = self.prepare_latents(
                (batch_size, num_channels_latents, height, width),
                text_encoder_hidden_states.dtype,
                generator,
                decoder_latents,
                self.decoder_scheduler,
            )

        for i, t in enumerate(self.progress_bar(decoder_timesteps_tensor)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = ops.cat([decoder_latents] * 2) if do_classifier_free_guidance else decoder_latents

            noise_pred = self.decoder(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=text_encoder_hidden_states,
                class_labels=additive_clip_time_embeddings,
                attention_mask=decoder_text_mask,
            )[0]

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred_uncond, _ = noise_pred_uncond.split(latent_model_input.shape[1], axis=1)
                noise_pred_text, predicted_variance = noise_pred_text.split(latent_model_input.shape[1], axis=1)
                noise_pred = noise_pred_uncond + decoder_guidance_scale * (noise_pred_text - noise_pred_uncond)
                noise_pred = ops.cat([noise_pred, predicted_variance], axis=1)

            if i + 1 == decoder_timesteps_tensor.shape[0]:
                prev_timestep = None
            else:
                prev_timestep = decoder_timesteps_tensor[i + 1]

            # compute the previous noisy sample x_t -> x_t-1
            decoder_latents = self.decoder_scheduler.step(
                noise_pred, t, decoder_latents, prev_timestep=prev_timestep, generator=generator
            )[0]

        decoder_latents = decoder_latents.clamp(-1, 1)

        image_small = decoder_latents

        # done decoder

        # super res

        self.super_res_scheduler.set_timesteps(super_res_num_inference_steps)
        super_res_timesteps_tensor = self.super_res_scheduler.timesteps

        channels = self.super_res_first.config.in_channels // 2
        height = self.super_res_first.config.sample_size
        width = self.super_res_first.config.sample_size

        if super_res_latents is None:
            super_res_latents = self.prepare_latents(
                (batch_size, channels, height, width),
                image_small.dtype,
                generator,
                super_res_latents,
                self.super_res_scheduler,
            )

        interpolate_antialias = {}
        if "antialias" in inspect.signature(ops.interpolate).parameters:
            interpolate_antialias["antialias"] = True

        image_upscaled = ops.interpolate(
            image_small, size=[height, width], mode="bicubic", align_corners=False, **interpolate_antialias
        )

        for i, t in enumerate(self.progress_bar(super_res_timesteps_tensor)):
            # no classifier free guidance

            if i == super_res_timesteps_tensor.shape[0] - 1:
                unet = self.super_res_last
            else:
                unet = self.super_res_first

            latent_model_input = ops.cat([super_res_latents, image_upscaled], axis=1)

            noise_pred = unet(
                sample=latent_model_input,
                timestep=t,
            )[0]

            if i + 1 == super_res_timesteps_tensor.shape[0]:
                prev_timestep = None
            else:
                prev_timestep = super_res_timesteps_tensor[i + 1]

            # compute the previous noisy sample x_t -> x_t-1
            super_res_latents = self.super_res_scheduler.step(
                noise_pred, t, super_res_latents, prev_timestep=prev_timestep, generator=generator
            )[0]

        image = super_res_latents

        # post processing

        image = image * 0.5 + 0.5
        image = image.clamp(0, 1)
        image = image.permute(0, 2, 3, 1).float().numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
