# Adapted from https://github.com/VectorSpaceLab/OmniGen2/blob/main/omnigen2/pipelines/omnigen2/pipeline_omnigen2_chat.py
"""
OmniGen2 Diffusion Pipeline

Copyright 2025 BAAI, The OmniGen2 Team and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import inspect
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
from PIL.Image import Image

import mindspore as ms
import mindspore.mint.nn.functional as F
from mindspore import mint

from mindone.diffusers.models.autoencoders import AutoencoderKL
from mindone.diffusers.pipelines.pipeline_utils import DiffusionPipeline
from mindone.diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from mindone.diffusers.utils import BaseOutput, logging
from mindone.diffusers.utils.mindspore_utils import randn_tensor
from mindone.transformers import Qwen2_5_VLForConditionalGeneration

from ...models.transformers import OmniGen2Transformer2DModel
from ...models.transformers.repo import OmniGen2RotaryPosEmbed
from ..image_processor import OmniGen2ImageProcessor

logger = logging.get_logger(__name__)


@dataclass
class OmniGen2PipelineOutput(BaseOutput):
    """
    Output class for OmniGen2 pipeline.

    Args:
        images (Union[list[Image], np.ndarray]):
            list of denoised PIL images of length `batch_size` or numpy array of shape
            `(batch_size, height, width, num_channels)`. Contains the generated images.
    """

    text: str
    images: Union[list[Image], np.ndarray]


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler, num_inference_steps: Optional[int] = None, timesteps: Optional[list[int]] = None, **kwargs
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        timesteps (`list[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`list[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `tuple[ms.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class OmniGen2ChatPipeline(DiffusionPipeline):
    """
    Pipeline for text-to-image generation using OmniGen2.

    This pipeline implements a text-to-image generation model that uses:
    - Qwen2.5-VL for text encoding
    - A custom transformer architecture for image generation
    - VAE for image encoding/decoding
    - FlowMatchEulerDiscreteScheduler for noise scheduling

    Args:
        transformer (OmniGen2Transformer2DModel): The transformer model for image generation.
        vae (AutoencoderKL): The VAE model for image encoding/decoding.
        scheduler (FlowMatchEulerDiscreteScheduler): The scheduler for noise scheduling.
        text_encoder (Qwen2_5_VLModel): The text encoder model.
        tokenizer (Union[Qwen2Tokenizer, Qwen2TokenizerFast]): The tokenizer for text processing.
    """

    model_cpu_offload_seq = "mllm->transformer->vae"

    def __init__(
        self,
        transformer: OmniGen2Transformer2DModel,
        vae: AutoencoderKL,
        scheduler: FlowMatchEulerDiscreteScheduler,
        mllm: Qwen2_5_VLForConditionalGeneration,
        processor,
    ) -> None:
        """
        Initialize the OmniGen2 pipeline.

        Args:
            transformer: The transformer model for image generation.
            vae: The VAE model for image encoding/decoding.
            scheduler: The scheduler for noise scheduling.
            text_encoder: The text encoder model.
            tokenizer: The tokenizer for text processing.
        """
        super().__init__()

        self.register_modules(transformer=transformer, vae=vae, scheduler=scheduler, mllm=mllm, processor=processor)
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        self.image_processor = OmniGen2ImageProcessor(vae_scale_factor=self.vae_scale_factor * 2, do_resize=True)
        self.default_sample_size = 128

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: ms.Type,
        generator: Optional[np.random.Generator],
        latents: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        """
        Prepare the initial latents for the diffusion process.

        Args:
            batch_size: The number of images to generate.
            num_channels_latents: The number of channels in the latent space.
            height: The height of the generated image.
            width: The width of the generated image.
            dtype: The data type of the latents.
            generator: The random number generator to use.
            latents: Optional pre-computed latents to use instead of random initialization.

        Returns:
            ms.Tensor: The prepared latents tensor.
        """
        height = int(height) // self.vae_scale_factor
        width = int(width) // self.vae_scale_factor

        shape = (batch_size, num_channels_latents, height, width)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, dtype=dtype)
        return latents

    def encode_vae(self, img: ms.Tensor) -> ms.Tensor:
        """
        Encode an image into the VAE latent space.

        Args:
            img: The input image tensor to encode.

        Returns:
            ms.Tensor: The encoded latent representation.
        """
        z0 = self.vae.diag_gauss_dist.sample(self.vae.encode(img.to(dtype=self.vae.dtype))[0])
        if self.vae.config.shift_factor is not None:
            z0 = z0 - self.vae.config.shift_factor
        if self.vae.config.scaling_factor is not None:
            z0 = z0 * self.vae.config.scaling_factor
        z0 = z0.to(dtype=self.vae.dtype)
        return z0

    def prepare_image(
        self,
        images: Union[list[Image], Image],
        batch_size: int,
        num_images_per_prompt: int,
        max_pixels: int,
        max_side_length: int,
        dtype: ms.Type,
    ) -> list[Optional[ms.Tensor]]:
        """
        Prepare input images for processing by encoding them into the VAE latent space.

        Args:
            images: Single image or list of images to process.
            batch_size: The number of images to generate per prompt.
            num_images_per_prompt: The number of images to generate for each prompt.
            dtype: The data type of the encoded latents.

        Returns:
            list[Optional[ms.Tensor]]: list of encoded latent representations for each image.
        """
        if batch_size == 1:
            images = [images]
        latents = []
        for i, img in enumerate(images):
            if img is not None and len(img) > 0:
                ref_latents = []
                for j, img_j in enumerate(img):
                    img_j = self.image_processor.preprocess(
                        img_j, max_pixels=max_pixels, max_side_length=max_side_length
                    )
                    ref_latents.append(self.encode_vae(img_j).squeeze(0))
            else:
                ref_latents = None
            for _ in range(num_images_per_prompt):
                latents.append(ref_latents)

        return latents

    def _apply_chat_template(self, prompt: str, images: list = None):
        if images is not None:
            prompt = (
                "".join([f"<img{i}>: <|vision_start|><|image_pad|><|vision_end|>" for i in range(1, len(images) + 1)])
                + prompt
            )
        prompt = (
            f"<|im_start|>system\nYou are a helpful assistant that generates high-quality images based on user instructions.<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        )
        return prompt

    def _get_qwen2_prompt_embeds(
        self, prompt: Union[str, list[str]], input_images=None, use_only_text_hidden_states: bool = True
    ) -> tuple[ms.Tensor, ms.Tensor]:
        """
        Get prompt embeddings from the Qwen2 text encoder.

        Args:
            prompt: The prompt or list of prompts to encode.

        Returns:
            tuple[ms.Tensor, ms.Tensor]: A tuple containing:
                - The prompt embeddings tensor
                - The attention mask tensor

        Raises:
            Warning: If the input text is truncated due to sequence length limitations.
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt

        inputs = self.processor(text=prompt, images=input_images, videos=None, padding=True, return_tensors="np")
        text_input_ids, text_mask = ms.tensor(inputs.input_ids), ms.tensor(inputs.attention_mask)

        prompt_embeds = self.mllm(text_input_ids, attention_mask=text_mask, output_hidden_states=True).hidden_states[-1]

        if use_only_text_hidden_states:
            mask = text_input_ids != self.mllm.config.image_token_id
            mask = mask & text_mask
            mask = mask.bool()

            text_l = mask.sum(dim=-1)
            max_l = text_l.max()
            text_batch_size = prompt_embeds.shape[0]
            new_prompt_embeds = mint.zeros((text_batch_size, max_l, prompt_embeds.shape[-1]), dtype=prompt_embeds.dtype)
            new_text_mask = mint.zeros((text_batch_size, max_l), dtype=text_mask.dtype)
            for i in range(text_batch_size):
                new_prompt_embeds[i, : text_l[i]] = prompt_embeds[i, mask[i]]
                new_text_mask[i, : text_l[i]] = 1

            prompt_embeds = new_prompt_embeds
            text_mask = new_text_mask

        prompt_embeds = prompt_embeds.to(dtype=self.mllm.dtype)
        return prompt_embeds, text_mask

    def encode_prompt(
        self,
        prompt: Union[str, list[str]],
        input_images: Optional[Union[str, list[str]]] = None,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, list[str]]] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[ms.Tensor] = None,
        negative_prompt_embeds: Optional[ms.Tensor] = None,
        prompt_attention_mask: Optional[ms.Tensor] = None,
        negative_prompt_attention_mask: Optional[ms.Tensor] = None,
        max_sequence_length: int = 256,
        use_text_encoder_penultimate_layer_feats: bool = False,
    ) -> tuple[ms.Tensor, ms.Tensor, ms.Tensor, ms.Tensor]:
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `list[str]`, *optional*):
                The prompt not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds`
                instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`). For
                Lumina-T2I, this should be "".
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                whether to use classifier free guidance or not
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                number of images that should be generated per prompt
            prompt_embeds (`ms.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`ms.Tensor`, *optional*):
                Pre-generated negative text embeddings. For Lumina-T2I, it's should be the embeddings of the "" string.
            max_sequence_length (`int`, defaults to `256`):
                Maximum sequence length to use for the prompt.
        """

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = self._get_qwen2_prompt_embeds(
                prompt=prompt, input_images=input_images
            )

        batch_size, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_images_per_prompt, 1)
        prompt_attention_mask = prompt_attention_mask.view(batch_size * num_images_per_prompt, -1)

        # Get negative embeddings for classifier free guidance
        negative_prompt_embeds, negative_prompt_attention_mask = None, None
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt if negative_prompt is not None else ""

            # Normalize str to list
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt = [self._apply_chat_template(_negative_prompt) for _negative_prompt in negative_prompt]

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            negative_prompt_embeds, negative_prompt_attention_mask = self._get_qwen2_prompt_embeds(
                prompt=negative_prompt
            )

            batch_size, seq_len, _ = negative_prompt_embeds.shape
            # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
            negative_prompt_attention_mask = negative_prompt_attention_mask.repeat(num_images_per_prompt, 1)
            negative_prompt_attention_mask = negative_prompt_attention_mask.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def text_guidance_scale(self):
        return self._text_guidance_scale

    @property
    def image_guidance_scale(self):
        return self._image_guidance_scale

    @property
    def cfg_range(self):
        return self._cfg_range

    def prepare_inputs_for_text_generation(self, prompts, input_images):
        if isinstance(prompts, str):
            prompts = [prompts]

        ori_padding_side = self.processor.tokenizer.padding_side
        self.processor.tokenizer.padding_side = "left"
        inputs = self.processor(text=prompts, images=input_images, videos=None, padding=True, return_tensors="np")
        inputs = {k: ms.tensor(v) for k, v in inputs.items()}
        self.processor.tokenizer.padding_side = ori_padding_side
        return inputs

    def generate_text(self, prompt, input_images):
        inputs = self.prepare_inputs_for_text_generation(prompt, input_images)
        generated_ids = self.mllm.generate(
            **inputs,
            tokenizer=self.processor.tokenizer,
            max_new_tokens=256,
            stop_strings=["<|im_end|>", "<|img|>", "<|endoftext|>"],
        )  # stop_words=[151643, 151645, 151665]
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed,
            # skip_special_tokens=True,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        return output_texts

    def generate_image(
        self,
        prompt: Optional[Union[str, list[str]]] = None,
        negative_prompt: Optional[Union[str, list[str]]] = None,
        prompt_embeds: Optional[ms.Tensor] = None,
        negative_prompt_embeds: Optional[ms.Tensor] = None,
        prompt_attention_mask: Optional[ms.Tensor] = None,
        negative_prompt_attention_mask: Optional[ms.Tensor] = None,
        use_text_encoder_penultimate_layer_feats: bool = False,
        max_sequence_length: Optional[int] = None,
        callback_on_step_end_tensor_inputs: Optional[list[str]] = None,
        input_images: Optional[list[Image]] = None,
        num_images_per_prompt: int = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        max_pixels: int = 1024 * 1024,
        max_input_image_side_length: int = 1024,
        align_res: bool = True,
        num_inference_steps: int = 28,
        text_guidance_scale: float = 4.0,
        image_guidance_scale: float = 1.0,
        cfg_range: tuple[float, float] = (0.0, 1.0),
        attention_kwargs: Optional[dict[str, Any]] = None,
        timesteps: list[int] = None,
        generator: Optional[Union[np.random.Generator, list[np.random.Generator]]] = None,
        latents: Optional[ms.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        verbose: bool = False,
        step_func=None,
    ):
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self._text_guidance_scale = text_guidance_scale
        self._image_guidance_scale = image_guidance_scale
        self._cfg_range = cfg_range
        self._attention_kwargs = attention_kwargs

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input promptb
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt,
            input_images,
            self.text_guidance_scale > 1.0,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            max_sequence_length=max_sequence_length,
            use_text_encoder_penultimate_layer_feats=use_text_encoder_penultimate_layer_feats,
        )

        dtype = self.vae.dtype
        # 3. Prepare control image
        ref_latents = self.prepare_image(
            images=input_images,
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            max_pixels=max_pixels,
            max_side_length=max_input_image_side_length,
            dtype=dtype,
        )

        if input_images is None:
            input_images = []

        if len(input_images) == 1 and align_res:
            width, height = (
                ref_latents[0][0].shape[-1] * self.vae_scale_factor,
                ref_latents[0][0].shape[-2] * self.vae_scale_factor,
            )
            ori_width, ori_height = width, height
        else:
            ori_width, ori_height = width, height

            cur_pixels = height * width
            ratio = (max_pixels / cur_pixels) ** 0.5
            ratio = min(ratio, 1.0)

            height, width = int(height * ratio) // 16 * 16, int(width * ratio) // 16 * 16

        if len(input_images) == 0:
            self._image_guidance_scale = 1

        # 4. Prepare latents.
        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt, latent_channels, height, width, prompt_embeds.dtype, generator, latents
        )

        freqs_cis = OmniGen2RotaryPosEmbed.get_freqs_cis(
            self.transformer.config.axes_dim_rope, self.transformer.config.axes_lens, theta=10000
        )

        image = self.processing(
            latents=latents,
            ref_latents=ref_latents,
            prompt_embeds=prompt_embeds,
            freqs_cis=freqs_cis,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            dtype=dtype,
            verbose=verbose,
            step_func=step_func,
        )

        image = F.interpolate(image, size=(ori_height, ori_width), mode="bilinear")

        image = self.image_processor.postprocess(image, output_type=output_type)

        return image

    # @torch.no_grad()  TODO
    def __call__(
        self,
        prompt: Optional[Union[str, list[str]]] = None,
        negative_prompt: Optional[Union[str, list[str]]] = None,
        prompt_embeds: Optional[ms.Tensor] = None,
        negative_prompt_embeds: Optional[ms.Tensor] = None,
        prompt_attention_mask: Optional[ms.Tensor] = None,
        negative_prompt_attention_mask: Optional[ms.Tensor] = None,
        use_text_encoder_penultimate_layer_feats: bool = False,
        max_sequence_length: Optional[int] = None,
        callback_on_step_end_tensor_inputs: Optional[list[str]] = None,
        input_images: Optional[list[Image]] = None,
        num_images_per_prompt: int = 1,
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        max_pixels: Optional[int] = 1024 * 1024,
        max_input_image_side_length: int = 1024,
        align_res: bool = True,
        num_inference_steps: int = 28,
        text_guidance_scale: float = 4.0,
        image_guidance_scale: float = 1.0,
        cfg_range: tuple[float, float] = (0.0, 1.0),
        attention_kwargs: Optional[dict[str, Any]] = None,
        timesteps: list[int] = None,
        generator: Optional[Union[np.random.Generator, list[np.random.Generator]]] = None,
        latents: Optional[ms.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        verbose: bool = False,
        step_func=None,
    ):
        assert isinstance(prompt, str), "prompt must be a string since chat mode only support one prompt per turn"

        # input_images = self.preprocess_images(input_images, max_input_image_size)
        prompt = self._apply_chat_template(prompt, input_images)
        generated_text = self.generate_text(prompt, input_images)[0]

        images = None
        if generated_text.startswith("<|img|>"):
            # TODO: reuse the hidden state when generate text instead of re-generating
            prompt = prompt + generated_text.split("<|img|>")[0]
            images = self.generate_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                use_text_encoder_penultimate_layer_feats=use_text_encoder_penultimate_layer_feats,
                max_sequence_length=max_sequence_length,
                input_images=input_images,
                num_images_per_prompt=num_images_per_prompt,
                height=height,
                width=width,
                max_pixels=max_pixels,
                max_input_image_side_length=max_input_image_side_length,
                align_res=align_res,
                num_inference_steps=num_inference_steps,
                text_guidance_scale=text_guidance_scale,
                image_guidance_scale=image_guidance_scale,
                cfg_range=cfg_range,
                timesteps=timesteps,
                generator=generator,
                latents=latents,
                return_dict=False,
                verbose=verbose,
                step_func=step_func,
            )

        generated_text = generated_text.replace("<|im_end|>", "")
        if not return_dict:
            return generated_text, images
        else:
            return OmniGen2PipelineOutput(text=generated_text, images=images)

    def processing(
        self,
        latents,
        ref_latents,
        prompt_embeds,
        freqs_cis,
        negative_prompt_embeds,
        prompt_attention_mask,
        negative_prompt_attention_mask,
        num_inference_steps,
        timesteps,
        dtype,
        verbose,
        step_func=None,
    ):
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, timesteps, num_tokens=latents.shape[-2] * latents.shape[-1]
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                model_pred = self.predict(
                    t=t,
                    latents=latents,
                    prompt_embeds=prompt_embeds,
                    freqs_cis=freqs_cis,
                    prompt_attention_mask=prompt_attention_mask,
                    ref_image_hidden_states=ref_latents,
                )

                text_guidance_scale = (
                    self.text_guidance_scale if self.cfg_range[0] <= i / len(timesteps) <= self.cfg_range[1] else 1.0
                )
                image_guidance_scale = (
                    self.image_guidance_scale if self.cfg_range[0] <= i / len(timesteps) <= self.cfg_range[1] else 1.0
                )
                if text_guidance_scale > 1.0 and image_guidance_scale > 1.0:
                    model_pred_ref = self.predict(
                        t=t,
                        latents=latents,
                        prompt_embeds=negative_prompt_embeds,
                        freqs_cis=freqs_cis,
                        prompt_attention_mask=negative_prompt_attention_mask,
                        ref_image_hidden_states=ref_latents,
                    )

                    if image_guidance_scale != 1:
                        model_pred_uncond = self.predict(
                            t=t,
                            latents=latents,
                            prompt_embeds=negative_prompt_embeds,
                            freqs_cis=freqs_cis,
                            prompt_attention_mask=negative_prompt_attention_mask,
                            ref_image_hidden_states=None,
                        )
                    else:
                        model_pred_uncond = mint.zeros_like(model_pred)

                    model_pred = (
                        model_pred_uncond
                        + image_guidance_scale * (model_pred_ref - model_pred_uncond)
                        + text_guidance_scale * (model_pred - model_pred_ref)
                    )
                elif text_guidance_scale > 1.0:
                    model_pred_uncond = self.predict(
                        t=t,
                        latents=latents,
                        prompt_embeds=negative_prompt_embeds,
                        freqs_cis=freqs_cis,
                        prompt_attention_mask=negative_prompt_attention_mask,
                        ref_image_hidden_states=None,
                    )
                    model_pred = model_pred_uncond + text_guidance_scale * (model_pred - model_pred_uncond)

                latents = self.scheduler.step(model_pred, t, latents, return_dict=False)[0]

                latents = latents.to(dtype=dtype)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if step_func is not None:
                    step_func(i, self._num_timesteps)

        latents = latents.to(dtype=dtype)
        if self.vae.config.scaling_factor is not None:
            latents = latents / self.vae.config.scaling_factor
        if self.vae.config.shift_factor is not None:
            latents = latents + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]

        return image

    def predict(self, t, latents, prompt_embeds, freqs_cis, prompt_attention_mask, ref_image_hidden_states):
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = mint.broadcast_to(t, (latents.shape[0],)).to(latents.dtype)

        batch_size, num_channels_latents, height, width = latents.shape

        optional_kwargs = {}
        if "ref_image_hidden_states" in set(inspect.signature(self.transformer.construct).parameters.keys()):
            optional_kwargs["ref_image_hidden_states"] = ref_image_hidden_states

        model_pred = self.transformer(
            latents, timestep, prompt_embeds, freqs_cis, prompt_attention_mask, **optional_kwargs
        )
        return model_pred
