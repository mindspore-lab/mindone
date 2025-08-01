# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# This code is adapted from https://github.com/huggingface/diffusers
# with modifications to run diffusers on mindspore.
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

from typing import Callable, List, Optional, Union

import numpy as np
import PIL.Image
from packaging import version
from PIL import Image
from transformers import XLMRobertaTokenizer

import mindspore as ms
from mindspore import mint

from ... import __version__
from ...models import UNet2DConditionModel, VQModel
from ...schedulers import DDIMScheduler
from ...utils import logging
from ...utils.mindspore_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from .text_encoder import MultilingualCLIP

XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from mindone.diffusers import KandinskyInpaintPipeline, KandinskyPriorPipeline
        >>> from mindone.diffusers.utils import load_image
        >>> import mindspore as ms
        >>> import numpy as np

        >>> pipe_prior = KandinskyPriorPipeline.from_pretrained(
        ...     "kandinsky-community/kandinsky-2-1-prior", mindspore_dtype=ms.float16
        ... )

        >>> prompt = "a hat"
        >>> image_emb, zero_image_emb = pipe_prior(prompt, return_dict=False)

        >>> pipe = KandinskyInpaintPipeline.from_pretrained(
        ...     "kandinsky-community/kandinsky-2-1-inpaint", mindspore_dtype=ms.float16
        ... )

        >>> init_image = load_image(
        ...     "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
        ...     "/kandinsky/cat.png"
        ... )

        >>> mask = np.zeros((768, 768), dtype=np.float32)
        >>> mask[:250, 250:-250] = 1

        >>> out = pipe(
        ...     prompt,
        ...     image=init_image,
        ...     mask_image=mask,
        ...     image_embeds=image_emb,
        ...     negative_image_embeds=zero_image_emb,
        ...     height=768,
        ...     width=768,
        ...     num_inference_steps=50,
        ... )

        >>> image = out[0][0]
        >>> image.save("cat_with_hat.png")
        ```
"""


def get_new_h_w(h, w, scale_factor=8):
    new_h = h // scale_factor**2
    if h % scale_factor**2 != 0:
        new_h += 1
    new_w = w // scale_factor**2
    if w % scale_factor**2 != 0:
        new_w += 1
    return new_h * scale_factor, new_w * scale_factor


def prepare_mask(masks):
    prepared_masks = []
    for mask in masks:
        old_mask = mask.copy()
        for i in range(mask.shape[1]):
            for j in range(mask.shape[2]):
                if old_mask[0][i][j] == 1:
                    continue
                if i != 0:
                    mask[:, i - 1, j] = 0
                if j != 0:
                    mask[:, i, j - 1] = 0
                if i != 0 and j != 0:
                    mask[:, i - 1, j - 1] = 0
                if i != mask.shape[1] - 1:
                    mask[:, i + 1, j] = 0
                if j != mask.shape[2] - 1:
                    mask[:, i, j + 1] = 0
                if i != mask.shape[1] - 1 and j != mask.shape[2] - 1:
                    mask[:, i + 1, j + 1] = 0
        prepared_masks.append(mask)
    return mint.stack(prepared_masks, dim=0)


def prepare_mask_and_masked_image(image, mask, height, width):
    r"""
    Prepares a pair (mask, image) to be consumed by the Kandinsky inpaint pipeline. This means that those inputs will
    be converted to ``ms.Tensor`` with shapes ``batch x channels x height x width`` where ``channels`` is ``3`` for
    the ``image`` and ``1`` for the ``mask``.

    The ``image`` will be converted to ``ms.float32`` and normalized to be in ``[-1, 1]``. The ``mask`` will be
    binarized (``mask > 0.5``) and cast to ``ms.float32`` too.

    Args:
        image (Union[np.array, PIL.Image, ms.Tensor]): The image to inpaint.
            It can be a ``PIL.Image``, or a ``height x width x 3`` ``np.array`` or a ``channels x height x width``
            ``ms.Tensor`` or a ``batch x channels x height x width`` ``ms.Tensor``.
        mask (_type_): The mask to apply to the image, i.e. regions to inpaint.
            It can be a ``PIL.Image``, or a ``height x width`` ``np.array`` or a ``1 x height x width``
            ``ms.Tensor`` or a ``batch x 1 x height x width`` ``ms.Tensor``.
        height (`int`, *optional*, defaults to 512):
            The height in pixels of the generated image.
        width (`int`, *optional*, defaults to 512):
            The width in pixels of the generated image.


    Raises:
        ValueError: ``ms.Tensor`` images should be in the ``[-1, 1]`` range. ValueError: ``ms.Tensor`` mask
        should be in the ``[0, 1]`` range. ValueError: ``mask`` and ``image`` should have the same spatial dimensions.
        TypeError: ``mask`` is a ``ms.Tensor`` but ``image`` is not
            (ot the other way around).

    Returns:
        tuple[ms.Tensor]: The pair (mask, image) as ``ms.Tensor`` with 4
            dimensions: ``batch x channels x height x width``.
    """

    if image is None:
        raise ValueError("`image` input cannot be undefined.")

    if mask is None:
        raise ValueError("`mask_image` input cannot be undefined.")

    if isinstance(image, ms.Tensor):
        if not isinstance(mask, ms.Tensor):
            raise TypeError(f"`image` is a ms.Tensor but `mask` (type: {type(mask)} is not")

        # Batch single image
        if image.ndim == 3:
            assert image.shape[0] == 3, "Image outside a batch should be of shape (3, H, W)"
            image = image.unsqueeze(0)

        # Batch and add channel dim for single mask
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)

        # Batch single mask or add channel dim
        if mask.ndim == 3:
            # Single batched mask, no channel dim or single mask not batched but channel dim
            if mask.shape[0] == 1:
                mask = mask.unsqueeze(0)

            # Batched masks no channel dim
            else:
                mask = mask.unsqueeze(1)

        assert image.ndim == 4 and mask.ndim == 4, "Image and Mask must have 4 dimensions"
        assert image.shape[-2:] == mask.shape[-2:], "Image and Mask must have the same spatial dimensions"
        assert image.shape[0] == mask.shape[0], "Image and Mask must have the same batch size"

        # Check image is in [-1, 1]
        if image.min() < -1 or image.max() > 1:
            raise ValueError("Image should be in [-1, 1] range")

        # Check mask is in [0, 1]
        if mask.min() < 0 or mask.max() > 1:
            raise ValueError("Mask should be in [0, 1] range")

        # Binarize mask
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        # Image as float32
        image = image.to(dtype=ms.float32)
    elif isinstance(mask, ms.Tensor):
        raise TypeError(f"`mask` is a ms.Tensor but `image` (type: {type(image)} is not")
    else:
        # preprocess image
        if isinstance(image, (PIL.Image.Image, np.ndarray)):
            image = [image]

        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            # resize all images w.r.t passed height an width
            image = [i.resize((width, height), resample=Image.BICUBIC, reducing_gap=1) for i in image]
            image = [np.array(i.convert("RGB"))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.concatenate([i[None, :] for i in image], axis=0)

        image = image.transpose(0, 3, 1, 2)
        image = ms.Tensor.from_numpy(image).to(dtype=ms.float32) / 127.5 - 1.0

        # preprocess mask
        if isinstance(mask, (PIL.Image.Image, np.ndarray)):
            mask = [mask]

        if isinstance(mask, list) and isinstance(mask[0], PIL.Image.Image):
            mask = [i.resize((width, height), resample=PIL.Image.LANCZOS) for i in mask]
            mask = np.concatenate([np.array(m.convert("L"))[None, None, :] for m in mask], axis=0)
            mask = mask.astype(np.float32) / 255.0
        elif isinstance(mask, list) and isinstance(mask[0], np.ndarray):
            mask = np.concatenate([m[None, None, :] for m in mask], axis=0)

        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = ms.Tensor.from_numpy(mask)

    mask = 1 - mask

    return mask, image


class KandinskyInpaintPipeline(DiffusionPipeline):
    """
    Pipeline for text-guided image inpainting using Kandinsky2.1

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        text_encoder ([`MultilingualCLIP`]):
            Frozen text-encoder.
        tokenizer ([`XLMRobertaTokenizer`]):
            Tokenizer of class
        scheduler ([`DDIMScheduler`]):
            A scheduler to be used in combination with `unet` to generate image latents.
        unet ([`UNet2DConditionModel`]):
            Conditional U-Net architecture to denoise the image embedding.
        movq ([`VQModel`]):
            MoVQ image encoder and decoder
    """

    def __init__(
        self,
        text_encoder: MultilingualCLIP,
        movq: VQModel,
        tokenizer: XLMRobertaTokenizer,
        unet: UNet2DConditionModel,
        scheduler: DDIMScheduler,
    ):
        super().__init__()

        self.register_modules(
            text_encoder=text_encoder,
            movq=movq,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )
        self.movq_scale_factor = 2 ** (len(self.movq.config.block_out_channels) - 1)
        self._warn_has_been_called = False

    # Copied from diffusers.pipelines.unclip.pipeline_unclip.UnCLIPPipeline.prepare_latents
    def prepare_latents(self, shape, dtype, generator, latents, scheduler):
        if latents is None:
            latents = randn_tensor(shape, generator=generator, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")

        latents = (latents * scheduler.init_noise_sigma).to(dtype)
        return latents

    def _encode_prompt(
        self,
        prompt,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
    ):
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        # get prompt text embeddings
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="np",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="np").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not np.array_equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        text_mask = ms.tensor(text_inputs.attention_mask)

        prompt_embeds, text_encoder_hidden_states = self.text_encoder(
            input_ids=ms.tensor(text_input_ids), attention_mask=text_mask
        )

        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        text_encoder_hidden_states = text_encoder_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
        text_mask = text_mask.repeat_interleave(num_images_per_prompt, dim=0)

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

            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="np",
            )
            uncond_text_input_ids = ms.tensor(uncond_input.input_ids)
            uncond_text_mask = ms.tensor(uncond_input.attention_mask)

            negative_prompt_embeds, uncond_text_encoder_hidden_states = self.text_encoder(
                input_ids=uncond_text_input_ids, attention_mask=uncond_text_mask
            )

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
            prompt_embeds = mint.cat([negative_prompt_embeds, prompt_embeds])
            text_encoder_hidden_states = mint.cat([uncond_text_encoder_hidden_states, text_encoder_hidden_states])

            text_mask = mint.cat([uncond_text_mask, text_mask])

        return prompt_embeds, text_encoder_hidden_states, text_mask

    def __call__(
        self,
        prompt: Union[str, List[str]],
        image: Union[ms.Tensor, PIL.Image.Image],
        mask_image: Union[ms.Tensor, PIL.Image.Image, np.ndarray],
        image_embeds: ms.Tensor,
        negative_image_embeds: ms.Tensor,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 100,
        guidance_scale: float = 4.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[np.random.Generator, List[np.random.Generator]]] = None,
        latents: Optional[ms.Tensor] = None,
        output_type: Optional[str] = "pil",
        callback: Optional[Callable[[int, int, ms.Tensor], None]] = None,
        callback_steps: int = 1,
        return_dict: bool = False,
    ):
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            image (`ms.Tensor`, `PIL.Image.Image` or `np.ndarray`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process.
            mask_image (`PIL.Image.Image`,`ms.Tensor` or `np.ndarray`):
                `Image`, or a tensor representing an image batch, to mask `image`. White pixels in the mask will be
                repainted, while black pixels will be preserved. You can pass a mindspore tensor as mask only if the
                image you passed is a mindspore tensor, and it should contain one color channel (L) instead of 3, so the
                expected shape would be either `(B, 1, H, W,)`, `(B, H, W)`, `(1, H, W)` or `(H, W)` If image is an PIL
                image or numpy array, mask should also be a either PIL image or numpy array. If it is a PIL image, it
                will be converted to a single channel (luminance) before use. If it is a nummpy array, the expected
                shape is `(H, W)`.
            image_embeds (`ms.Tensor` or `List[ms.Tensor]`):
                The clip image embeddings for text prompt, that will be used to condition the image generation.
            negative_image_embeds (`ms.Tensor` or `List[ms.Tensor]`):
                The clip image embeddings for negative text prompt, will be used to condition the image generation.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`np.random.Generator` or `List[np.random.Generator]`, *optional*):
                One or a list of [np.random.Generator(s)](https://numpy.org/doc/stable/reference/random/generator.html)
                to make generation deterministic.
            latents (`ms.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between: `"pil"` (`PIL.Image.Image`), `"np"`
                (`np.array`) or `"pt"` (`ms.Tensor`).
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: ms.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            return_dict (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Examples:

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`
        """
        if not self._warn_has_been_called and version.parse(version.parse(__version__).base_version) < version.parse(
            "0.23.0.dev0"
        ):
            logger.warning(
                "Please note that the expected format of `mask_image` has recently been changed. "
                "Before diffusers == 0.19.0, Kandinsky Inpainting pipelines repainted black pixels and preserved black pixels. "
                "As of diffusers==0.19.0 this behavior has been inverted. Now white pixels are repainted and black pixels are preserved. "
                "This way, Kandinsky's masking behavior is aligned with Stable Diffusion. "
                "THIS means that you HAVE to invert the input mask to have the same behavior as before as explained in https://github.com/huggingface/diffusers/pull/4207. "  # noqa
                "This warning will be suppressed after the first inference call and will be removed in diffusers>0.23.0"
            )
            self._warn_has_been_called = True

        # Define call parameters
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        batch_size = batch_size * num_images_per_prompt
        do_classifier_free_guidance = guidance_scale > 1.0

        prompt_embeds, text_encoder_hidden_states, _ = self._encode_prompt(
            prompt, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        if isinstance(image_embeds, list):
            image_embeds = mint.cat(image_embeds, dim=0)
        if isinstance(negative_image_embeds, list):
            negative_image_embeds = mint.cat(negative_image_embeds, dim=0)

        if do_classifier_free_guidance:
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            negative_image_embeds = negative_image_embeds.repeat_interleave(num_images_per_prompt, dim=0)

            image_embeds = mint.cat([negative_image_embeds, image_embeds], dim=0).to(dtype=prompt_embeds.dtype)

        # preprocess image and mask
        mask_image, image = prepare_mask_and_masked_image(image, mask_image, height, width)

        image = image.to(dtype=prompt_embeds.dtype)
        image = self.movq.encode(image)[0]

        mask_image = mask_image.to(dtype=prompt_embeds.dtype)

        image_shape = tuple(image.shape[-2:])
        mask_image = mint.nn.functional.interpolate(
            mask_image,
            image_shape,
            mode="nearest",
        )
        mask_image = prepare_mask(mask_image)
        masked_image = image * mask_image

        mask_image = mask_image.repeat_interleave(num_images_per_prompt, dim=0)
        masked_image = masked_image.repeat_interleave(num_images_per_prompt, dim=0)
        if do_classifier_free_guidance:
            mask_image = mask_image.tile((2, 1, 1, 1))
            masked_image = masked_image.tile((2, 1, 1, 1))

        self.scheduler.set_timesteps(num_inference_steps)
        timesteps_tensor = self.scheduler.timesteps

        num_channels_latents = self.movq.config.latent_channels

        # get h, w for latents
        sample_height, sample_width = get_new_h_w(height, width, self.movq_scale_factor)

        # create initial latent
        latents = self.prepare_latents(
            (batch_size, num_channels_latents, sample_height, sample_width),
            text_encoder_hidden_states.dtype,
            generator,
            latents,
            self.scheduler,
        )

        # Check that sizes of mask, masked image and latents match with expected
        num_channels_mask = mask_image.shape[1]
        num_channels_masked_image = masked_image.shape[1]
        if num_channels_latents + num_channels_mask + num_channels_masked_image != self.unet.config.in_channels:
            raise ValueError(
                f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                f" = {num_channels_latents + num_channels_masked_image + num_channels_mask}. Please verify the config of"
                " `pipeline.unet` or your `mask_image` or `image` input."
            )

        for i, t in enumerate(self.progress_bar(timesteps_tensor)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = mint.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = mint.cat([latent_model_input, masked_image, mask_image], dim=1)

            added_cond_kwargs = {"text_embeds": prompt_embeds, "image_embeds": image_embeds}
            noise_pred = self.unet(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=text_encoder_hidden_states,
                added_cond_kwargs=ms.mutable(added_cond_kwargs),
                return_dict=False,
            )[0]

            if do_classifier_free_guidance:
                noise_pred, variance_pred = noise_pred.split(latents.shape[1], dim=1)
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                _, variance_pred_text = variance_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                noise_pred = mint.cat([noise_pred, variance_pred_text], dim=1)

            if not (
                hasattr(self.scheduler.config, "variance_type")
                and self.scheduler.config.variance_type in ["learned", "learned_range"]
            ):
                noise_pred, _ = noise_pred.split(latents.shape[1], dim=1)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(
                noise_pred,
                t,
                latents,
                generator=generator,
            )[0]

            if callback is not None and i % callback_steps == 0:
                step_idx = i // getattr(self.scheduler, "order", 1)
                callback(step_idx, t, latents)

        # post-processing
        image = self.movq.decode(latents, force_not_quantize=True)[0]

        if output_type not in ["ms", "np", "pil"]:
            raise ValueError(f"Only the output types `pt`, `pil` and `np` are supported not output_type={output_type}")

        if output_type in ["np", "pil"]:
            image = image * 0.5 + 0.5
            image = image.clamp(0, 1)
            image = image.permute((0, 2, 3, 1)).float().numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
