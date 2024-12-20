from typing import Callable, Dict, List, Optional, Union

import numpy as np
import PIL.Image
from transformers import CLIPImageProcessor, CLIPTokenizer

import mindspore as ms
from mindspore import ops

from mindone.transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection

from ...models import PriorTransformer
from ...schedulers import UnCLIPScheduler
from ...utils import logging
from ...utils.mindspore_utils import randn_tensor
from ..kandinsky import KandinskyPriorPipelineOutput
from ..pipeline_utils import DiffusionPipeline

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from mindone.diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
        >>> import mindspore as ms

        >>> pipe_prior = KandinskyV22PriorPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-prior")
        >>> prompt = "red cat, 4k photo"
        >>> image_emb, negative_image_emb = pipe_prior(prompt)

        >>> pipe = KandinskyV22Pipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder")
        >>> image = pipe(
        ...     image_embeds=image_emb,
        ...     negative_image_embeds=negative_image_emb,
        ...     height=768,
        ...     width=768,
        ...     num_inference_steps=50,
        ... )[0]
        >>> image[0].save("cat.png")
        ```
"""

EXAMPLE_INTERPOLATE_DOC_STRING = """
    Examples:
        ```py
        >>> from mindone.diffusers import KandinskyV22PriorPipeline, KandinskyV22Pipeline
        >>> from mindone.diffusers.utils import load_image
        >>> import PIL
        >>> import mindspore as ms

        >>> pipe_prior = KandinskyV22PriorPipeline.from_pretrained(
        ...     "kandinsky-community/kandinsky-2-2-prior", mindspore_dtype=ms.float16
        ... )

        >>> img1 = load_image(
        ...     "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
        ...     "/kandinsky/cat.png"
        ... )
        >>> img2 = load_image(
        ...     "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
        ...     "/kandinsky/starry_night.jpeg"
        ... )
        >>> images_texts = ["a cat", img1, img2]
        >>> weights = [0.3, 0.3, 0.4]
        >>> out = pipe_prior.interpolate(images_texts, weights)
        >>> pipe = KandinskyV22Pipeline.from_pretrained(
        ...     "kandinsky-community/kandinsky-2-2-decoder", mindspore_dtype=ms.float16
        ... )
        >>> image = pipe(
        ...     image_embeds=out[0],
        ...     negative_image_embeds=out[1],
        ...     height=768,
        ...     width=768,
        ...     num_inference_steps=50,
        ... )[0][0]
        >>> image.save("starry_cat.png")
        ```
"""


class KandinskyV22PriorPipeline(DiffusionPipeline):
    """
    Pipeline for generating image prior for Kandinsky

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        prior ([`PriorTransformer`]):
            The canonincal unCLIP prior to approximate the image embedding from the text embedding.
        image_encoder ([`CLIPVisionModelWithProjection`]):
            Frozen image-encoder.
        text_encoder ([`CLIPTextModelWithProjection`]):
            Frozen text-encoder.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        scheduler ([`UnCLIPScheduler`]):
            A scheduler to be used in combination with `prior` to generate image embedding.
        image_processor ([`CLIPImageProcessor`]):
            A image_processor to be used to preprocess image from clip.
    """

    model_cpu_offload_seq = "text_encoder->image_encoder->prior"
    _exclude_from_cpu_offload = ["prior"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "text_encoder_hidden_states", "text_mask"]

    def __init__(
        self,
        prior: PriorTransformer,
        image_encoder: CLIPVisionModelWithProjection,
        text_encoder: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        scheduler: UnCLIPScheduler,
        image_processor: CLIPImageProcessor,
    ):
        super().__init__()

        self.register_modules(
            prior=prior,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            image_encoder=image_encoder,
            image_processor=image_processor,
        )

    def interpolate(
        self,
        images_and_prompts: List[Union[str, PIL.Image.Image, ms.Tensor]],
        weights: List[float],
        num_images_per_prompt: int = 1,
        num_inference_steps: int = 25,
        generator: Optional[Union[np.random.Generator, List[np.random.Generator]]] = None,
        latents: Optional[ms.Tensor] = None,
        negative_prior_prompt: Optional[str] = None,
        negative_prompt: str = "",
        guidance_scale: float = 4.0,
    ):
        """
        Function invoked when using the prior pipeline for interpolation.

        Args:
            images_and_prompts (`List[Union[str, PIL.Image.Image, ms.Tensor]]`):
                list of prompts and images to guide the image generation.
            weights: (`List[float]`):
                list of weights for each condition in `images_and_prompts`
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            generator (`np.random.Generator` or `List[np.random.Generator]`, *optional*):
                One or a list of [np.random.Generator(s)](https://numpy.org/doc/stable/reference/random/generator.html)
                to make generation deterministic.
            latents (`ms.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            negative_prior_prompt (`str`, *optional*):
                The prompt not to guide the prior diffusion process. Ignored when not using guidance (i.e., ignored if
                `guidance_scale` is less than `1`).
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt not to guide the image generation. Ignored when not using guidance (i.e., ignored if
                `guidance_scale` is less than `1`).
            guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.

        Examples:

        Returns:
            [`KandinskyPriorPipelineOutput`] or `tuple`
        """
        if len(images_and_prompts) != len(weights):
            raise ValueError(
                f"`images_and_prompts` contains {len(images_and_prompts)} items and "
                f"`weights` contains {len(weights)} items - they should be lists of same length"
            )

        image_embeddings = []
        for cond, weight in zip(images_and_prompts, weights):
            if isinstance(cond, str):
                image_emb = self(
                    cond,
                    num_inference_steps=num_inference_steps,
                    num_images_per_prompt=num_images_per_prompt,
                    generator=generator,
                    latents=latents,
                    negative_prompt=negative_prior_prompt,
                    guidance_scale=guidance_scale,
                )[0].unsqueeze(0)

            elif isinstance(cond, (PIL.Image.Image, ms.Tensor)):
                if isinstance(cond, PIL.Image.Image):
                    cond = (
                        ms.tensor(self.image_processor(cond, return_tensors="np").pixel_values[0])
                        .unsqueeze(0)
                        .to(dtype=self.image_encoder.dtype)
                    )

                image_emb = self.image_encoder(cond)[0].tile((num_images_per_prompt, 1)).unsqueeze(0)

            else:
                raise ValueError(
                    f"`images_and_prompts` can only contains elements to be of type `str`, `PIL.Image.Image` or `ms.Tensor`  but is {type(cond)}"
                )

            image_embeddings.append(image_emb * weight)

        image_emb = ops.cat(image_embeddings).sum(axis=0)

        out_zero = self(
            negative_prompt,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            latents=latents,
            negative_prompt=negative_prior_prompt,
            guidance_scale=guidance_scale,
        )
        zero_image_emb = out_zero[1] if negative_prompt == "" else out_zero[0]

        return (image_emb, zero_image_emb)

    # Copied from diffusers.pipelines.unclip.pipeline_unclip.UnCLIPPipeline.prepare_latents
    def prepare_latents(self, shape, dtype, generator, latents, scheduler):
        if latents is None:
            latents = randn_tensor(shape, generator=generator, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")

        latents = (latents * scheduler.init_noise_sigma).to(dtype)
        return latents

    # Copied from diffusers.pipelines.kandinsky.pipeline_kandinsky_prior.KandinskyPriorPipeline.get_zero_embed
    def get_zero_embed(self, batch_size=1):
        zero_img = ops.zeros((1, 3, self.image_encoder.config.image_size, self.image_encoder.config.image_size)).to(
            dtype=self.image_encoder.dtype
        )
        zero_image_emb = self.image_encoder(zero_img)[0]
        zero_image_emb = zero_image_emb.tile((batch_size, 1))
        return zero_image_emb

    # Copied from diffusers.pipelines.kandinsky.pipeline_kandinsky_prior.KandinskyPriorPipeline._encode_prompt
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
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        )
        text_input_ids = text_inputs.input_ids
        text_mask = ms.tensor(text_inputs.attention_mask)

        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="np").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not np.array_equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )
            text_input_ids = text_input_ids[:, : self.tokenizer.model_max_length]

        text_encoder_output = self.text_encoder(ms.tensor(text_input_ids))

        prompt_embeds = text_encoder_output[0]
        text_encoder_hidden_states = text_encoder_output[1]

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
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="np",
            )
            uncond_text_mask = ms.tensor(uncond_input.attention_mask)
            negative_prompt_embeds_text_encoder_output = self.text_encoder(ms.tensor(uncond_input.input_ids))

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

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        num_inference_steps: int = 25,
        generator: Optional[Union[np.random.Generator, List[np.random.Generator]]] = None,
        latents: Optional[ms.Tensor] = None,
        guidance_scale: float = 4.0,
        output_type: Optional[str] = "ms",  # ms only
        return_dict: bool = False,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    ):
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            generator (`np.random.Generator` or `List[np.random.Generator]`, *optional*):
                One or a list of [np.random.Generator(s)](https://numpy.org/doc/stable/reference/random/generator.html)
                to make generation deterministic.
            latents (`ms.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            output_type (`str`, *optional*, defaults to `"ms"`):
                The output format of the generate image. Choose between: `"np"` (`np.array`) or `"ms"`
                (`ms.Tensor`).
            return_dict (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`KandinskyPriorPipelineOutput`] or `tuple`
        """

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, "
                f"but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if isinstance(prompt, str):
            prompt = [prompt]
        elif not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        elif not isinstance(negative_prompt, list) and negative_prompt is not None:
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

        # if the negative prompt is defined we double the batch size to
        # directly retrieve the negative prompt embedding
        if negative_prompt is not None:
            prompt = prompt + negative_prompt
            negative_prompt = 2 * negative_prompt

        batch_size = len(prompt)
        batch_size = batch_size * num_images_per_prompt

        self._guidance_scale = guidance_scale

        prompt_embeds, text_encoder_hidden_states, text_mask = self._encode_prompt(
            prompt, num_images_per_prompt, self.do_classifier_free_guidance, negative_prompt
        )

        # prior
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        embedding_dim = self.prior.config.embedding_dim

        latents = self.prepare_latents(
            (batch_size, embedding_dim),
            prompt_embeds.dtype,
            generator,
            latents,
            self.scheduler,
        )
        self._num_timesteps = len(timesteps)
        for i, t in enumerate(self.progress_bar(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = ops.cat([latents] * 2) if self.do_classifier_free_guidance else latents

            predicted_image_embedding = self.prior(
                latent_model_input,
                timestep=t,
                proj_embedding=prompt_embeds,
                encoder_hidden_states=text_encoder_hidden_states,
                attention_mask=text_mask,
            )[0]

            if self.do_classifier_free_guidance:
                predicted_image_embedding_uncond, predicted_image_embedding_text = predicted_image_embedding.chunk(2)
                predicted_image_embedding = predicted_image_embedding_uncond + self.guidance_scale * (
                    predicted_image_embedding_text - predicted_image_embedding_uncond
                )

            if i + 1 == timesteps.shape[0]:
                prev_timestep = None
            else:
                prev_timestep = timesteps[i + 1]

            latents = self.scheduler.step(
                predicted_image_embedding,
                timestep=t,
                sample=latents,
                generator=generator,
                prev_timestep=prev_timestep,
            )[0]

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                text_encoder_hidden_states = callback_outputs.pop(
                    "text_encoder_hidden_states", text_encoder_hidden_states
                )
                text_mask = callback_outputs.pop("text_mask", text_mask)

        latents = self.prior.post_process_latents(latents)

        image_embeddings = latents

        # if negative prompt has been defined, we retrieve split the image embedding into two
        if negative_prompt is None:
            zero_embeds = self.get_zero_embed(latents.shape[0])
        else:
            image_embeddings, zero_embeds = image_embeddings.chunk(2)

        if output_type not in ["ms", "np"]:
            raise ValueError(f"Only the output types `pt` and `np` are supported not output_type={output_type}")

        if output_type == "np":
            image_embeddings = image_embeddings.numpy()
            zero_embeds = zero_embeds.numpy()

        if not return_dict:
            return (image_embeddings, zero_embeds)

        return KandinskyPriorPipelineOutput(image_embeds=image_embeddings, negative_image_embeds=zero_embeds)
