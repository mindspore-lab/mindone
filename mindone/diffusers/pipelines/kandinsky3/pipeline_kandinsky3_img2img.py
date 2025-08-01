"""Adapted from https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/kandinsky3/pipeline_kandinsky3_img2img.py."""

import inspect
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import PIL
import PIL.Image
from transformers import T5Tokenizer

import mindspore as ms
from mindspore import mint

from mindone.transformers import T5EncoderModel

from ...image_processor import VaeImageProcessor
from ...loaders import StableDiffusionLoraLoaderMixin
from ...models import Kandinsky3UNet, VQModel
from ...schedulers import DDPMScheduler
from ...utils import deprecate, logging
from ...utils.mindspore_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput

XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from mindone.diffusers import Kandinsky3Img2ImgPipeline
        >>> from mindone.diffusers.utils import load_image
        >>> import mindspore as ms
        >>> import numpy as np

        >>> pipe = Kandinsky3Img2ImgPipeline.from_pretrained("kandinsky-community/kandinsky-3", variant="fp16", mindspore_dtype=ms.float16)

        >>> prompt = "A painting of the inside of a subway train with tiny raccoons."
        >>> image = load_image("https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky3/t2i.png")

        >>> generator = np.random.Generator(np.random.PCG64(43))
        >>> image = pipe(prompt, image=image, strength=0.75, num_inference_steps=25, generator=generator)[0][0]
        ```
"""


class Kandinsky3Img2ImgPipeline(DiffusionPipeline, StableDiffusionLoraLoaderMixin):
    model_cpu_offload_seq = "text_encoder->movq->unet->movq"
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
        "negative_attention_mask",
        "attention_mask",
    ]

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        unet: Kandinsky3UNet,
        scheduler: DDPMScheduler,
        movq: VQModel,
    ):
        super().__init__()

        self.register_modules(tokenizer=tokenizer, text_encoder=text_encoder, unet=unet, scheduler=scheduler, movq=movq)
        movq_scale_factor = 2 ** (len(self.movq.config.block_out_channels) - 1) if getattr(self, "movq", None) else 8
        movq_latent_channels = self.movq.config.latent_channels if getattr(self, "movq", None) else 4
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=movq_scale_factor,
            vae_latent_channels=movq_latent_channels,
            resample="bicubic",
            reducing_gap=1,
        )

    def get_timesteps(self, num_inference_steps, strength):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start

    def _process_embeds(self, embeddings, attention_mask, cut_context):
        # return embeddings, attention_mask
        if cut_context:
            embeddings[attention_mask == 0] = mint.zeros_like(embeddings[attention_mask == 0])
            max_seq_length = attention_mask.sum(dim=-1).max() + 1
            embeddings = embeddings[:, :max_seq_length]
            attention_mask = attention_mask[:, :max_seq_length]
        return embeddings, attention_mask

    def encode_prompt(
        self,
        prompt,
        do_classifier_free_guidance=True,
        num_images_per_prompt=1,
        negative_prompt=None,
        prompt_embeds: Optional[ms.Tensor] = None,
        negative_prompt_embeds: Optional[ms.Tensor] = None,
        _cut_context=False,
        attention_mask: Optional[ms.Tensor] = None,
        negative_attention_mask: Optional[ms.Tensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            prompt_embeds (`ms.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`ms.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            attention_mask (`ms.Tensor`, *optional*):
                Pre-generated attention mask. Must provide if passing `prompt_embeds` directly.
            negative_attention_mask (`ms.Tensor`, *optional*):
                Pre-generated negative attention mask. Must provide if passing `negative_prompt_embeds` directly.
        """
        if prompt is not None and negative_prompt is not None:
            if type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        max_length = 128

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="np",
            )
            text_input_ids = ms.tensor(text_inputs.input_ids)
            attention_mask = ms.tensor(text_inputs.attention_mask)
            prompt_embeds = self.text_encoder(
                text_input_ids,
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]
            prompt_embeds, attention_mask = self._process_embeds(prompt_embeds, attention_mask, _cut_context)
            prompt_embeds = prompt_embeds * attention_mask.unsqueeze(2)

        if self.text_encoder is not None:
            dtype = self.text_encoder.dtype
        else:
            dtype = None

        prompt_embeds = prompt_embeds.to(dtype=dtype)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.tile((1, num_images_per_prompt, 1))
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        attention_mask = attention_mask.tile((num_images_per_prompt, 1))
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]

            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
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
            if negative_prompt is not None:
                uncond_input = self.tokenizer(
                    uncond_tokens,
                    padding="max_length",
                    max_length=128,
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors="np",
                )
                text_input_ids = ms.tensor(uncond_input.input_ids)
                negative_attention_mask = ms.tensor(uncond_input.attention_mask)

                negative_prompt_embeds = self.text_encoder(
                    text_input_ids,
                    attention_mask=negative_attention_mask,
                )
                negative_prompt_embeds = negative_prompt_embeds[0]
                negative_prompt_embeds = negative_prompt_embeds[:, : prompt_embeds.shape[1]]
                negative_attention_mask = negative_attention_mask[:, : prompt_embeds.shape[1]]
                negative_prompt_embeds = negative_prompt_embeds * negative_attention_mask.unsqueeze(2)

            else:
                negative_prompt_embeds = mint.zeros_like(prompt_embeds)
                negative_attention_mask = mint.zeros_like(attention_mask)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype)
            if negative_prompt_embeds.shape != prompt_embeds.shape:
                negative_prompt_embeds = negative_prompt_embeds.tile((1, num_images_per_prompt, 1))
                negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
                negative_attention_mask = negative_attention_mask.tile((num_images_per_prompt, 1))

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
        else:
            negative_prompt_embeds = None
            negative_attention_mask = None
        return prompt_embeds, negative_prompt_embeds, attention_mask, negative_attention_mask

    def prepare_latents(self, image, timestep, batch_size, num_images_per_prompt, dtype, generator=None):
        if not isinstance(image, (ms.Tensor, PIL.Image.Image, list)):
            raise ValueError(f"`image` has to be of type `ms.Tensor`, `PIL.Image.Image` or list but is {type(image)}")

        image = image.to(dtype=dtype)

        batch_size = batch_size * num_images_per_prompt

        if image.shape[1] == 4:
            init_latents = image

        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            elif isinstance(generator, list):
                init_latents = [
                    self.movq.diag_gauss_dist.sample(self.movq.encode(image[i : i + 1])[0]) for i in range(batch_size)
                ]
                init_latents = mint.cat(init_latents, dim=0)
            else:
                init_latents = self.movq.diag_gauss_dist.sample(self.movq.encode(image)[0])

            init_latents = self.movq.config.scaling_factor * init_latents

        init_latents = mint.cat([init_latents], dim=0)

        shape = init_latents.shape
        noise = randn_tensor(shape, generator=generator, dtype=dtype)

        # get latents
        init_latents = self.scheduler.add_noise(init_latents, noise, timestep)

        latents = init_latents

        return latents

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://huggingface.co/papers/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        attention_mask=None,
        negative_attention_mask=None,
    ):
        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, "
                f"but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if negative_prompt_embeds is not None and negative_attention_mask is None:
            raise ValueError("Please provide `negative_attention_mask` along with `negative_prompt_embeds`")

        if negative_prompt_embeds is not None and negative_attention_mask is not None:
            if negative_prompt_embeds.shape[:2] != negative_attention_mask.shape:
                raise ValueError(
                    "`negative_prompt_embeds` and `negative_attention_mask` must have the same batch_size and token length when passed directly, but"
                    f" got: `negative_prompt_embeds` {negative_prompt_embeds.shape[:2]} != `negative_attention_mask`"
                    f" {negative_attention_mask.shape}."
                )

        if prompt_embeds is not None and attention_mask is None:
            raise ValueError("Please provide `attention_mask` along with `prompt_embeds`")

        if prompt_embeds is not None and attention_mask is not None:
            if prompt_embeds.shape[:2] != attention_mask.shape:
                raise ValueError(
                    "`prompt_embeds` and `attention_mask` must have the same batch_size and token length when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape[:2]} != `attention_mask`"
                    f" {attention_mask.shape}."
                )

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[ms.Tensor, PIL.Image.Image, List[ms.Tensor], List[PIL.Image.Image]] = None,
        strength: float = 0.3,
        num_inference_steps: int = 25,
        guidance_scale: float = 3.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[np.random.Generator, List[np.random.Generator]]] = None,
        prompt_embeds: Optional[ms.Tensor] = None,
        negative_prompt_embeds: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        negative_attention_mask: Optional[ms.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = False,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            image (`ms.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[ms.Tensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process.
            strength (`float`, *optional*, defaults to 0.8):
                Indicates extent to transform the reference `image`. Must be between 0 and 1. `image` is used as a
                starting point and more noise is added the higher the `strength`. The number of denoising steps depends
                on the amount of noise initially added. When `strength` is 1, added noise is maximum and the denoising
                process runs for the full number of iterations specified in `num_inference_steps`. A value of 1
                essentially ignores `image`.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 3.0):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`np.random.Generator` or `List[np.random.Generator]`, *optional*):
                One or a list of [np.random.Generator(s)](https://numpy.org/doc/stable/reference/random/generator.html)
                to make generation deterministic.
            prompt_embeds (`ms.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`ms.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            attention_mask (`ms.Tensor`, *optional*):
                Pre-generated attention mask. Must provide if passing `prompt_embeds` directly.
            negative_attention_mask (`ms.Tensor`, *optional*):
                Pre-generated negative attention mask. Must provide if passing `negative_prompt_embeds` directly.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~pipelines.stable_diffusion.IFPipelineOutput`] instead of a plain tuple.
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
            [`~pipelines.ImagePipelineOutput`] or `tuple`

        """
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, "
                f"but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        cut_context = True
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
            attention_mask,
            negative_attention_mask,
        )

        self._guidance_scale = guidance_scale

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds, attention_mask, negative_attention_mask = self.encode_prompt(
            prompt,
            self.do_classifier_free_guidance,
            num_images_per_prompt=num_images_per_prompt,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            _cut_context=cut_context,
            attention_mask=attention_mask,
            negative_attention_mask=negative_attention_mask,
        )

        if self.do_classifier_free_guidance:
            prompt_embeds = mint.cat([negative_prompt_embeds, prompt_embeds])
            attention_mask = mint.cat([negative_attention_mask, attention_mask])
        if not isinstance(image, list):
            image = [image]
        if not all(isinstance(i, (PIL.Image.Image, ms.Tensor)) for i in image):
            raise ValueError(
                f"Input is in incorrect format: {[type(i) for i in image]}. Currently, we only support  PIL image and mindspore tensor"
            )

        image = mint.cat([self.image_processor.preprocess(i) for i in image], dim=0)
        image = image.to(dtype=prompt_embeds.dtype)
        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength)
        # 5. Prepare latents
        latents = self.movq.encode(image)[0]
        latents = latents.repeat_interleave(num_images_per_prompt, dim=0)
        latent_timestep = timesteps[:1].tile((batch_size * num_images_per_prompt,))
        latents = self.prepare_latents(
            latents, latent_timestep, batch_size, num_images_per_prompt, prompt_embeds.dtype, generator
        )

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = mint.cat([latents] * 2) if self.do_classifier_free_guidance else latents

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=attention_mask,
                )[0]
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

                    noise_pred = (guidance_scale + 1.0) * noise_pred_text - guidance_scale * noise_pred_uncond

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred,
                    t,
                    latents,
                    generator=generator,
                )[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    attention_mask = callback_outputs.pop("attention_mask", attention_mask)
                    negative_attention_mask = callback_outputs.pop("negative_attention_mask", negative_attention_mask)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

            # post-processing
            if not output_type == "latent":
                image = self.movq.decode(latents, force_not_quantize=True)[0]
                image = self.image_processor.postprocess(image, output_type)
            else:
                image = latents

            if not return_dict:
                return (image,)

            return ImagePipelineOutput(images=image)
