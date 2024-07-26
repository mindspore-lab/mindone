# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from PIL import Image

import mindspore as ms
from mindspore import ops

from ...models import UNet2DConditionModel, VQModel
from ...schedulers import DDPMScheduler
from ...utils import logging
from ...utils.mindspore_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import mindspore as ms
        >>> import numpy as np

        >>> from mindone.diffusers import KandinskyV22PriorEmb2EmbPipeline, KandinskyV22ControlnetImg2ImgPipeline
        >>> from transformers import pipeline
        >>> from mindone.diffusers.utils import load_image


        >>> def make_hint(image, depth_estimator):
        ...     image = depth_estimator(image)["depth"]
        ...     image = np.array(image)
        ...     image = image[:, :, None]
        ...     image = np.concatenate([image, image, image], axis=2)
        ...     detected_map = ms.tensor(image).float() / 255.0
        ...     hint = detected_map.permute(2, 0, 1)
        ...     return hint


        >>> depth_estimator = pipeline("depth-estimation")

        >>> pipe_prior = KandinskyV22PriorEmb2EmbPipeline.from_pretrained(
        ...     "kandinsky-community/kandinsky-2-2-prior", mindspore_dtype=ms.float16
        ... )

        >>> pipe = KandinskyV22ControlnetImg2ImgPipeline.from_pretrained(
        ...     "kandinsky-community/kandinsky-2-2-controlnet-depth", mindspore_dtype=ms.float16
        ... )

        >>> img = load_image(
        ...     "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
        ...     "/kandinsky/cat.png"
        ... ).resize((768, 768))


        >>> hint = make_hint(img, depth_estimator).unsqueeze(0).half()

        >>> prompt = "A robot, 4k photo"
        >>> negative_prior_prompt = "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts,
        ...     ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands,
        ...     poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs,
        ...     cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms,
        ...     extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"

        >>> generator = np.random.Generator(np.random.PCG64(43))

        >>> img_emb = pipe_prior(prompt=prompt, image=img, strength=0.85, generator=generator)
        >>> negative_emb = pipe_prior(prompt=negative_prior_prompt, image=img, strength=1, generator=generator)

        >>> images = pipe(
        ...     image=img,
        ...     strength=0.5,
        ...     image_embeds=img_emb[0],
        ...     negative_image_embeds=negative_emb[0],
        ...     hint=hint,
        ...     num_inference_steps=50,
        ...     generator=generator,
        ...     height=768,
        ...     width=768,
        ... )[0]

        >>> images[0].save("robot_cat.png")
        ```
"""


# Copied from diffusers.pipelines.kandinsky2_2.pipeline_kandinsky2_2.downscale_height_and_width
def downscale_height_and_width(height, width, scale_factor=8):
    new_height = height // scale_factor**2
    if height % scale_factor**2 != 0:
        new_height += 1
    new_width = width // scale_factor**2
    if width % scale_factor**2 != 0:
        new_width += 1
    return new_height * scale_factor, new_width * scale_factor


# Copied from diffusers.pipelines.kandinsky.pipeline_kandinsky_img2img.prepare_image
def prepare_image(pil_image, w=512, h=512):
    pil_image = pil_image.resize((w, h), resample=Image.BICUBIC, reducing_gap=1)
    arr = np.array(pil_image.convert("RGB"))
    arr = arr.astype(np.float32) / 127.5 - 1
    arr = np.transpose(arr, [2, 0, 1])
    image = ms.Tensor.from_numpy(arr).unsqueeze(0)
    return image


class KandinskyV22ControlnetImg2ImgPipeline(DiffusionPipeline):
    """
    Pipeline for image-to-image generation using Kandinsky

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        scheduler ([`DDIMScheduler`]):
            A scheduler to be used in combination with `unet` to generate image latents.
        unet ([`UNet2DConditionModel`]):
            Conditional U-Net architecture to denoise the image embedding.
        movq ([`VQModel`]):
            MoVQ Decoder to generate the image from the latents.
    """

    model_cpu_offload_seq = "unet->movq"

    def __init__(
        self,
        unet: UNet2DConditionModel,
        scheduler: DDPMScheduler,
        movq: VQModel,
    ):
        super().__init__()

        self.register_modules(
            unet=unet,
            scheduler=scheduler,
            movq=movq,
        )
        self.movq_scale_factor = 2 ** (len(self.movq.config.block_out_channels) - 1)

    # Copied from diffusers.pipelines.kandinsky.pipeline_kandinsky_img2img.KandinskyImg2ImgPipeline.get_timesteps
    def get_timesteps(self, num_inference_steps, strength):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start

    # Copied from diffusers.pipelines.kandinsky2_2.pipeline_kandinsky2_2_img2img.KandinskyV22Img2ImgPipeline.prepare_latents
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
                init_latents = ops.cat(init_latents, axis=0)
            else:
                init_latents = self.movq.diag_gauss_dist.sample(self.movq.encode(image)[0])

            init_latents = self.movq.config.scaling_factor * init_latents

        init_latents = ops.cat([init_latents], axis=0)

        shape = init_latents.shape
        noise = randn_tensor(shape, generator=generator, dtype=dtype)

        # get latents
        init_latents = self.scheduler.add_noise(init_latents, noise, timestep)

        latents = init_latents

        return latents

    def __call__(
        self,
        image_embeds: Union[ms.Tensor, List[ms.Tensor]],
        image: Union[ms.Tensor, PIL.Image.Image, List[ms.Tensor], List[PIL.Image.Image]],
        negative_image_embeds: Union[ms.Tensor, List[ms.Tensor]],
        hint: ms.Tensor,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 100,
        guidance_scale: float = 4.0,
        strength: float = 0.3,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[np.random.Generator, List[np.random.Generator]]] = None,
        output_type: Optional[str] = "pil",
        callback: Optional[Callable[[int, int, ms.Tensor], None]] = None,
        callback_steps: int = 1,
        return_dict: bool = False,
    ):
        """
        Function invoked when calling the pipeline for generation.

        Args:
            image_embeds (`ms.Tensor` or `List[ms.Tensor]`):
                The clip image embeddings for text prompt, that will be used to condition the image generation.
            image (`ms.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[ms.Tensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process. Can also accept image latents as `image`, if passing latents directly, it will not be encoded
                again.
            strength (`float`, *optional*, defaults to 0.8):
                Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1. `image`
                will be used as a starting point, adding more noise to it the larger the `strength`. The number of
                denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise will
                be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `image`.
            hint (`ms.Tensor`):
                The controlnet condition.
            negative_image_embeds (`ms.Tensor` or `List[ms.Tensor]`):
                The clip image embeddings for negative text prompt, will be used to condition the image generation.
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`np.random.Generator` or `List[np.random.Generator]`, *optional*):
                One or a list of [np.random.Generator(s)](https://numpy.org/doc/stable/reference/random/generator.html)
                to make generation deterministic.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between: `"pil"` (`PIL.Image.Image`), `"np"`
                (`np.array`) or `"ms"` (`ms.Tensor`).
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
        do_classifier_free_guidance = guidance_scale > 1.0

        if isinstance(image_embeds, list):
            image_embeds = ops.cat(image_embeds, axis=0)
        if isinstance(negative_image_embeds, list):
            negative_image_embeds = ops.cat(negative_image_embeds, axis=0)
        if isinstance(hint, list):
            hint = ops.cat(hint, axis=0)

        batch_size = image_embeds.shape[0]

        if do_classifier_free_guidance:
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            negative_image_embeds = negative_image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            hint = hint.repeat_interleave(num_images_per_prompt, dim=0)

            image_embeds = ops.cat([negative_image_embeds, image_embeds], axis=0).to(dtype=self.unet.dtype)
            hint = ops.cat([hint, hint], axis=0).to(dtype=self.unet.dtype)

        if not isinstance(image, list):
            image = [image]
        if not all(isinstance(i, (PIL.Image.Image, ms.Tensor)) for i in image):
            raise ValueError(
                f"Input is in incorrect format: {[type(i) for i in image]}. Currently, we only support  PIL image and mindspore tensor"
            )

        image = ops.cat([prepare_image(i, width, height) for i in image], axis=0)
        image = image.to(dtype=image_embeds.dtype)

        latents = self.movq.encode(image)[0]
        latents = latents.repeat_interleave(num_images_per_prompt, dim=0)
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength)
        latent_timestep = timesteps[:1].tile((batch_size * num_images_per_prompt,))
        height, width = downscale_height_and_width(height, width, self.movq_scale_factor)
        latents = self.prepare_latents(
            latents, latent_timestep, batch_size, num_images_per_prompt, image_embeds.dtype, generator
        )
        for i, t in enumerate(self.progress_bar(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = ops.cat([latents] * 2) if do_classifier_free_guidance else latents

            added_cond_kwargs = {"image_embeds": image_embeds, "hint": hint}
            noise_pred = self.unet(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=None,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            if do_classifier_free_guidance:
                noise_pred, variance_pred = noise_pred.split(latents.shape[1], axis=1)
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                _, variance_pred_text = variance_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                noise_pred = ops.cat([noise_pred, variance_pred_text], axis=1)

            if not (
                hasattr(self.scheduler.config, "variance_type")
                and self.scheduler.config.variance_type in ["learned", "learned_range"]
            ):
                noise_pred, _ = noise_pred.split(latents.shape[1], axis=1)

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
            raise ValueError(f"Only the output types `ms`, `pil` and `np` are supported not output_type={output_type}")

        if output_type in ["np", "pil"]:
            image = image * 0.5 + 0.5
            image = image.clamp(0, 1)
            image = image.permute(0, 2, 3, 1).float().numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
