# Copyright 2025 Open AI and The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import PIL.Image
from transformers import CLIPImageProcessor

import mindspore as ms
from mindspore import mint

from ....transformers import CLIPVisionModel
from ...models import PriorTransformer
from ...schedulers import HeunDiscreteScheduler
from ...utils import BaseOutput, logging
from ...utils.mindspore_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .renderer import ShapERenderer

XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from PIL import Image
        >>> import mindspore as ms
        >>> from mindone.diffusers import DiffusionPipeline
        >>> from mindone.diffusers.utils import export_to_gif, load_image

        >>> repo = "openai/shap-e-img2img"
        >>> pipe = DiffusionPipeline.from_pretrained(repo, mindspore_dtype=ms.float16)

        >>> guidance_scale = 3.0
        >>> image_url = "https://hf.co/datasets/diffusers/docs-images/resolve/main/shap-e/corgi.png"
        >>> image = load_image(image_url).convert("RGB")

        >>> images = pipe(
        ...     image,
        ...     guidance_scale=guidance_scale,
        ...     num_inference_steps=64,
        ...     frame_size=256,
        ... )[0]

        >>> gif_path = export_to_gif(images[0], "corgi_3d.gif")
        ```
"""


@dataclass
class ShapEPipelineOutput(BaseOutput):
    """
    Output class for [`ShapEPipeline`] and [`ShapEImg2ImgPipeline`].

    Args:
        images (`ms.Tensor`)
            A list of images for 3D rendering.
    """

    images: Union[PIL.Image.Image, np.ndarray]


class ShapEImg2ImgPipeline(DiffusionPipeline):
    """
    Pipeline for generating latent representation of a 3D asset and rendering with the NeRF method from an image.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        prior ([`PriorTransformer`]):
            The canonincal unCLIP prior to approximate the image embedding from the text embedding.
        image_encoder ([`~transformers.CLIPVisionModel`]):
            Frozen image-encoder.
        image_processor ([`~transformers.CLIPImageProcessor`]):
             A `CLIPImageProcessor` to process images.
        scheduler ([`HeunDiscreteScheduler`]):
            A scheduler to be used in combination with the `prior` model to generate image embedding.
        shap_e_renderer ([`ShapERenderer`]):
            Shap-E renderer projects the generated latents into parameters of a MLP to create 3D objects with the NeRF
            rendering method.
    """

    model_cpu_offload_seq = "image_encoder->prior"
    _exclude_from_cpu_offload = ["shap_e_renderer"]

    def __init__(
        self,
        prior: PriorTransformer,
        image_encoder: CLIPVisionModel,
        image_processor: CLIPImageProcessor,
        scheduler: HeunDiscreteScheduler,
        shap_e_renderer: ShapERenderer,
    ):
        super().__init__()

        self.register_modules(
            prior=prior,
            image_encoder=image_encoder,
            image_processor=image_processor,
            scheduler=scheduler,
            shap_e_renderer=shap_e_renderer,
        )

    # Copied from diffusers.pipelines.unclip.pipeline_unclip.UnCLIPPipeline.prepare_latents
    def prepare_latents(self, shape, dtype, generator, latents, scheduler):
        if latents is None:
            latents = randn_tensor(shape, generator=generator, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")

        latents = latents * scheduler.init_noise_sigma.to(dtype)
        return latents

    def _encode_image(
        self,
        image,
        num_images_per_prompt,
        do_classifier_free_guidance,
    ):
        if isinstance(image, List) and isinstance(image[0], ms.Tensor):
            image = mint.cat(image, dim=0) if image[0].ndim == 4 else mint.stack(image, dim=0)

        if not isinstance(image, ms.Tensor):
            image = self.image_processor(image, return_tensors="np").pixel_values[0]
            image = ms.tensor(image).unsqueeze(0)

        image = image.to(dtype=self.image_encoder.dtype)

        image_embeds = self.image_encoder(image)[0]
        image_embeds = image_embeds[:, 1:, :]  # batch_size, dim, 256

        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)

        if do_classifier_free_guidance:
            negative_image_embeds = mint.zeros_like(image_embeds)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeds = mint.cat([negative_image_embeds, image_embeds])

        return image_embeds

    def __call__(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image]],
        num_images_per_prompt: int = 1,
        num_inference_steps: int = 25,
        generator: Optional[Union[np.random.Generator, List[np.random.Generator]]] = None,
        latents: Optional[ms.Tensor] = None,
        guidance_scale: float = 4.0,
        frame_size: int = 64,
        output_type: Optional[str] = "pil",  # pil, np, latent, mesh
        return_dict: bool = False,
    ):
        """
        The call function to the pipeline for generation.

        Args:
            image (`ms.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[ms.Tensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image` or tensor representing an image batch to be used as the starting point. Can also accept image
                latents as image, but if passing latents directly it is not encoded again.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            num_inference_steps (`int`, *optional*, defaults to 25):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            generator (`np.random.Generator` or `List[np.random.Generator]`, *optional*):
                A [`np.random.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`ms.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            frame_size (`int`, *optional*, default to 64):
                The width and height of each image frame of the generated 3D output.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `"pil"` (`PIL.Image.Image`), `"np"`
                (`np.array`), `"latent"` (`ms.Tensor`), or mesh ([`MeshDecoderOutput`]).
            return_dict (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~pipelines.shap_e.pipeline_shap_e.ShapEPipelineOutput`] instead of a plain
                tuple.

        Examples:

        Returns:
            [`~pipelines.shap_e.pipeline_shap_e.ShapEPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.shap_e.pipeline_shap_e.ShapEPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images.
        """

        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, ms.Tensor):
            batch_size = image.shape[0]
        elif isinstance(image, list) and isinstance(image[0], (ms.Tensor, PIL.Image.Image)):
            batch_size = len(image)
        else:
            raise ValueError(
                f"`image` has to be of type `PIL.Image.Image`, `ms.Tensor`, `List[PIL.Image.Image]` or `List[ms.Tensor]` but is {type(image)}"
            )

        batch_size = batch_size * num_images_per_prompt

        do_classifier_free_guidance = guidance_scale > 1.0
        image_embeds = self._encode_image(image, num_images_per_prompt, do_classifier_free_guidance)

        # prior

        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        num_embeddings = self.prior.config.num_embeddings
        embedding_dim = self.prior.config.embedding_dim
        if latents is None:
            latents = self.prepare_latents(
                (batch_size, num_embeddings * embedding_dim),
                image_embeds.dtype,
                generator,
                latents,
                self.scheduler,
            )

        # YiYi notes: for testing only to match ldm, we can directly create a latents with desired shape: batch_size, num_embeddings, embedding_dim
        latents = latents.reshape(latents.shape[0], num_embeddings, embedding_dim)

        for i, t in enumerate(self.progress_bar(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = mint.cat([latents] * 2) if do_classifier_free_guidance else latents
            # TODO: method of scheduler should not change the dtype of input.
            #  Remove the casting after cuiyushi confirm that.
            tmp_dtype = latent_model_input.dtype
            scaled_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            scaled_model_input = scaled_model_input.to(tmp_dtype)

            noise_pred = self.prior(
                scaled_model_input,
                timestep=t,
                proj_embedding=image_embeds,
            )[0]

            # remove the variance
            noise_pred, _ = noise_pred.split(
                scaled_model_input.shape[2], dim=2
            )  # batch_size, num_embeddings, embedding_dim

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

            # TODO: method of scheduler should not change the dtype of input.
            #  Remove the casting after cuiyushi confirm that.
            tmp_dtype = latents.dtype
            latents = self.scheduler.step(
                noise_pred,
                timestep=t,
                sample=latents,
            )[0]
            latents = latents.to(tmp_dtype)

        if output_type not in ["np", "pil", "latent", "mesh"]:
            raise ValueError(
                f"Only the output types `pil`, `np`, `latent` and `mesh` are supported not output_type={output_type}"
            )

        if output_type == "latent":
            return ShapEPipelineOutput(images=latents)

        images = []
        if output_type == "mesh":
            for i, latent in enumerate(latents):
                mesh = self.shap_e_renderer.decode_to_mesh(
                    latent[None, :],
                )
                images.append(mesh)

        else:
            # np, pil
            for i, latent in enumerate(latents):
                image = self.shap_e_renderer.decode_to_image(
                    latent[None, :],
                    size=frame_size,
                )
                images.append(image)

            images = mint.stack(images)

            images = images.numpy()

            if output_type == "pil":
                images = [self.numpy_to_pil(image) for image in images]

        if not return_dict:
            return (images,)

        return ShapEPipelineOutput(images=images)
