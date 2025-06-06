# Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
# William Peebles and Saining Xie
#
# Copyright (c) 2021 OpenAI
# MIT License
#
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

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

import mindspore as ms
from mindspore import mint, ops

from ...models import AutoencoderKL, DiTTransformer2DModel
from ...schedulers import KarrasDiffusionSchedulers
from ...utils.mindspore_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput


class DiTPipeline(DiffusionPipeline):
    r"""
    Pipeline for image generation based on a Transformer backbone instead of a UNet.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        transformer ([`DiTTransformer2DModel`]):
            A class conditioned `DiTTransformer2DModel` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        scheduler ([`DDIMScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
    """

    model_cpu_offload_seq = "transformer->vae"

    def __init__(
        self,
        transformer: DiTTransformer2DModel,
        vae: AutoencoderKL,
        scheduler: KarrasDiffusionSchedulers,
        id2label: Optional[Dict[int, str]] = None,
    ):
        super().__init__()
        self.register_modules(transformer=transformer, vae=vae, scheduler=scheduler)

        # create a imagenet -> id dictionary for easier use
        self.labels = {}
        if id2label is not None:
            for key, value in id2label.items():
                for label in value.split(","):
                    self.labels[label.lstrip().rstrip()] = int(key)
            self.labels = dict(sorted(self.labels.items()))

    def get_label_ids(self, label: Union[str, List[str]]) -> List[int]:
        r"""

        Map label strings from ImageNet to corresponding class ids.

        Parameters:
            label (`str` or `dict` of `str`):
                Label strings to be mapped to class ids.

        Returns:
            `list` of `int`:
                Class ids to be processed by pipeline.
        """

        if not isinstance(label, list):
            label = list(label)

        for i in label:
            if i not in self.labels:
                raise ValueError(
                    f"{i} does not exist. Please make sure to select one of the following labels: \n {self.labels}."
                )

        return [self.labels[i] for i in label]

    def __call__(
        self,
        class_labels: List[int],
        guidance_scale: float = 4.0,
        generator: Optional[Union[np.random.Generator, List[np.random.Generator]]] = None,
        num_inference_steps: int = 50,
        output_type: Optional[str] = "pil",
        return_dict: bool = False,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            class_labels (List[int]):
                List of ImageNet class labels for the images to be generated.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            generator (`np.random.Generator`, *optional*):
                A [`np.random.Generator`](https://numpy.org/doc/stable/reference/random/generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 250):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`ImagePipelineOutput`] instead of a plain tuple.

        Examples:

        ```py
        >>> from mindone.diffusers import DiTPipeline, DPMSolverMultistepScheduler
        >>> import mindspore as ms

        >>> import numpy as np

        >>> pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", mindspore_dtype=ms.float16)
        >>> pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

        >>> # pick words from Imagenet class labels
        >>> pipe.labels  # to print all available words

        >>> # pick words that exist in ImageNet
        >>> words = ["white shark", "umbrella"]

        >>> class_ids = pipe.get_label_ids(words)

        >>> generator = np.random.default_rng(33)
        >>> output = pipe(class_labels=class_ids, num_inference_steps=25, generator=generator)

        >>> image = output[0][0]  # label 'white shark'
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """

        batch_size = len(class_labels)
        latent_size = self.transformer.config.sample_size
        latent_channels = self.transformer.config.in_channels

        latents = randn_tensor(
            shape=(batch_size, latent_channels, latent_size, latent_size),
            generator=generator,
            dtype=self.transformer.dtype,
        )
        latent_model_input = mint.cat([latents] * 2) if guidance_scale > 1 else latents

        class_labels = mint.reshape(ms.Tensor(class_labels), (-1,))
        class_null = ms.Tensor([1000] * batch_size)
        class_labels_input = mint.cat([class_labels, class_null], 0) if guidance_scale > 1 else class_labels

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        for t in self.progress_bar(self.scheduler.timesteps):
            if guidance_scale > 1:
                half = latent_model_input[: len(latent_model_input) // 2]
                latent_model_input = mint.cat([half, half], dim=0)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            timesteps = t
            # todo: unavailable mint interface
            if not ops.is_tensor(timesteps):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                # This would be a good case for the `match` statement (Python 3.10+)
                is_mps = False
                if isinstance(timesteps, float):
                    dtype = ms.float32 if is_mps else ms.float64
                else:
                    dtype = ms.int32 if is_mps else ms.int64
                timesteps = ms.Tensor([timesteps], dtype=dtype)
            elif len(timesteps.shape) == 0:
                timesteps = timesteps[None]
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timesteps = timesteps.broadcast_to((latent_model_input.shape[0],))
            # predict noise model_output
            noise_pred = self.transformer(latent_model_input, timestep=timesteps, class_labels=class_labels_input)[0]

            # perform guidance
            if guidance_scale > 1:
                eps, rest = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
                cond_eps, uncond_eps = mint.split(eps, len(eps) // 2, dim=0)

                half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
                eps = mint.cat([half_eps, half_eps], dim=0)

                noise_pred = mint.cat([eps, rest], dim=1)

            # learned sigma
            if self.transformer.config.out_channels // 2 == latent_channels:
                model_output, _ = mint.split(noise_pred, latent_channels, dim=1)
            else:
                model_output = noise_pred

            # compute previous image: x_t -> x_t-1
            latent_model_input = self.scheduler.step(model_output, t, latent_model_input)[0]

        if guidance_scale > 1:
            latents, _ = latent_model_input.chunk(2, dim=0)
        else:
            latents = latent_model_input

        latents = 1 / self.vae.config.scaling_factor * latents
        samples = self.vae.decode(latents)[0]

        samples = (samples / 2 + 0.5).clamp(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        samples = samples.permute(0, 2, 3, 1).float().asnumpy()

        if output_type == "pil":
            samples = self.numpy_to_pil(samples)

        if not return_dict:
            return (samples,)

        return ImagePipelineOutput(images=samples)
