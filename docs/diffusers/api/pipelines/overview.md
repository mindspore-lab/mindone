<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Pipelines

Pipelines provide a simple way to run state-of-the-art diffusion models in inference by bundling all of the necessary components (multiple independently-trained models, schedulers, and processors) into a single end-to-end class. Pipelines are flexible and they can be adapted to use different schedulers or even model components.

All pipelines are built from the base [`DiffusionPipeline`](overview.md#mindone.diffusers.DiffusionPipeline) class which provides basic functionality for loading, downloading, and saving all the components. Specific pipeline types (for example [`StableDiffusionPipeline`](stable_diffusion/text2img.md)) loaded with [`DiffusionPipeline.from_pretrained`](overview.md#mindone.diffusers.DiffusionPipeline) are automatically detected and the pipeline components are loaded and passed to the `__init__` function of the pipeline.

!!! warning

	You shouldn't use the [`DiffusionPipeline`](overview.md#mindone.diffusers.DiffusionPipeline) class for training. Individual components (for example, [`UNet2DModel`](../models/unet2d.md) and [`UNet2DConditionModel`](../models/unet2d-cond.md)) of diffusion pipelines are usually trained individually, so we suggest directly working with them instead.

!!! warning

	Pipelines do not offer any training functionality. You'll notice MindSpore's autograd is disabled by decorating the [`DiffusionPipeline.__call__`](overview.md#mindone.diffusers.DiffusionPipeline) method with a [`mindspore._no_grad`] decorator because pipelines should not be used for training. If you're interested in training, please take a look at the [Training](../../training/overview.md) guides instead!

The table below lists all the pipelines currently available in 🤗 Diffusers and the tasks they support. Click on a pipeline to view its abstract and published paper.

| Pipeline                                                       | Tasks |
|----------------------------------------------------------------|---|
| [AnimateDiff](animatediff.md)                                  | text2video |
| [AuraFlow](auraflow) 											 | text2image |
| [BLIP Diffusion](blip_diffusion.md)                            | text2image |
| [CogVideoX](cogvideox) 										 | text2video |
| [Consistency Models](consistency_models.md)                    | unconditional image generation |
| [ControlNet](controlnet.md)                                    | text2image, image2image, inpainting |
| [ControlNet with Flux.1](controlnet_flux) 					 | text2image |
| [ControlNet with Hunyuan-DiT](controlnet_hunyuandit) 			 | text2image |
| [ControlNet with Stable Diffusion 3](controlnet_sd3.md)        | text2image |
| [ControlNet with Stable Diffusion XL](controlnet_sdxl.md)      | text2image |
| [ControlNet-XS](controlnetxs.md)                               | text2image |
| [ControlNet-XS with Stable Diffusion XL](controlnetxs_sdxl.md) | text2image |
| [Dance Diffusion](dance_diffusion.md)                          | unconditional audio generation |
| [DDIM](ddim.md)                                                | unconditional image generation |
| [DDPM](ddpm.md)                                                | unconditional image generation |
| [DeepFloyd IF](deepfloyd_if.md)                                | text2image, image2image, inpainting, super-resolution |
| [DiffEdit](diffedit.md)                                        | inpainting |
| [DiT](dit.md)                                                  | text2image |
| [Flux](flux) 													 | text2image |
| [Hunyuan-DiT](hunyuandit.md)                                   | text2image |
| [I2VGen-XL](i2vgenxl.md)                                       | text2video |
| [InstructPix2Pix](pix2pix.md)                                  | image editing |
| [Kandinsky 2.1](kandinsky.md)                                  | text2image, image2image, inpainting, interpolation |
| [Kandinsky 2.2](kandinsky_v22.md)                              | text2image, image2image, inpainting |
| [Kandinsky 3](kandinsky3.md)                                   | text2image, image2image |
| [Kolors](kolors) 												 | text2image |
| [Latent Consistency Models](latent_consistency_models.md)      | text2image |
| [Latent Diffusion](latent_diffusion.md)                        | text2image, super-resolution |
| [Latte](latte) 												 | text2image |
| [Lumina-T2X](lumina) 											 | text2image |
| [Marigold](marigold.md)                                        | depth |
| [PAG](pag) 													 | text2image |
| [PixArt-α](pixart.md)                                          | text2image |
| [PixArt-Σ](pixart_sigma.md)                                    | text2image |
| [Shap-E](shap_e.md)                                            | text-to-3D, image-to-3D |
| [Stable Cascade](stable_cascade.md)                            | text2image |
| [Stable Diffusion](stable_diffusion/overview) 				 | text2image, image2image, depth2image, inpainting, image variation, latent upscaler, super-resolution |
| [Stable Diffusion XL](stable_diffusion/stable_diffusion_xl) 	 | text2image, image2image, inpainting |
| [T2I-Adapter](stable_diffusion/adapter) 						 | text2image |
| [unCLIP](unclip.md)                                            | text2image, image variation |
| [Wuerstchen](wuerstchen.md)                                    | text2image |

::: mindone.diffusers.DiffusionPipeline

::: mindone.diffusers.utils.PushToHubMixin
