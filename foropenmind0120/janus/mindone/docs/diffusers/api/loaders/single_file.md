<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Single files

The [`from_single_file`](single_file.md#mindone.diffusers.loaders.single_file.FromSingleFileMixin) method allows you to load:

* a model stored in a single file, which is useful if you're working with models from the diffusion ecosystem, like Automatic1111, and commonly rely on a single-file layout to store and share models
* a model stored in their originally distributed layout, which is useful if you're working with models finetuned with other services, and want to load it directly into Diffusers model objects and pipelines

!!! tip

    Read the [Model files and layouts](../../using-diffusers/other-formats.md) guide to learn more about the Diffusers-multifolder layout versus the single-file layout, and how to load models stored in these different layouts.

## Supported pipelines

- [`CogVideoXPipeline`](../pipelines/cogvideox.md)
- [`CogVideoXImageToVideoPipeline`](../pipelines/cogvideox.md)
- [`CogVideoXVideoToVideoPipeline`](../pipelines/cogvideox.md)
- [`StableDiffusionPipeline`](../pipelines/stable_diffusion/text2img.md)
- [`StableDiffusionImg2ImgPipeline`](../pipelines/stable_diffusion/text2img.md)
- [`StableDiffusionInpaintPipeline`](../pipelines/stable_diffusion/text2img.md)
- [`StableDiffusionControlNetPipeline`](../pipelines/controlnet.md)
- [`StableDiffusionControlNetImg2ImgPipeline`](../pipelines/controlnet.md)
- [`StableDiffusionControlNetInpaintPipeline`](../pipelines/controlnet.md)
- [`StableDiffusionUpscalePipeline`](../pipelines/stable_diffusion/stable_diffusion_2.md)
- [`StableDiffusionXLPipeline`](../pipelines/stable_diffusion/stable_diffusion_xl.md#stable-diffusion-xl)
- [`StableDiffusionXLImg2ImgPipeline`](../pipelines/stable_diffusion/stable_diffusion_xl.md#stable-diffusion-xl)
- [`StableDiffusionXLInpaintPipeline`](../pipelines/stable_diffusion/stable_diffusion_xl.md#stable-diffusion-xl)
- [`StableDiffusionXLInstructPix2PixPipeline`](../pipelines/pix2pix.md#instructpix2pix)
- [`StableDiffusionXLControlNetPipeline`](../pipelines/controlnet_sdxl.md)
- [`StableDiffusion3Pipeline`](../pipelines/stable_diffusion/stable_diffusion_3.md)
- [`LatentConsistencyModelPipeline`](../pipelines/latent_consistency_models.md)
- [`LatentConsistencyModelImg2ImgPipeline`](../pipelines/latent_consistency_models.md)
- [`StableDiffusionControlNetXSPipeline`](../pipelines/controlnetxs.md)
- [`StableDiffusionXLControlNetXSPipeline`](../pipelines/controlnetxs_sdxl.md)

## Supported models

- [`UNet2DConditionModel`](../models/unet2d-cond.md)
- [`StableCascadeUNet`](../models/stable_cascade_unet.md)
- [`AutoencoderKL`](../models/autoencoderkl.md)
- [`AutoencoderKLCogVideoX`](../models/autoencoderkl_cogvideox.md)
- [`ControlNetModel`](../models/controlnet.md)
- [`SD3Transformer2DModel`](../models/sd3_transformer2d.md)
- [`FluxTransformer2DModel`](../models/flux_transformer.md)

::: mindone.diffusers.loaders.single_file.FromSingleFileMixin
