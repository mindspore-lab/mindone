<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# LoRA

LoRA is a fast and lightweight training method that inserts and trains a significantly smaller number of parameters instead of all the model parameters. This produces a smaller file (~100 MBs) and makes it easier to quickly train a model to learn a new concept. LoRA weights are typically loaded into the denoiser, text encoder or both. The denoiser usually corresponds to a UNet (`UNet2DConditionModel`, for example) or a Transformer (`SD3Transformer2DModel`, for example). There are several classes for loading LoRA weights:

- `StableDiffusionLoraLoaderMixin` provides functions for loading and unloading, fusing and unfusing, enabling and disabling, and more functions for managing LoRA weights. This class can be used with any model.
- `StableDiffusionXLLoraLoaderMixin` is a [Stable Diffusion (SDXL)](../../api/pipelines/stable_diffusion/stable_diffusion_xl.md) version of the `StableDiffusionLoraLoaderMixin` class for loading and saving LoRA weights. It can only be used with the SDXL model.
- `SD3LoraLoaderMixin` provides similar functions for [Stable Diffusion 3](../../api/pipelines/stable_diffusion/stable_diffusion_3.md).
- `FluxLoraLoaderMixin` provides similar functions for [Flux](../../api/pipelines/flux.md).
- `CogVideoXLoraLoaderMixin` provides similar functions for [CogVideoX](../../api/pipelines/cogvideox.md).
- `Mochi1LoraLoaderMixin` provides similar functions for [Mochi](../../api/pipelines/mochi.md).
- `LTXVideoLoraLoaderMixin` provides similar functions for [LTX-Video](../../api/pipelines/ltx_video.md).
- `SanaLoraLoaderMixin` provides similar functions for [Sana](../../api/pipelines/sana.md).
- `HunyuanVideoLoraLoaderMixin` provides similar functions for [HunyuanVideo](../../api/pipelines/hunyuan_video.md).
- `Lumina2LoraLoaderMixin` provides similar functions for [Lumina2](../../api/pipelines/lumina2.md).
- `WanLoraLoaderMixin` provides similar functions for [Wan](../../api/pipelines/wan.md).
- `SkyReelsV2LoraLoaderMixin` provides similar functions for [SkyReels-V2](../../api/pipelines/skyreels_v2.md).
- `AmusedLoraLoaderMixin` is for the [AmusedPipeline](../../api/pipelines/amused.md).
- `LoraBaseMixin` provides a base class with several utility methods to fuse, unfuse, unload, LoRAs and more.

!!! tip

    To learn more about how to load LoRA weights, see the [LoRA](../../using-diffusers/loading_adapters.md#lora) loading guide.


::: mindone.diffusers.loaders.lora_pipeline.StableDiffusionLoraLoaderMixin

::: mindone.diffusers.loaders.lora_pipeline.StableDiffusionXLLoraLoaderMixin

::: mindone.diffusers.loaders.lora_pipeline.SD3LoraLoaderMixin

::: mindone.diffusers.loaders.lora_pipeline.FluxLoraLoaderMixin

::: mindone.diffusers.loaders.lora_pipeline.CogVideoXLoraLoaderMixin

::: mindone.diffusers.loaders.lora_pipeline.Mochi1LoraLoaderMixin

::: mindone.diffusers.loaders.lora_pipeline.LTXVideoLoraLoaderMixin

::: mindone.diffusers.loaders.lora_pipeline.SanaLoraLoaderMixin

::: mindone.diffusers.loaders.lora_pipeline.HunyuanVideoLoraLoaderMixin

::: mindone.diffusers.loaders.lora_pipeline.Lumina2LoraLoaderMixin

::: mindone.diffusers.loaders.lora_pipeline.WanLoraLoaderMixin

::: mindone.diffusers.loaders.lora_pipeline.SkyReelsV2LoraLoaderMixin

::: mindone.diffusers.loaders.lora_pipeline.AmusedLoraLoaderMixin

::: mindone.diffusers.loaders.lora_base.LoraBaseMixin
