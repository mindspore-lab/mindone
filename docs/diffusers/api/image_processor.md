<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# VAE Image Processor

The [`VaeImageProcessor`](image_processor.md#mindone.diffusers.image_processor.VaeImageProcessor) provides a unified API for [`StableDiffusionPipeline`](pipelines/stable_diffusion/text2img.md#mindone.diffusers.StableDiffusionPipeline) to prepare image inputs for VAE encoding and post-processing outputs once they're decoded. This includes transformations such as resizing, normalization, and conversion between PIL Image, MindSpore, and NumPy arrays.

All pipelines with [`VaeImageProcessor`](image_processor.md#mindone.diffusers.image_processor.VaeImageProcessor) accept PIL Image, MindSpore tensor, or NumPy arrays as image inputs and return outputs based on the `output_type` argument by the user. You can pass encoded image latents directly to the pipeline and return latents from the pipeline as a specific output with the `output_type` argument (for example `output_type="latent"`). This allows you to take the generated latents from one pipeline and pass it to another pipeline as input without leaving the latent space. It also makes it much easier to use multiple pipelines together by passing MindSpore tensors directly between different pipelines.

::: mindone.diffusers.image_processor.VaeImageProcessor

::: mindone.diffusers.image_processor.PixArtImageProcessor

::: mindone.diffusers.image_processor.IPAdapterMaskProcessor
