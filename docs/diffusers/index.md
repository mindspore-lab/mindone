<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

<div align="center" markdown>

# Make 🤗 D🧨ffusers Run on MindSpore

</div>

> State-of-the-art diffusion models for image and audio generation in MindSpore.
> We've tried to provide a completely consistent interface and usage with the [huggingface/diffusers](https://github.com/huggingface/diffusers).
> Only necessary changes are made to the [huggingface/diffusers](https://github.com/huggingface/diffusers) to make it seamless for users from torch.

???+ info

    Due to differences in framework, some APIs will not be identical to [huggingface/diffusers](https://github.com/huggingface/diffusers) in the foreseeable future, see [Limitations](./limitations.md) for details.


# Diffusers

Diffusers is a library of state-of-the-art pretrained diffusion models for generating videos, images, and audio.

The library revolves around the [`DiffusionPipeline`], an API designed for:

- easy inference with only a few lines of code
- flexibility to mix-and-match pipeline components (models, schedulers)
- loading and using adapters like LoRA

Diffusers also comes with optimizations - such as offloading and quantization - to ensure even the largest models are accessible on memory-constrained devices. If memory is not an issue, Diffusers supports torch.compile to boost inference speed.

Get started right away with a Diffusers model on the [Hub](https://huggingface.co/models?library=diffusers&sort=trending) today!

## Learn

If you're a beginner, we recommend starting with the [Hugging Face Diffusion Models Course](https://huggingface.co/learn/diffusion-course/unit0/1). You'll learn the theory behind diffusion models, and learn how to use the Diffusers library to generate images, fine-tune your own models, and more.
