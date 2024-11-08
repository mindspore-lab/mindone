<!--Copyright 2024 The HuggingFace Team. All rights reserved.

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


🤗 Diffusers is the go-to library for state-of-the-art pretrained diffusion models for generating images, audio, and even 3D structures of molecules. Whether you're looking for a simple inference solution or want to train your own diffusion model, 🤗 Diffusers is a modular toolbox that supports both. Our library is designed with a focus on [usability over performance](conceptual/philosophy.md#usability-over-performance), [simple over easy](conceptual/philosophy.md#simple-over-easy), and [customizability over abstractions](conceptual/philosophy.md#tweakable-contributor-friendly-over-abstraction).

The library has three main components:

- State-of-the-art diffusion pipelines for inference with just a few lines of code. There are many pipelines in 🤗 Diffusers, check out the table in the pipeline [overview](api/pipelines/overview.md) for a complete list of available pipelines and the task they solve.
- Interchangeable [noise schedulers](api/schedulers/overview.md) for balancing trade-offs between generation speed and quality.
- Pretrained [models](api/models.md) that can be used as building blocks, and combined with schedulers, for creating your own end-to-end diffusion systems.


<div class="grid cards" markdown>

-   __[Tutorials](./tutorials/tutorial_overview.md)__

    ---

    Learn the fundamental skills you need to start generating outputs, build your own diffusion system, and train a diffusion model. We recommend starting here if you're using 🤗 Diffusers for the first time!

-   __[How-to guides](./using-diffusers/loading_overview.md)__

    ---

    Practical guides for helping you load pipelines, models, and schedulers. You'll also learn how to use pipelines for specific tasks, control how outputs are generated, optimize for inference speed, and different training techniques.

-   __[Conceptual guides](./conceptual/philosophy.md)__

    ---

    Understand why the library was designed the way it was, and learn more about the ethical guidelines and safety implementations for using the library.

-   __[Reference](./api/models/overview.md)__

    ---

    Technical descriptions of how 🤗 Diffusers classes and methods work.

</div>
