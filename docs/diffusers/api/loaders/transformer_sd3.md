<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# SD3Transformer2D

This class is useful when *only* loading weights into a [`SD3Transformer2DModel`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/models/sd3_transformer2d/#mindone.diffusers.SD3Transformer2DModel). If you need to load weights into the text encoder or a text encoder and SD3Transformer2DModel, check [`SD3LoraLoaderMixin`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/loaders/lora/#mindone.diffusers.loaders.lora_pipeline.SD3LoraLoaderMixin) class instead.

The [`SD3Transformer2DLoadersMixin`] class currently only loads IP-Adapter weights, but will be used in the future to save weights and load LoRAs.

!!! tip

    To learn more about how to load LoRA weights, see the [LoRA](https://mindspore-lab.github.io/mindone/latest/diffusers/using-diffusers/loading_adapters#lora) loading guide.


::: mindone.diffusers.loaders.transformer_sd3.SD3Transformer2DLoadersMixin
