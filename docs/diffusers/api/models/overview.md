<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Models

🤗 Diffusers provides pretrained models for popular algorithms and modules to create custom diffusion systems. The primary function of models is to denoise an input sample as modeled by the distribution p<sub>{&theta;}</sub>(x<sub>{t-1}</sub>|x<sub>{t}</sub>)

All models are built from the base [`ModelMixin`](overview.md#mindone.diffusers.ModelMixin) class which is a [`mindspore.nn.Cell`](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Cell.html?highlight=cell#mindspore.nn.Cell) providing basic functionality for saving and loading models, locally and from the Hugging Face Hub.

::: mindone.diffusers.ModelMixin

::: mindone.diffusers.utils.PushToHubMixin
