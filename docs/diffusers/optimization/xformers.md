<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# xFormers

!!! warning

    ⚠️ MindONE currently only support `AttnProcessor`, Please check if only `AttnProcessor` is used in the pipeline before using `enable_xformers_memory_efficient_attention()`.

We recommend [xFormers](https://github.com/facebookresearch/xformers) for both inference and training. In our tests, the optimizations performed in the attention blocks allow for both faster speed and reduced memory consumption.

You can use `enable_xformers_memory_efficient_attention()` for faster inference and reduced memory consumption as shown in this [section](memory.md#memory-efficient-attention).
