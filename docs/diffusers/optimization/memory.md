<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Reduce memory usage

A barrier to using diffusion models is the large amount of memory required. To overcome this challenge, there are several memory-reducing techniques you can use to run even some of the largest models on Ascend. Some of these techniques can even be combined to further reduce memory usage.

!!! tip

    In many cases, optimizing for memory or speed leads to improved performance in the other, so you should try to optimize for both whenever you can. This guide focuses on minimizing memory usage, but you can also learn more about how to [Speed up inference](fp16.md).

## Memory-efficient attention

Recent work on optimizing bandwidth in the attention block has generated huge speed-ups and reductions in memory usage. The most recent type of memory-efficient attention is [Flash Attention](https://arxiv.org/abs/2205.14135) (you can check out the original code at [HazyResearch/flash-attention](https://github.com/HazyResearch/flash-attention)).

Now call [`enable_xformers_memory_efficient_attention`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/models/overview/#mindone.diffusers.ModelMixin.disable_xformers_memory_efficient_attention) on the pipeline:

```python
from mindone.diffusers import DiffusionPipeline
import mindspore as ms

pipe = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    mindspore_dtype=ms.float16,
    use_safetensors=True,
)

pipe.enable_xformers_memory_efficient_attention()

sample = pipe("a small cat")

# optional: You can disable it via
# pipe.disable_xformers_memory_efficient_attention()
```
