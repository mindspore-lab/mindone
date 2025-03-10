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

`AttnProcessors` will automatically invoke flash-attention for scaled dot-product attention calculations when the MindSpore version and hardware support it; otherwise, it will perform the original calculation according to the formula.

!!! tip

    It is important to note that we need to manually set whether to force data type conversion since the flash-attention operator in MindSpore only supports `float16` and `bfloat16` data-types. When the attention interface encounters data of an unsupported data type, if `force_cast_dtype` is not None, the function will forcibly convert the data to `force_cast_dtype` for computation and then restore it to the original data type afterward. If `force_cast_dtype` is None, it will fall back to the original attention calculation using mathematical formulas.

By default, `force_cast_dtype` is set to `mindspore.float16`, call [`set_flash_attention_force_cast_dtype`](../api/pipelines/overview.md#mindone.diffusers.DiffusionPipeline.set_flash_attention_force_cast_dtype) on the pipeline to change it, and you can alse call [`enable_flash_sdp(False)`](../api/pipelines/overview.md#mindone.diffusers.DiffusionPipeline.enable_flash_sdp) to disable flash-attention:

```python
from mindone.diffusers import DiffusionPipeline
import mindspore as ms

pipe = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    mindspore_dtype=ms.float16,
    use_safetensors=True,
)

# Optional: You can set `force_cast_dtype` for flash-attention on model-level or pipeline-level.
# Default: mindspore.float16
pipe.set_flash_attention_force_cast_dtype(force_cast_dtype=ms.bfloat16)
pipe.unet.set_flash_attention_force_cast_dtype(force_cast_dtype=None)

# Optional: You can disable flash-attention on model-level or pipeline-level:
# pipe.enable_flash_sdp(False)
# pipe.vae.enable_flash_sdp(True)

sample = pipe("a small cat")
```
