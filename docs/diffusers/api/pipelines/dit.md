<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# DiT

[Scalable Diffusion Models with Transformers](https://huggingface.co/papers/2212.09748) (DiT) is by William Peebles and Saining Xie.

The abstract from the paper is:

*We explore a new class of diffusion models based on the transformer architecture. We train latent diffusion models of images, replacing the commonly-used U-Net backbone with a transformer that operates on latent patches. We analyze the scalability of our Diffusion Transformers (DiTs) through the lens of forward pass complexity as measured by Gflops. We find that DiTs with higher Gflops -- through increased transformer depth/width or increased number of input tokens -- consistently have lower FID. In addition to possessing good scalability properties, our largest DiT-XL/2 models outperform all prior diffusion models on the class-conditional ImageNet 512x512 and 256x256 benchmarks, achieving a state-of-the-art FID of 2.27 on the latter.*

The original codebase can be found at [facebookresearch/dit](https://github.com/facebookresearch/dit).

!!! tip

    Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers.md) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading.md) section to learn how to efficiently load the same components into multiple pipelines.

# Usage Example

```python
import mindspore as ms
from mindone.diffusers import DiTPipeline, DPMSolverMultistepScheduler

pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", mindspore_dtype=ms.float16, revision="refs/pr/1")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

pipe.labels
word = ["white shark", "umbrella"]

class_ids = pipe.get_label_ids(word)

output = pipe(class_labels=class_ids, num_inference_steps=25)

image = output[0][0]
image.save("image.png")
```

::: mindone.diffusers.DiTPipeline

::: mindone.diffusers.pipelines.ImagePipelineOutput
