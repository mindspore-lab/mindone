<!--Copyright 2024 The HuggingFace Team and Tencent Hunyuan Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Hunyuan-DiT
![chinese elements understanding](https://github.com/gnobitab/diffusers-hunyuan/assets/1157982/39b99036-c3cb-4f16-bb1a-40ec25eda573)

[Hunyuan-DiT : A Powerful Multi-Resolution Diffusion Transformer with Fine-Grained Chinese Understanding](https://arxiv.org/abs/2405.08748) from Tencent Hunyuan.

The abstract from the paper is:

*We present Hunyuan-DiT, a text-to-image diffusion transformer with fine-grained understanding of both English and Chinese. To construct Hunyuan-DiT, we carefully design the transformer structure, text encoder, and positional encoding. We also build from scratch a whole data pipeline to update and evaluate data for iterative model optimization. For fine-grained language understanding, we train a Multimodal Large Language Model to refine the captions of the images. Finally, Hunyuan-DiT can perform multi-turn multimodal dialogue with users, generating and refining images according to the context. Through our holistic human evaluation protocol with more than 50 professional human evaluators, Hunyuan-DiT sets a new state-of-the-art in Chinese-to-image generation compared with other open-source models.*


You can find the original codebase at [Tencent/HunyuanDiT](https://github.com/Tencent/HunyuanDiT) and all the available checkpoints at [Tencent-Hunyuan](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT).

**Highlights**: HunyuanDiT supports Chinese/English-to-image, multi-resolution generation.

HunyuanDiT has the following components:
* It uses a diffusion transformer as the backbone
* It combines two text encoders, a bilingual CLIP and a multilingual T5 encoder

!!! tip

    Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers.md) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading.md) section to learn how to efficiently load the same components into multiple pipelines.

!!! tip

    You can further improve generation quality by passing the generated image from [`HungyuanDiTPipeline`](#mindone.diffusers.HunyuanDiTPipeline) to the [SDXL refiner](../../using-diffusers/sdxl.md#base-to-refiner-model) model.

## Optimization

You can optimize the pipeline's runtime and memory consumption with mindspore graph mode and feed-forward chunking. To learn about other optimization methods, check out the [Speed up inference](../../optimization/fp16.md) and [Reduce memory usage](../../optimization/memory.md) guides.

### Inference

First, load the pipeline:

```python
from mindone.diffusers import HunyuanDiTPipeline
import mindspore as ms

ms.set_context(mode=1)
pipeline = HunyuanDiTPipeline.from_pretrained(
	"Tencent-Hunyuan/HunyuanDiT-Diffusers", mindspore_dtype=ms.float16
)

image = pipeline(prompt="一个宇航员在骑马")[0][0]
image.save("image.png")
```

::: mindone.diffusers.HunyuanDiTPipeline
