<!-- # Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. -->

# Latte

![latte text-to-video](https://github.com/Vchitect/Latte/blob/52bc0029899babbd6e9250384c83d8ed2670ff7a/visuals/latte.gif?raw=true)

[Latte: Latent Diffusion Transformer for Video Generation](https://arxiv.org/abs/2401.03048) from Monash University, Shanghai AI Lab, Nanjing University, and Nanyang Technological University.

The abstract from the paper is:

*We propose a novel Latent Diffusion Transformer, namely Latte, for video generation. Latte first extracts spatio-temporal tokens from input videos and then adopts a series of Transformer blocks to model video distribution in the latent space. In order to model a substantial number of tokens extracted from videos, four efficient variants are introduced from the perspective of decomposing the spatial and temporal dimensions of input videos. To improve the quality of generated videos, we determine the best practices of Latte through rigorous experimental analysis, including video clip patch embedding, model variants, timestep-class information injection, temporal positional embedding, and learning strategies. Our comprehensive evaluation demonstrates that Latte achieves state-of-the-art performance across four standard video generation datasets, i.e., FaceForensics, SkyTimelapse, UCF101, and Taichi-HD. In addition, we extend Latte to text-to-video generation (T2V) task, where Latte achieves comparable results compared to recent T2V models. We strongly believe that Latte provides valuable insights for future research on incorporating Transformers into diffusion models for video generation.*

**Highlights**: Latte is a latent diffusion transformer proposed as a backbone for modeling different modalities (trained for text-to-video generation here). It achieves state-of-the-art performance across four standard video benchmarks - [FaceForensics](https://arxiv.org/abs/1803.09179), [SkyTimelapse](https://arxiv.org/abs/1709.07592), [UCF101](https://arxiv.org/abs/1212.0402) and [Taichi-HD](https://arxiv.org/abs/2003.00196). To prepare and download the datasets for evaluation, please refer to [this https URL](https://github.com/Vchitect/Latte/blob/main/docs/datasets_evaluation.md).

This pipeline was contributed by [maxin-cn](https://github.com/maxin-cn). The original codebase can be found [here](https://github.com/Vchitect/Latte). The original weights can be found under [hf.co/maxin-cn](https://huggingface.co/maxin-cn).

!!! tip

    Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers.md) to learn how to explore the tradeoff between scheduler speed and quality.

### Inference

```python
import mindspore as ms
from mindone.diffusers import LattePipeline

pipeline = LattePipeline.from_pretrained(
	"maxin-cn/Latte-1", mindspore_dtype=ms.float16
)

video = pipeline(prompt="A dog wearing sunglasses floating in space, surreal, nebulae in background")[0][0]
```

::: mindone.diffusers.LattePipeline
