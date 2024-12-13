<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Tiny AutoEncoder

Tiny AutoEncoder for Stable Diffusion (TAESD) was introduced in [madebyollin/taesd](https://github.com/madebyollin/taesd) by Ollin Boer Bohan. It is a tiny distilled version of Stable Diffusion's VAE that can quickly decode the latents in a [`StableDiffusionPipeline`](../pipelines/stable_diffusion/text2img.md) or [`StableDiffusionXLPipeline`](../pipelines/stable_diffusion/stable_diffusion_xl.md) almost instantly.

To use with Stable Diffusion v-2.1:

```python
import mindspore as ms
from mindone.diffusers import DiffusionPipeline, AutoencoderTiny

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base", mindspore_dtype=ms.float16
)
pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", mindspore_dtype=ms.float16)

prompt = "slice of delicious New York-style berry cheesecake"
image = pipe(prompt, num_inference_steps=25)[0][0]
image
```

To use with Stable Diffusion XL 1.0

```python
import mindspore as ms
from mindone.diffusers import DiffusionPipeline, AutoencoderTiny

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", mindspore_dtype=ms.float16
)
pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", mindspore_dtype=ms.float16)

prompt = "slice of delicious New York-style berry cheesecake"
image = pipe(prompt, num_inference_steps=25)[0][0]
image
```

::: mindone.diffusers.AutoencoderTiny

::: mindone.diffusers.models.autoencoders.autoencoder_tiny.AutoencoderTinyOutput
