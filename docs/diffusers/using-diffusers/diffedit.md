<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# DiffEdit

Image editing typically requires providing a mask of the area to be edited. DiffEdit automatically generates the mask for you based on a text query, making it easier overall to create a mask without image editing software. The DiffEdit algorithm works in three steps:

1. the diffusion model denoises an image conditioned on some query text and reference text which produces different noise estimates for different areas of the image; the difference is used to infer a mask to identify which area of the image needs to be changed to match the query text
2. the input image is encoded into latent space with DDIM
3. the latents are decoded with the diffusion model conditioned on the text query, using the mask as a guide such that pixels outside the mask remain the same as in the input image

This guide will show you how to use DiffEdit to edit images without manually creating a mask.

Before you begin, make sure you have the following libraries installed:

```py
# uncomment to install the necessary libraries
#!pip install mindone transformers
```

The [`StableDiffusionDiffEditPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/diffedit/#mindone.diffusers.StableDiffusionDiffEditPipeline) requires an image mask and a set of partially inverted latents. The image mask is generated from the [`generate_mask`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/diffedit/#mindone.diffusers.StableDiffusionDiffEditPipeline.generate_mask) function, and includes two parameters, `source_prompt` and `target_prompt`. These parameters determine what to edit in the image. For example, if you want to change a bowl of *fruits* to a bowl of *pears*, then:

```py
source_prompt = "a bowl of fruits"
target_prompt = "a bowl of pears"
```

The partially inverted latents are generated from the [`invert`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/diffedit/#mindone.diffusers.StableDiffusionDiffEditPipeline.invert) function, and it is generally a good idea to include a `prompt` or *caption* describing the image to help guide the inverse latent sampling process. The caption can often be your `source_prompt`, but feel free to experiment with other text descriptions!

Let's load the pipeline, scheduler, inverse scheduler, and enable some optimizations to reduce memory usage:

```py
import mindspore as ms
from mindone.diffusers import DDIMScheduler, DDIMInverseScheduler, StableDiffusionDiffEditPipeline

pipeline = StableDiffusionDiffEditPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    mindspore_dtype=ms.float16,
    safety_checker=None,
    use_safetensors=True,
)
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
```

Load the image to edit:

```py
from mindone.diffusers.utils import load_image, make_image_grid

img_url = "https://github.com/Xiang-cd/DiffEdit-stable-diffusion/raw/main/assets/origin.png"
raw_image = load_image(img_url).resize((768, 768))
raw_image
```

Use the [`generate_mask`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/diffedit/#mindone.diffusers.StableDiffusionDiffEditPipeline.generate_mask) function to generate the image mask. You'll need to pass it the `source_prompt` and `target_prompt` to specify what to edit in the image:

```py
from PIL import Image

source_prompt = "a bowl of fruits"
target_prompt = "a basket of pears"
mask_image = pipeline.generate_mask(
    image=raw_image,
    source_prompt=source_prompt,
    target_prompt=target_prompt,
)
Image.fromarray((mask_image.squeeze()*255).astype("uint8"), "L").resize((768, 768))
```

Next, create the inverted latents and pass it a caption describing the image:

```py
inv_latents = pipeline.invert(prompt=source_prompt, image=raw_image)[0]
```

Finally, pass the image mask and inverted latents to the pipeline. The `target_prompt` becomes the `prompt` now, and the `source_prompt` is used as the `negative_prompt`:

```py
output_image = pipeline(
    prompt=target_prompt,
    mask_image=mask_image,
    image_latents=inv_latents,
    negative_prompt=source_prompt,
)[0][0]
mask_image = Image.fromarray((mask_image.squeeze()*255).astype("uint8"), "L").resize((768, 768))
make_image_grid([raw_image, mask_image, output_image], rows=1, cols=3)
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div>
    <img class="rounded-xl" src="https://github.com/Xiang-cd/DiffEdit-stable-diffusion/raw/main/assets/origin.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">original image</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/25fd1779-f673-4cb0-8441-c0b9dd50e608"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">edited image</figcaption>
  </div>
</div>
