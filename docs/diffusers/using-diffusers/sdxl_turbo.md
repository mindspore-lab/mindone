<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Stable Diffusion XL Turbo

SDXL Turbo is an adversarial time-distilled [Stable Diffusion XL](https://huggingface.co/papers/2307.01952) (SDXL) model capable
of running inference in as little as 1 step.

This guide will show you how to use SDXL-Turbo for text-to-image and image-to-image.

Before you begin, make sure you have the following libraries installed:

```py
# uncomment to install the necessary libraries
#!pip install mindone transformers
```

## Load model checkpoints

Model weights may be stored in separate subfolders on the Hub or locally, in which case, you should use the [`from_pretrained`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/overview/#mindone.diffusers.DiffusionPipeline.from_pretrained) method:

```py
from mindone.diffusers import StableDiffusionXLPipeline
import mindspore as ms

pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/sdxl-turbo", mindspore_dtype=ms.float16, variant="fp16")
```

You can also use the [`from_single_file`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/loaders/single_file/#mindone.diffusers.loaders.single_file.FromSingleFileMixin.from_single_file) method to load a model checkpoint stored in a single file format (`.ckpt` or `.safetensors`) from the Hub or locally. For this loading method, you need to set `timestep_spacing="trailing"` (feel free to experiment with the other scheduler config values to get better results):

```py
from mindone.diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
import mindspore as ms

pipeline = StableDiffusionXLPipeline.from_single_file(
    "https://huggingface.co/stabilityai/sdxl-turbo/blob/main/sd_xl_turbo_1.0_fp16.safetensors",
    mindspore_dtype=ms.float16, variant="fp16")
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config, timestep_spacing="trailing")
```

## Text-to-image

For text-to-image, pass a text prompt. By default, SDXL Turbo generates a 512x512 image, and that resolution gives the best results. You can try setting the `height` and `width` parameters to 768x768 or 1024x1024, but you should expect quality degradations when doing so.

Make sure to set `guidance_scale` to 0.0 to disable, as the model was trained without it. A single inference step is enough to generate high quality images.
Increasing the number of steps to 2, 3 or 4 should improve image quality.

```py
from mindone.diffusers import StableDiffusionXLPipeline
import mindspore as ms

pipeline_text2image = StableDiffusionXLPipeline.from_pretrained("stabilityai/sdxl-turbo", mindspore_dtype=ms.float16, variant="fp16")

prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."

image = pipeline_text2image(prompt=prompt, guidance_scale=0.0, num_inference_steps=1)[0][0]
image
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
    <img src="https://github.com/user-attachments/assets/14ee3620-fc2a-402c-a283-7f33c6bb07d1" alt="generated image of a racoon in a robe"/>
</div>

## Image-to-image

For image-to-image generation, make sure that `num_inference_steps * strength` is larger or equal to 1.
The image-to-image pipeline will run for `int(num_inference_steps * strength)` steps, e.g. `0.5 * 2.0 = 1` step in
our example below.

```py
from mindone.diffusers import StableDiffusionXLImg2ImgPipeline
from mindone.diffusers.utils import load_image, make_image_grid
import mindspore as ms

# use from_pipe to avoid consuming additional memory when loading a checkpoint
pipeline_image2image = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/sdxl-turbo", mindspore_dtype=ms.float16, variant="fp16")

init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")
init_image = init_image.resize((512, 512))

prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"

image = pipeline_image2image(prompt, image=init_image, strength=0.5, guidance_scale=0.0, num_inference_steps=2)[0][0]
make_image_grid([init_image, image], rows=1, cols=2)
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
    <img src="https://github.com/user-attachments/assets/e690dfbd-4177-4d32-92c7-a4d800230aaa" alt="Image-to-image generation sample using SDXL Turbo"/>
</div>

## Speed-up SDXL Turbo even more

- When using the default VAE, keep it in `float32` to avoid costly `dtype` conversions before and after each generation. You only need to do this one before your first generation:

```py
pipe.upcast_vae()
```

As an alternative, you can also use a [16-bit VAE](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix) created by community member [`@madebyollin`](https://huggingface.co/madebyollin) that does not need to be upcasted to `float32`.
