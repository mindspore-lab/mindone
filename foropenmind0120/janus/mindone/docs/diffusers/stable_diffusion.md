<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Effective and efficient diffusion

Getting the [`DiffusionPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/overview/#mindone.diffusers.DiffusionPipeline) to generate images in a certain style or include what you want can be tricky. Often times, you have to run the [`DiffusionPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/overview/#mindone.diffusers.DiffusionPipeline) several times before you end up with an image you're happy with. But generating something out of nothing is a computationally intensive process, especially if you're running inference over and over again.

This is why it's important to get the most *computational* (speed) and *memory* (NPU vRAM) efficiency from the pipeline to reduce the time between inference cycles so you can iterate faster.

This tutorial walks you through how to generate faster and better with the [`DiffusionPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/overview/#mindone.diffusers.DiffusionPipeline).

Begin by loading the [`stable-diffusion-v1-5/stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) model:

```python
from mindone.diffusers import DiffusionPipeline

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
pipeline = DiffusionPipeline.from_pretrained(model_id, use_safetensors=True)
```

The example prompt you'll use is a portrait of an old warrior chief, but feel free to use your own prompt:

```python
prompt = "portrait photo of a old warrior chief"
```

## Speed

One of the simplest ways to speed up inference is to place the pipeline on a NPU the same way you would with any Mindspore cell.
That is, do nothing! MindSpore will automatically take care of model placement, so you don't need to:

```diff
- pipeline = pipeline.to("cuda")
```

To make sure you can use the same image and improve on it, use a [`Generator`](https://numpy.org/doc/stable/reference/random/generator.html) and set a seed for [reproducibility](./using-diffusers/reusing_seeds.md):

```python
import numpy as np

generator = np.random.Generator(np.random.PCG64(seed=0))
```

Now you can generate an image:

```python
image = pipeline(prompt, generator=generator)[0][0]
image
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
    <img src="https://github.com/user-attachments/assets/67b06273-9081-4b4f-a31f-585b23f70f27">
</div>

This process took ~5.6 seconds on a Ascend 910B in Graph mode. By default, the [`DiffusionPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/overview/#mindone.diffusers.DiffusionPipeline) runs inference with full `float32` precision for 50 inference steps. You can speed this up by switching to a lower precision like `float16` or running fewer inference steps.

Let's start by loading the model in `float16` and generate an image:

```python
import mindspore

pipeline = DiffusionPipeline.from_pretrained(model_id, mindspore_dtype=mindspore.float16, use_safetensors=True)
generator = np.random.Generator(np.random.PCG64(seed=0))
image = pipeline(prompt, generator=generator)[0][0]
image
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
    <img src="https://github.com/user-attachments/assets/1fc7d859-4164-4eff-841a-57b073cd8bb3">
</div>

This time, it only took ~3.8 seconds to generate the image, which is almost 1.5x faster than before!

!!! tip

    💡 We strongly suggest always running your pipelines in `float16`, and so far, we've rarely seen any degradation in output quality.

Another option is to reduce the number of inference steps. Choosing a more efficient scheduler could help decrease the number of steps without sacrificing output quality. You can find which schedulers are compatible with the current model in the [`DiffusionPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/overview/#mindone.diffusers.DiffusionPipeline) by calling the `compatibles` method:

```python
pipeline.scheduler.compatibles
[
    <class 'mindone.diffusers.schedulers.scheduling_ddpm.DDPMScheduler'>,
    <class 'mindone.diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler'>,
    <class 'mindone.diffusers.schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteScheduler'>,
    <class 'mindone.diffusers.schedulers.scheduling_lms_discrete.LMSDiscreteScheduler'>,
    <class 'mindone.diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler'>,
    <class 'mindone.diffusers.schedulers.scheduling_k_dpm_2_discrete.KDPM2DiscreteScheduler'>,
    <class 'mindone.diffusers.schedulers.scheduling_edm_euler.EDMEulerScheduler'>,
    <class 'mindone.diffusers.schedulers.scheduling_dpmsolver_singlestep.DPMSolverSinglestepScheduler'>,
    <class 'mindone.diffusers.schedulers.scheduling_pndm.PNDMScheduler'>,
    <class 'mindone.diffusers.schedulers.scheduling_heun_discrete.HeunDiscreteScheduler'>,
    <class 'mindone.diffusers.schedulers.scheduling_unipc_multistep.UniPCMultistepScheduler'>,
    <class 'mindone.diffusers.schedulers.scheduling_ddim.DDIMScheduler'>,
    <class 'mindone.diffusers.schedulers.scheduling_k_dpm_2_ancestral_discrete.KDPM2AncestralDiscreteScheduler'>,
    <class 'mindone.diffusers.schedulers.scheduling_deis_multistep.DEISMultistepScheduler'>
]
```

The Stable Diffusion model uses the [`PNDMScheduler`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/schedulers/pndm/#mindone.diffusers.PNDMScheduler) by default which usually requires ~50 inference steps, but more performant schedulers like [`DPMSolverMultistepScheduler`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/schedulers/multistep_dpm_solver/#mindone.diffusers.DPMSolverMultistepScheduler), require only ~20 or 25 inference steps. Use the [`from_config`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/configuration/#mindone.diffusers.configuration_utils.ConfigMixin.from_config) method to load a new scheduler:

```python
from mindone.diffusers import DPMSolverMultistepScheduler

pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
```

Now set the `num_inference_steps` to 20:

```python
generator = np.random.Generator(np.random.PCG64(seed=0))
image = pipeline(prompt, generator=generator, num_inference_steps=20)[0][0]
image
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
    <img src="https://github.com/user-attachments/assets/2461137f-8a3d-4f66-8190-d1ac398b8c86">
</div>

Great, you've managed to cut the inference time to just 4 seconds! ⚡️

## Memory

The other key to improving pipeline performance is consuming less memory, which indirectly implies more speed, since you're often trying to maximize the number of images generated per second. The easiest way to see how many images you can generate at once is to try out different batch sizes until you get an `OutOfMemoryError` (OOM).

Create a function that'll generate a batch of images from a list of prompts and `Generators`. Make sure to assign each `Generator` a seed so you can reuse it if it produces a good result.

```python
def get_inputs(batch_size=1):
    generator = [np.random.Generator(np.random.PCG64(seed=i)) for i in range(batch_size)]
    prompts = batch_size * [prompt]
    num_inference_steps = 20

    return {"prompt": prompts, "generator": generator, "num_inference_steps": num_inference_steps}
```

Start with `batch_size=4` and see how much memory you've consumed:

```python
from mindone.diffusers.utils import make_image_grid

images = pipeline(**get_inputs(batch_size=4))[0]
make_image_grid(images, 2, 2)
```

Now try increasing the `batch_size` to 8!

```python
images = pipeline(**get_inputs(batch_size=8))[0]
make_image_grid(images, rows=2, cols=4)
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
    <img src="https://github.com/user-attachments/assets/5028a23d-7acd-4bb0-8633-38f8371eb393">
</div>

Whereas before you couldn't even generate a batch of 4 images, now you can generate a batch of 8 images at ~1.6 seconds per image! This is probably the fastest you can go on a Ascend 910B without sacrificing quality.

## Quality

In the last two sections, you learned how to optimize the speed of your pipeline by using `fp16`, reducing the number of inference steps by using a more performant scheduler, and enabling attention slicing to reduce memory consumption. Now you're going to focus on how to improve the quality of generated images.

### Better checkpoints

The most obvious step is to use better checkpoints. The Stable Diffusion model is a good starting point, and since its official launch, several improved versions have also been released. However, using a newer version doesn't automatically mean you'll get better results. You'll still have to experiment with different checkpoints yourself, and do a little research (such as using [negative prompts](https://minimaxir.com/2022/11/stable-diffusion-negative-prompt/)) to get the best results.

As the field grows, there are more and more high-quality checkpoints finetuned to produce certain styles. Try exploring the [Hub](https://huggingface.co/models?library=diffusers&sort=downloads) and [Diffusers Gallery](https://huggingface.co/spaces/huggingface-projects/diffusers-gallery) to find one you're interested in!

### Better pipeline components

You can also try replacing the current pipeline components with a newer version. Let's try loading the latest [autoencoder](https://huggingface.co/stabilityai/stable-diffusion-2-1/tree/main/vae) from Stability AI into the pipeline, and generate some images:

```python
from mindone.diffusers import AutoencoderKL

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", mindspore_dtype=mindspore.float16)
pipeline.vae = vae
images = pipeline(**get_inputs(batch_size=8))[0]
make_image_grid(images, rows=2, cols=4)
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
    <img src="https://github.com/user-attachments/assets/377b1522-58e8-41a4-96e7-6a8511dd1d05">
</div>

### Better prompt engineering

The text prompt you use to generate an image is super important, so much so that it is called *prompt engineering*. Some considerations to keep during prompt engineering are:

- How is the image or similar images of the one I want to generate stored on the internet?
- What additional detail can I give that steers the model towards the style I want?

With this in mind, let's improve the prompt to include color and higher quality details:

```python
prompt += ", tribal panther make up, blue on red, side profile, looking away, serious eyes"
prompt += " 50mm portrait photography, hard rim lighting photography--beta --ar 2:3  --beta --upbeta"
```

Generate a batch of images with the new prompt:

```python
images = pipeline(**get_inputs(batch_size=8))[0]
make_image_grid(images, rows=2, cols=4)
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
    <img src="https://github.com/user-attachments/assets/90bb69c1-0174-4d05-b47f-83b16281f8a1">
</div>

Pretty impressive! Let's tweak the second image - corresponding to the `Generator` with a seed of `1` - a bit more by adding some text about the age of the subject:

```python
prompts = [
    "portrait photo of the oldest warrior chief, tribal panther make up, blue on red, side profile, looking away, serious eyes 50mm portrait photography, hard rim lighting photography--beta --ar 2:3  --beta --upbeta",
    "portrait photo of a old warrior chief, tribal panther make up, blue on red, side profile, looking away, serious eyes 50mm portrait photography, hard rim lighting photography--beta --ar 2:3  --beta --upbeta",
    "portrait photo of a warrior chief, tribal panther make up, blue on red, side profile, looking away, serious eyes 50mm portrait photography, hard rim lighting photography--beta --ar 2:3  --beta --upbeta",
    "portrait photo of a young warrior chief, tribal panther make up, blue on red, side profile, looking away, serious eyes 50mm portrait photography, hard rim lighting photography--beta --ar 2:3  --beta --upbeta",
]

generator = [np.random.Generator(np.random.PCG64(seed=1)) for _ in range(len(prompts))]
images = pipeline(prompt=prompts, generator=generator, num_inference_steps=25)[0]
make_image_grid(images, 2, 2)
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
    <img src="https://github.com/user-attachments/assets/7dab5585-e3c8-4b56-a421-ff8d49b4a7f2">
</div>

## Next steps

In this tutorial, you learned how to optimize a [`DiffusionPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/overview/#mindone.diffusers.DiffusionPipeline) for computational and memory efficiency as well as improving the quality of generated outputs. If you're interested in making your pipeline even faster, take a look at the following resources:

- We recommend you use [xFormers](./optimization/xformers.md). Its memory-efficient attention mechanism works great for faster speed and reduced memory consumption.
- Other optimization techniques, such as model offloading, are covered in [this guide](./optimization/fp16.md).
