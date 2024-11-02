<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Reproducible pipelines

Diffusion models are inherently random which is what allows it to generate different outputs every time it is run. But there are certain times when you want to generate the same output every time, like when you're testing, replicating results, and even [improving image quality](#deterministic-batch-generation). While you can't expect to get identical results across platforms, you can expect reproducible results across releases and platforms within a certain tolerance range (though even this may vary).

This guide will show you how to control randomness for deterministic generation on a Ascend.

## Control randomness

During inference, pipelines rely heavily on random sampling operations which include creating the
Gaussian noise tensors to denoise and adding noise to the scheduling step.

Take a look at the tensor values in the [`DDIMPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/ddim/#mindone.diffusers.DDIMPipeline) after two inference steps.

```python
from mindone.diffusers import DDIMPipeline
import numpy as np

ddim = DDIMPipeline.from_pretrained( "google/ddpm-cifar10-32", use_safetensors=True)
image = ddim(num_inference_steps=2, output_type="np")[0]
print(np.abs(image).sum())
```

Running the code above prints one value, but if you run it again you get a different value.

Each time the pipeline is run, [numpy.random.Generator.standard_normal](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.standard_normal.html) uses a different random seed to create the Gaussian noise tensors. This leads to a different result each time it is run and enables the diffusion pipeline to generate a different random image each time.

But if you need to reliably generate the same image, Diffusers has a [`randn_tensor`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/utilities/#mindone.diffusers.utils.mindspore_utils.randn_tensor) function for creating random noise using numpy, and then convert the array to tensor. The [`randn_tensor`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/utilities/#mindone.diffusers.utils.mindspore_utils.randn_tensor) function is used everywhere inside the pipeline. Now you can call [numpy.random.Generator](https://numpy.org/doc/stable/reference/random/generator.html) which automatically creates a `Generator` that can be passed to the pipeline.

```python
import numpy as np
from mindone.diffusers import DDIMPipeline

ddim = DDIMPipeline.from_pretrained("google/ddpm-cifar10-32", use_safetensors=True)
generator = np.random.Generator(np.random.PCG64(0))
image = ddim(num_inference_steps=2, output_type="np", generator=generator)[0]
print(np.abs(image).sum())
```

Finally, more complex pipelines such as [`UnCLIPPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/unclip/#mindone.diffusers.UnCLIPPipeline), are often extremely
susceptible to precision error propagation. You'll need to use
exactly the same hardware and MindSpore version for full reproducibility.

## Deterministic batch generation

A practical application of creating reproducible pipelines is *deterministic batch generation*. You generate a batch of images and select one image to improve with a more detailed prompt. The main idea is to pass a list of [Generator's](https://numpy.org/doc/stable/reference/random/generator.html) to the pipeline and tie each `Generator` to a seed so you can reuse it.

Let's use the [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) checkpoint and generate a batch of images.

```py
import mindspore as ms
from mindone.diffusers import DiffusionPipeline
from mindone.diffusers.utils import make_image_grid
import numpy as np

pipeline = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", mindspore_dtype=ms.float16, use_safetensors=True
)
```

Define four different `Generator`s and assign each `Generator` a seed (`0` to `3`). Then generate a batch of images and pick one to iterate on.

!!! warning

    Use a list comprehension that iterates over the batch size specified in `range()` to create a unique `Generator` object for each image in the batch. If you multiply the `Generator` by the batch size integer, it only creates *one* `Generator` object that is used sequentially for each image in the batch.

    ```py
    [np.random.Generator(np.random.PCG64(seed))] * 4
    ```

```python
generator = [np.random.Generator(np.random.PCG64(i)) for i in range(4)]
prompt = "Labrador in the style of Vermeer"
images = pipeline(prompt, generator=generator, num_images_per_prompt=4)[0]
make_image_grid(images, rows=2, cols=2)
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
    <img src="https://github.com/user-attachments/assets/5c26aaae-af49-4a0c-9af2-e12e76c89bee"/>
</div>

Let's improve the first image (you can choose any image you want) which corresponds to the `Generator` with seed `0`. Add some additional text to your prompt and then make sure you reuse the same `Generator` with seed `0`. All the generated images should resemble the first image.

```python
prompt = [prompt + t for t in [", highly realistic", ", artsy", ", trending", ", colorful"]]
generator = [np.random.Generator(np.random.PCG64(0)) for i in range(4)]
images = pipeline(prompt, generator=generator)[0]
make_image_grid(images, rows=2, cols=2)
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
    <img src="https://github.com/user-attachments/assets/b44ad73f-7505-4339-b898-3a7e21a863d4"/>
</div>
