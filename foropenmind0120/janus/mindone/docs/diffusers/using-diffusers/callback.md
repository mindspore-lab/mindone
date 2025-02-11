<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Pipeline callbacks

The denoising loop of a pipeline can be modified with custom defined functions using the `callback_on_step_end` parameter. The callback function is executed at the end of each step, and modifies the pipeline attributes and variables for the next step. This is really useful for *dynamically* adjusting certain pipeline attributes or modifying tensor variables. This versatility allows for interesting use cases such as changing the prompt embeddings at each timestep, assigning different weights to the prompt embeddings, and editing the guidance scale. With callbacks, you can implement new features without modifying the underlying code!

This guide will demonstrate how callbacks work by a few features you can implement with them.

## Dynamic classifier-free guidance

Dynamic classifier-free guidance (CFG) is a feature that allows you to disable CFG after a certain number of inference steps which can help you save compute with minimal cost to performance. The callback function for this should have the following arguments:

- `pipeline` (or the pipeline instance) provides access to important properties such as `num_timesteps` and `guidance_scale`. You can modify these properties by updating the underlying attributes. For this example, you'll disable CFG by setting `pipeline._guidance_scale=0.0`.
- `step_index` and `timestep` tell you where you are in the denoising loop. Use `step_index` to turn off CFG after reaching 40% of `num_timesteps`.
- `callback_kwargs` is a dict that contains tensor variables you can modify during the denoising loop. It only includes variables specified in the `callback_on_step_end_tensor_inputs` argument, which is passed to the pipeline's `__call__` method. Different pipelines may use different sets of variables, so please check a pipeline's `_callback_tensor_inputs` attribute for the list of variables you can modify. Some common variables include `latents` and `prompt_embeds`. For this function, change the batch size of `prompt_embeds` after setting `guidance_scale=0.0` in order for it to work properly.

Your callback function should look something like this:

```python
def callback_dynamic_cfg(pipeline, step_index, timestep, callback_kwargs):
        # adjust the batch_size of prompt_embeds according to guidance_scale
        if step_index == int(pipeline.num_timesteps * 0.4):
                prompt_embeds = callback_kwargs["prompt_embeds"]
                prompt_embeds = prompt_embeds.chunk(2)[-1]

                # update guidance_scale and prompt_embeds
                pipeline._guidance_scale = 0.0
                callback_kwargs["prompt_embeds"] = prompt_embeds
        return callback_kwargs
```

Now, you can pass the callback function to the `callback_on_step_end` parameter and the `prompt_embeds` to `callback_on_step_end_tensor_inputs`.

```py
import mindspore as ms
from mindone.diffusers import StableDiffusionPipeline
import numpy as np

pipeline = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", mindspore_dtype=ms.float16)

prompt = "a photo of an astronaut riding a horse on mars"

generator = np.random.Generator(np.random.PCG64(1))
out = pipeline(
    prompt,
    generator=generator,
    callback_on_step_end=callback_dynamic_cfg,
    callback_on_step_end_tensor_inputs=['prompt_embeds']
)

out[0][0].save("out_custom_cfg.png")
```

## Interrupt the diffusion process

!!! tip

    The interruption callback is supported for text-to-image, image-to-image, and inpainting for the [StableDiffusionPipeline](../api/pipelines/stable_diffusion/overview.md) and [StableDiffusionXLPipeline](../api/pipelines/stable_diffusion/stable_diffusion_xl.md).

Stopping the diffusion process early is useful when building UIs that work with Diffusers because it allows users to stop the generation process if they're unhappy with the intermediate results. You can incorporate this into your pipeline with a callback.

This callback function should take the following arguments: `pipeline`, `i`, `t`, and `callback_kwargs` (this must be returned). Set the pipeline's `_interrupt` attribute to `True` to stop the diffusion process after a certain number of steps. You are also free to implement your own custom stopping logic inside the callback.

In this example, the diffusion process is stopped after 10 steps even though `num_inference_steps` is set to 50.

```python
from mindone.diffusers import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
num_inference_steps = 50

def interrupt_callback(pipeline, i, t, callback_kwargs):
    stop_idx = 10
    if i == stop_idx:
        pipeline._interrupt = True

    return callback_kwargs

pipeline(
    "A photo of a cat",
    num_inference_steps=num_inference_steps,
    callback_on_step_end=interrupt_callback,
)
```

## Display image after each generation step

!!! tip

    This tip was contributed by [asomoza](https://github.com/asomoza).

Display an image after each generation step by accessing and converting the latents after each step into an image. The latent space is compressed to 128x128, so the images are also 128x128 which is useful for a quick preview.

1. Use the function below to convert the SDXL latents (4 channels) to RGB tensors (3 channels) as explained in the [Explaining the SDXL latent space](https://huggingface.co/blog/TimothyAlexisVass/explaining-the-sdxl-latent-space) blog post.

```py
def latents_to_rgb(latents):
    weights = (
        (60, -60, 25, -70),
        (60,  -5, 15, -50),
        (60,  10, -5, -35)
    )

    def einsum(tensor1, tensor2):
        l, x, y = tensor1.shape[-3:]
        l, r = tensor2.shape
        res = ops.matmul(tensor2.transpose(1, 0), tensor1.view(*tensor1.shape[: -2], -1)).view(-1, r, x, y)
        return res

    weights_tensor = ops.t(ms.Tensor(weights, dtype=latents.dtype))
    biases_tensor = ms.Tensor((150, 140, 130), dtype=latents.dtype)
    rgb_tensor = einsum(latents, weights_tensor) + biases_tensor.unsqueeze(-1).unsqueeze(-1)
    image_array = rgb_tensor.clamp(0, 255)[0].to(ms.uint8).asnumpy()
    image_array = image_array.transpose(1, 2, 0)

    return Image.fromarray(image_array)
```

2. Create a function to decode and save the latents into an image.

```py
def decode_tensors(pipe, step, timestep, callback_kwargs):
    latents = callback_kwargs["latents"]

    image = latents_to_rgb(latents)
    image.save(f"{step}.png")

    return callback_kwargs
```

3. Pass the `decode_tensors` function to the `callback_on_step_end` parameter to decode the tensors after each step. You also need to specify what you want to modify in the `callback_on_step_end_tensor_inputs` parameter, which in this case are the latents.

```py
from mindone.diffusers import DiffusionPipeline
import mindspore as ms
from PIL import Image

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    mindspore_dtype=ms.float16,
    use_safetensors=True
)

image = pipeline(
    prompt="A croissant shaped like a cute bear.",
    negative_prompt="Deformed, ugly, bad anatomy",
    callback_on_step_end=decode_tensors,
    callback_on_step_end_tensor_inputs=["latents"],
)[0][0]
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div>
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/fe48d917-ccc9-4bce-ba5d-0adc2600c806"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">step 0</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/37268eef-e77a-433f-be39-b986fcc41754"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">step 19
    </figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/fe41f3a6-c11c-4691-a222-51a87f54b96a"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">step 29</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/f7940a8f-9f93-4d1f-a96a-6d0841c57179"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">step 39</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/c337d6ef-7c90-4767-bf87-a587c04f9289"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">step 49</figcaption>
  </div>
</div>
