<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Load LoRAs for inference

There are many adapter types (with [LoRAs](https://huggingface.co/docs/peft/conceptual_guides/adapter#low-rank-adaptation-lora) being the most popular) trained in different styles to achieve different effects. You can even combine multiple adapters to create new and unique images.

In this tutorial, you'll learn how to easily load and manage adapters for inference with the 🤗 [PEFT](https://huggingface.co/docs/peft/index) integration in 🤗 Diffusers. You'll use LoRA as the main adapter technique, so you'll see the terms LoRA and adapter used interchangeably.

Let's first install all the required libraries.

```bash
!pip install transformers mindone
```

Now, load a pipeline with a [Stable Diffusion XL (SDXL)](../api/pipelines/stable_diffusion/stable_diffusion_xl.md) checkpoint:

```python
from mindone.diffusers import DiffusionPipeline
import mindspore
import numpy as np

pipe_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(pipe_id, mindspore_dtype=mindspore.float16)
```

Next, load a [CiroN2022/toy-face](https://huggingface.co/CiroN2022/toy-face) adapter with the [`load_lora_weights`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/loaders/lora#mindone.diffusers.loaders.lora.StableDiffusionXLLoraLoaderMixin.load_lora_weights) method. With the 🤗 PEFT integration, you can assign a specific `adapter_name` to the checkpoint, which lets you easily switch between different LoRA checkpoints. Let's call this adapter `"toy"`.

```python
pipe.load_lora_weights("CiroN2022/toy-face", weight_name="toy_face_sdxl.safetensors", adapter_name="toy")
```

Make sure to include the token `toy_face` in the prompt and then you can perform inference:

```python
prompt = "toy_face of a hacker with a hoodie"

lora_scale = 0.9
image = pipe(
    prompt, num_inference_steps=30, cross_attention_kwargs={"scale": lora_scale}, generator=np.random.Generator(np.random.PCG64(0))
)[0][0]
image
```

![toy-face](https://github.com/user-attachments/assets/c1796924-ee98-49c4-829b-887874ed7f3d)

With the `adapter_name` parameter, it is really easy to use another adapter for inference! Load the [nerijs/pixel-art-xl](https://huggingface.co/nerijs/pixel-art-xl) adapter that has been fine-tuned to generate pixel art images and call it `"pixel"`.

The pipeline automatically sets the first loaded adapter (`"toy"`) as the active adapter, but you can activate the `"pixel"` adapter with the [`set_adapters`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/loaders/unet/#mindone.diffusers.loaders.unet.UNet2DConditionLoadersMixin.set_adapters) method:

```python
pipe.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors", adapter_name="pixel")
pipe.set_adapters("pixel")
```

Make sure you include the token `pixel art` in your prompt to generate a pixel art image:

```python
prompt = "a hacker with a hoodie, pixel art"
image = pipe(
    prompt, num_inference_steps=30, cross_attention_kwargs={"scale": lora_scale}, generator=np.random.Generator(np.random.PCG64(0))
)[0][0]
image
```

![pixel-art](https://github.com/user-attachments/assets/fa0e31c8-787e-42dd-8027-a8be89884863)

## Merge adapters

You can also merge different adapter checkpoints for inference to blend their styles together.

Once again, use the [`set_adapters`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/loaders/unet/#mindone.diffusers.loaders.unet.UNet2DConditionLoadersMixin.set_adapters) method to activate the `pixel` and `toy` adapters and specify the weights for how they should be merged.

```python
pipe.set_adapters(["pixel", "toy"], adapter_weights=[0.5, 1.0])
```

!!! tip

    LoRA checkpoints in the diffusion community are almost always obtained with [DreamBooth](https://mindspore-lab.github.io/mindone/latest/diffusers/training/dreambooth/). DreamBooth training often relies on "trigger" words in the input text prompts in order for the generation results to look as expected. When you combine multiple LoRA checkpoints, it's important to ensure the trigger words for the corresponding LoRA checkpoints are present in the input text prompts.

Remember to use the trigger words for [CiroN2022/toy-face](https://hf.co/CiroN2022/toy-face) and [nerijs/pixel-art-xl](https://hf.co/nerijs/pixel-art-xl) (these are found in their repositories) in the prompt to generate an image.

```python
prompt = "toy_face of a hacker with a hoodie, pixel art"
image = pipe(
    prompt, num_inference_steps=30, cross_attention_kwargs={"scale": 1.0}, generator=np.random.Generator(np.random.PCG64(0))
)[0][0]
image
```

![toy-face-pixel-art](https://github.com/user-attachments/assets/ee327669-3c18-4293-8eaa-0bbd93afbe02)

Impressive! As you can see, the model generated an image that mixed the characteristics of both adapters.

!!! tip

    Through its PEFT integration, Diffusers also offers more efficient merging methods which you can learn about in the [Merge LoRAs](../using-diffusers/merge_loras.md) guide!

To return to only using one adapter, use the [`set_adapters`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/loaders/unet/#mindone.diffusers.loaders.unet.UNet2DConditionLoadersMixin.set_adapters) method to activate the `"toy"` adapter:

```python
pipe.set_adapters("toy")

prompt = "toy_face of a hacker with a hoodie"
lora_scale = 0.9
image = pipe(
    prompt, num_inference_steps=30, cross_attention_kwargs={"scale": lora_scale}, generator=np.random.Generator(np.random.PCG64(0))
)[0][0]
image
```

Or to disable all adapters entirely, use the [`disable_lora`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/loaders/unet/#mindone.diffusers.loaders.unet.UNet2DConditionLoadersMixin.disable_lora) method to return the base model.

```python
pipe.disable_lora()

prompt = "toy_face of a hacker with a hoodie"
image = pipe(prompt, num_inference_steps=30, generator=np.random.Generator(np.random.PCG64(0)))[0][0]
image
```

![no-lora](https://github.com/user-attachments/assets/c17dc29e-4a5f-4243-b5f6-18b3dc05e570)

### Customize adapters strength
For even more customization, you can control how strongly the adapter affects each part of the pipeline. For this, pass a dictionary with the control strengths (called "scales") to [`set_adapters`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/loaders/unet/#mindone.diffusers.loaders.unet.UNet2DConditionLoadersMixin.set_adapters).

For example, here's how you can turn on the adapter for the `down` parts, but turn it off for the `mid` and `up` parts:
```python
pipe.enable_lora()  # enable lora again, after we disabled it above
prompt = "toy_face of a hacker with a hoodie, pixel art"
adapter_weight_scales = { "unet": { "down": 1, "mid": 0, "up": 0} }
pipe.set_adapters("pixel", adapter_weight_scales)
image = pipe(prompt, num_inference_steps=30, generator=np.random.Generator(np.random.PCG64(0)))[0][0]
image
```

![block-lora-text-and-down](https://github.com/user-attachments/assets/97822bc2-643b-44bd-837d-94b3f309cf20)

Let's see how turning off the `down` part and turning on the `mid` and `up` part respectively changes the image.
```python
adapter_weight_scales = { "unet": { "down": 0, "mid": 1, "up": 0} }
pipe.set_adapters("pixel", adapter_weight_scales)
image = pipe(prompt, num_inference_steps=30, generator=np.random.Generator(np.random.PCG64(0)))[0][0]
image
```

![block-lora-text-and-mid](https://github.com/user-attachments/assets/86469036-8492-4cd3-bed7-493cf0c28da2)

```python
adapter_weight_scales = { "unet": { "down": 0, "mid": 0, "up": 1} }
pipe.set_adapters("pixel", adapter_weight_scales)
image = pipe(prompt, num_inference_steps=30, generator=np.random.Generator(np.random.PCG64(0)))[0][0]
image
```

![block-lora-text-and-up](https://github.com/user-attachments/assets/b5d80d23-e463-41f3-a9b6-6a5f8f55a7b8)

Looks cool!

This is a really powerful feature. You can use it to control the adapter strengths down to per-transformer level. And you can even use it for multiple adapters.
```python
adapter_weight_scales_toy = 0.5
adapter_weight_scales_pixel = {
    "unet": {
        "down": 0.9,  # all transformers in the down-part will use scale 0.9
        # "mid"  # because, in this example, "mid" is not given, all transformers in the mid part will use the default scale 1.0
        "up": {
            "block_0": 0.6,  # all 3 transformers in the 0th block in the up-part will use scale 0.6
            "block_1": [0.4, 0.8, 1.0],  # the 3 transformers in the 1st block in the up-part will use scales 0.4, 0.8 and 1.0 respectively
        }
    }
}
pipe.set_adapters(["toy", "pixel"], [adapter_weight_scales_toy, adapter_weight_scales_pixel])
image = pipe(prompt, num_inference_steps=30, generator=np.random.Generator(np.random.PCG64(0)))[0][0]
image
```

![block-lora-mixed](https://github.com/user-attachments/assets/c4ffe4dc-6bf9-48e1-9a9e-4ca35c24a7a1)

## Manage active adapters

You have attached multiple adapters in this tutorial, and if you're feeling a bit lost on what adapters have been attached to the pipeline's components, use the [`get_active_adapters`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/loaders/lora/#mindone.diffusers.loaders.lora.LoraLoaderMixin.get_active_adapters) method to check the list of active adapters:

```py
active_adapters = pipe.get_active_adapters()
active_adapters
['toy', 'pixel']
```

You can also get the active adapters of each pipeline component with [`~diffusers.loaders.LoraLoaderMixin.get_list_adapters`]:

```py
list_adapters_component_wise = pipe.get_list_adapters()
list_adapters_component_wise
{"unet": ['toy', 'pixel']}
```
